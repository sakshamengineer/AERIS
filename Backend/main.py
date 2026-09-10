from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from datetime import datetime
from collections import Counter
from threading import Thread, Event
import shutil
import uuid
import json
import traceback
import rasterio
from PIL import Image
from agent.controller import SatQueryController
from agent.modality import detect_modality, check_optical_sar_pair
from agent.router import route_query, get_task_description
from preprocessing.validation import validate_file
from preprocessing.preview import generate_display_image
from config import SUPPORTED_GEOSPATIAL_FORMATS, SUPPORTED_IMAGE_FORMATS

app = FastAPI(
    title="SatQuery AI API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app.mount(
    "/uploads",
    StaticFiles(directory=UPLOAD_DIR),
    name="uploads"
)

app.mount(
    "/outputs",
    StaticFiles(directory=OUTPUT_DIR),
    name="outputs"
)

controller = SatQueryController()

analysis_history = []
analysis_jobs = {}

# How often (seconds) the background progress-ticker nudges the bar
# forward while controller.analyze() is still running. Keep this
# short so the frontend (which polls every ~1s) sees smooth motion.
TICKER_INTERVAL_SECONDS = 2

# The ticker will not push progress past this ceiling on its own -
# the real jump to "Model Execution complete" (70%+) only happens
# once controller.analyze() actually returns.
TICKER_PROGRESS_CEILING = 65


def parse_modalities(value):
    if not value:
        return []

    try:
        parsed = json.loads(value)

        if isinstance(parsed, list):
            return parsed

    except json.JSONDecodeError:
        pass

    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


def inspect_raster_file(path):
    """
    Read real structural + geospatial metadata for a single
    uploaded file. Used by /analysis/inspect to power the
    Input Summary / Validation Status panel on the frontend
    with live data instead of placeholders.
    """

    file_path = Path(path)
    extension = file_path.suffix.lower()

    info = {
        "file_name": file_path.name,
        "file_type": None,
        "width": None,
        "height": None,
        "bands": None,
        "dtype": None,
        "crs": None,
        "georeferenced": False,
        "resolution_x": None,
        "resolution_y": None,
        "resolution_unit": None,
        "valid": True,
        "errors": []
    }

    try:
        if extension in SUPPORTED_GEOSPATIAL_FORMATS:
            with rasterio.open(file_path) as src:
                info["file_type"] = "GeoTIFF"
                info["width"] = src.width
                info["height"] = src.height
                info["bands"] = src.count
                info["dtype"] = str(src.dtypes[0]) if src.dtypes else None
                info["crs"] = str(src.crs) if src.crs else None
                info["georeferenced"] = src.crs is not None

                res_x, res_y = src.res

                info["resolution_x"] = round(res_x, 3)
                info["resolution_y"] = round(res_y, 3)
                info["resolution_unit"] = (
                    "m" if (src.crs and src.crs.is_projected)
                    else "deg" if src.crs
                    else None
                )

        elif extension in SUPPORTED_IMAGE_FORMATS:
            with Image.open(file_path) as image:
                info["file_type"] = "Standard Image"
                info["width"], info["height"] = image.size
                info["bands"] = len(image.getbands())
                info["dtype"] = "uint8"

        else:
            info["valid"] = False
            info["errors"].append(f"Unsupported file format: {extension or 'unknown'}")

    except Exception as error:
        info["valid"] = False
        info["errors"].append(str(error))

    return info


def compute_pair_status(files_metadata):
    """
    Compare dimensions/CRS across an image pair. Mirrors the
    logic in agent/compatibility.py but doesn't require a
    non-empty query, since this runs while the person is still
    uploading files.
    """

    if len(files_metadata) != 2:
        return "single_image", "single_image"

    first, second = files_metadata

    same_dimensions = (
        first["width"] is not None
        and first["width"] == second["width"]
        and first["height"] == second["height"]
    )

    dimension_status = "compatible" if same_dimensions else "different_but_alignable"

    if not first["crs"] or not second["crs"]:
        crs_status = "unknown"
    elif first["crs"] == second["crs"]:
        crs_status = "matching"
    else:
        crs_status = "different"

    return dimension_status, crs_status


def summarize_resolution(files_metadata):
    resolutions = [
        item["resolution_x"]
        for item in files_metadata
        if item.get("resolution_x") and item.get("resolution_unit") == "m"
    ]

    if not resolutions:
        return None

    average = sum(resolutions) / len(resolutions)

    return f"{average:.1f} m / pixel" if average % 1 else f"{int(average)} m / pixel"


def convert_path_to_url(path, request):
    if not path:
        return ""

    path = Path(path).resolve()

    try:
        relative_upload = path.relative_to(
            UPLOAD_DIR.resolve()
        )

        return (
            str(request.base_url).rstrip("/")
            + "/uploads/"
            + str(relative_upload).replace("\\", "/")
        )

    except ValueError:
        pass

    try:
        relative_output = path.relative_to(
            OUTPUT_DIR.resolve()
        )

        return (
            str(request.base_url).rstrip("/")
            + "/outputs/"
            + str(relative_output).replace("\\", "/")
        )

    except ValueError:
        pass

    return ""


def prepare_evidence(evidence, request):
    prepared = []

    for item in evidence or []:
        if not isinstance(item, dict):
            continue

        item = item.copy()

        item["url"] = convert_path_to_url(
            item.get("display_path") or item.get("path"),
            request
        )

        prepared.append(item)

    return prepared


def get_analysis_type(task):
    names = {
        "vqa": "Visual Question Answering",
        "captioning": "Image Captioning",
        "change_detection": "Change Detection",
        "change_vqa": "Change VQA",
        "optical_sar": "Optical-SAR Fusion"
    }

    return names.get(
        task,
        task.replace("_", " ").title()
    )


def get_sensor(result):
    modalities = result.get("modalities", [])

    if not modalities:
        return "Unknown"

    names = []

    for item in modalities:
        if isinstance(item, dict):
            modality = item.get("modality")

            if modality:
                names.append(modality)

    return " + ".join(names) if names else "Unknown"


def build_dashboard_stats():
    total = len(analysis_history)

    successful = sum(
        1
        for item in analysis_history
        if item.get("success")
    )

    confidences = [
        item.get("confidence", 0)
        for item in analysis_history
        if isinstance(
            item.get("confidence"),
            (int, float)
        )
    ]

    processing_times = [
        item.get("processing_time", 0)
        for item in analysis_history
        if isinstance(
            item.get("processing_time"),
            (int, float)
        )
    ]

    average_confidence = (
        sum(confidences) / len(confidences)
        if confidences
        else 0
    )

    average_processing_time = (
        sum(processing_times) / len(processing_times)
        if processing_times
        else 0
    )

    return {
        "total_analyses": total,
        "successful_analyses": successful,
        "average_confidence": round(
            average_confidence,
            1
        ),
        "average_processing_time": round(
            average_processing_time,
            1
        )
    }


def create_job_steps():
    return [
        {
            "name": "Input Validation",
            "status": "pending"
        },
        {
            "name": "Metadata Extraction",
            "status": "pending"
        },
        {
            "name": "Query Understanding",
            "status": "pending"
        },
        {
            "name": "Model Selection",
            "status": "pending"
        },
        {
            "name": "Model Execution",
            "status": "pending"
        },
        {
            "name": "Evidence Generation",
            "status": "pending"
        },
        {
            "name": "Result Generation",
            "status": "pending"
        }
    ]


def start_progress_ticker(job, floor, ceiling):
    """
    controller.analyze() is a single, long, blocking call - it is
    where the VLM actually gets loaded and run, and it reports no
    progress of its own while that happens. Without this, job["progress"]
    would sit frozen at whatever value it had before the call (this
    was the "stuck at 10%" bug) for however long that call takes.

    This spins up a small daemon thread that nudges job["progress"]
    upward every TICKER_INTERVAL_SECONDS while we wait, capped at
    `ceiling` so it never overtakes the real progress value that gets
    set once controller.analyze() actually returns. Call stop_event.set()
    (via the returned tuple) as soon as the real call finishes.
    """

    stop_event = Event()

    def _tick():
        progress = floor

        while not stop_event.is_set() and progress < ceiling:
            if stop_event.wait(TICKER_INTERVAL_SECONDS):
                break

            progress = min(progress + 2, ceiling)
            job["progress"] = progress

    ticker_thread = Thread(target=_tick, daemon=True)
    ticker_thread.start()

    return stop_event, ticker_thread


def run_analysis(
    analysis_id,
    query,
    image_paths,
    modality_list,
    display_paths=None
):
    start_time = datetime.now()

    display_paths = display_paths or image_paths

    try:
        job = analysis_jobs[analysis_id]

        job["status"] = "processing"

        # ------------------------------------------------------
        # Steps 1-4 (validation / metadata / query understanding /
        # model selection) all genuinely happen inside the single
        # controller.analyze() call below, so we can't report them
        # incrementally from real signals without changing
        # agent/controller.py. We still walk the stepper through
        # them quickly here so the UI shows forward motion instead
        # of parking on step 1, then hand off to the progress
        # ticker for the long-running model call.
        # ------------------------------------------------------

        job["step"] = "Input Validation"
        job["progress"] = 5
        job["steps"][0]["status"] = "running"
        job["steps"][0]["status"] = "completed"

        job["step"] = "Metadata Extraction"
        job["progress"] = 15
        job["steps"][1]["status"] = "running"
        job["steps"][1]["status"] = "completed"

        job["step"] = "Query Understanding"
        job["progress"] = 25
        job["steps"][2]["status"] = "running"
        job["steps"][2]["status"] = "completed"

        job["step"] = "Model Selection"
        job["progress"] = 35
        job["steps"][3]["status"] = "running"
        job["steps"][3]["status"] = "completed"

        job["step"] = "Model Execution"
        job["steps"][4]["status"] = "running"

        # ------------------------------------------------------
        # THE FIX: don't let progress sit frozen while the real
        # (potentially multi-minute, especially on first run or
        # on CPU) model call is in flight.
        # ------------------------------------------------------

        stop_event, ticker_thread = start_progress_ticker(
            job,
            floor=35,
            ceiling=TICKER_PROGRESS_CEILING
        )

        try:
            result = controller.analyze(
                query=query,
                images=image_paths,
                modalities=modality_list
            )
        finally:
            stop_event.set()
            ticker_thread.join(timeout=1)

        job["steps"][4]["status"] = "completed"

        job["step"] = "Evidence Generation"
        job["progress"] = 85
        job["steps"][5]["status"] = "running"

        job["steps"][5]["status"] = "completed"

        job["step"] = "Result Generation"
        job["progress"] = 95
        job["steps"][6]["status"] = "running"

        processing_time = (
            datetime.now() - start_time
        ).total_seconds()

        result["analysis_id"] = analysis_id

        result["status"] = (
            "Completed"
            if result.get("success")
            else "Failed"
        )

        result["processing_time"] = round(
            processing_time,
            2
        )

        result["created_at"] = datetime.now().isoformat()

        result["query"] = query

        result["images"] = [
            "/uploads/"
            + analysis_id
            + "/"
            + Path(path).name
            for path in display_paths
        ]

        # The evidence list's "input_image" entries were built by
        # the controller from the ORIGINAL image_paths (in the
        # same order), so they may still point at a raw .tif/.tiff.
        # Attach the browser-safe display path to each so the API
        # response links to something the frontend can actually
        # render, without touching what was fed to the model.
        input_evidence_items = [
            item
            for item in result.get("evidence", [])
            if isinstance(item, dict)
            and item.get("type") == "input_image"
        ]

        for item, display_path in zip(
            input_evidence_items,
            display_paths
        ):
            item["display_path"] = display_path

        result["input_type"] = (
            f"{len(image_paths)} Image"
            if len(image_paths) == 1
            else f"{len(image_paths)} Images"
        )

        result["sensor"] = get_sensor(result)

        job["steps"][6]["status"] = "completed"

        job["status"] = "completed"
        job["step"] = "Completed"
        job["progress"] = 100
        job["result"] = result

        analysis_history.append(result)

    except Exception as error:
        print("=" * 70)
        print("SATQUERY ANALYSIS ERROR")
        print(str(error))
        traceback.print_exc()
        print("=" * 70)

        job = analysis_jobs.get(analysis_id)

        if job:
            job["status"] = "failed"
            job["step"] = "Analysis Failed"
            job["progress"] = 100
            job["error"] = str(error)

            for step in job["steps"]:
                if step["status"] == "running":
                    step["status"] = "failed"


@app.get("/")
def root():
    return {
        "message": "SatQuery AI API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


EVALUATION_METRICS_PATH = OUTPUT_DIR / "evaluation_metrics.json"
evaluation_job = {"status": "idle", "error": None}
CURRENT_EVALUATION_MODELS = {
    "siamese_temporal_cd",
    "optical_sar_change_net",
    "satquery_vlm",
}


def _current_evaluation_placeholder():
    return {
        "generated_at": None,
        "models": [
            {
                "model": "siamese_temporal_cd",
                "display_name": "SiameseTemporalCD",
                "model_type": "Bi-temporal optical change detection network",
                "status": "pending",
                "pending_reason": "Run the evaluation to load runs/optical/best.pt and score the current model.",
                "sample_size": 0,
                "metrics": {},
            },
            {
                "model": "optical_sar_change_net",
                "display_name": "OpticalSARChangeNet",
                "model_type": "Bi-temporal optical + SAR change detection network",
                "status": "pending",
                "pending_reason": "Run the evaluation to load runs/optical_sar/best.pt and score the current model.",
                "sample_size": 0,
                "metrics": {},
            },
            {
                "model": "satquery_vlm",
                "display_name": "VQA / Captioning / Grounding (Shared VLM)",
                "model_type": "Fine-tuned Qwen2.5-VL-3B-Instruct (LoRA adapter)",
                "status": "pending",
                "pending_reason": "A held-out evaluation split is required before reporting VQA/captioning metrics.",
                "sample_size": 0,
                "metrics": {},
            },
        ],
    }


def _evaluation_worker():
    global evaluation_job
    try:
        from evaluation.run_all import run_all
        evaluation_job = {"status": "running", "error": None}
        run_all()
        evaluation_job = {"status": "completed", "error": None}
    except Exception as error:
        traceback.print_exc()
        evaluation_job = {"status": "error", "error": str(error)}


@app.get("/models/metrics")
def models_metrics():
    """
    Serves the evaluation metrics generated by
    `python evaluation/run_all.py` (see Backend/evaluation/).

    Each entry in "models" has a "status" of:
      - "evaluated"          real metrics, scored against a
                              ground-truth (or synthetic
                              ground-truth) benchmark.
      - "evaluated_baseline" real metrics, but for an untrained
                              model - see that entry's
                              "methodology" for what's actually
                              being measured.
      - "pending"            metrics could not be computed in the
                              environment that last ran the
                              evaluation (e.g. no GPU / no model
                              weights available) - see
                              "pending_reason".
      - "error"               the evaluator itself crashed - see
                              "error".

    If evaluation has never been run, returns a payload with an
    empty "models" list and "generated_at": null rather than a
    404, so the frontend can render an empty/"not run yet" state.
    """

    if not EVALUATION_METRICS_PATH.exists():
        payload = _current_evaluation_placeholder()
        payload["evaluation_status"] = evaluation_job["status"]
        payload["evaluation_error"] = evaluation_job.get("error")
        return payload

    try:
        with EVALUATION_METRICS_PATH.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except Exception:
        payload = _current_evaluation_placeholder()

    model_ids = {item.get("model") for item in payload.get("models", [])}
    if not model_ids.intersection(CURRENT_EVALUATION_MODELS):
        payload = _current_evaluation_placeholder()

    payload["evaluation_status"] = evaluation_job["status"]
    payload["evaluation_error"] = evaluation_job.get("error")
    return payload


@app.post("/models/metrics/run")
def run_model_metrics():
    global evaluation_job

    if evaluation_job.get("status") == "running":
        return {"status": "running"}

    evaluation_job = {"status": "running", "error": None}
    thread = Thread(target=_evaluation_worker, daemon=True)
    thread.start()
    return {"status": "started"}


@app.get("/models/metrics/status")
def model_metrics_status():
    return evaluation_job


@app.get("/dashboard")
def dashboard():
    type_counter = Counter(
        item.get("task")
        for item in analysis_history
        if item.get("task")
    )

    analysis_types = []

    for task, count in type_counter.items():
        analysis_types.append({
            "name": get_analysis_type(task),
            "task": task,
            "count": count
        })

    analysis_types.sort(
        key=lambda item: item["count"],
        reverse=True
    )

    recent_analyses = []

    for item in analysis_history[-8:][::-1]:
        recent_analyses.append({
            "id": item.get("analysis_id"),
            "task": get_analysis_type(
                item.get("task", "")
            ),
            "task_key": item.get("task"),
            "status": item.get("status"),
            "model": item.get("model"),
            "confidence": item.get("confidence", 0),
            "sensor": item.get("sensor"),
            "input_type": item.get("input_type"),
            "created_at": item.get("created_at")
        })

    activities = []

    for item in analysis_history[-8:][::-1]:
        activities.append({
            "id": item.get("analysis_id"),
            "type": "analysis",
            "title": (
                f"{get_analysis_type(item.get('task', ''))} "
                f"completed"
            ),
            "description": item.get(
                "query",
                ""
            ),
            "created_at": item.get(
                "created_at"
            ),
            "status": item.get(
                "status"
            )
        })

    return {
        "stats": build_dashboard_stats(),
        "analysis_types": analysis_types,
        "recent_analyses": recent_analyses,
        "activities": activities
    }


@app.post("/analysis/inspect")
async def inspect_analysis(
    images: list[UploadFile] = File(...),
    query: str = Form(""),
    modalities: str = Form("")
):
    """
    Lightweight, read-only preview used by the New Analysis
    screen while the person is still uploading files and typing
    their question. Runs real format/geospatial/modality checks
    and (when a query is present) a task-routing preview, without
    kicking off a full analysis job.
    """

    if not images:
        raise HTTPException(status_code=400, detail="At least one image is required.")

    if len(images) > 2:
        raise HTTPException(status_code=400, detail="A maximum of two images is supported.")

    preview_id = str(uuid.uuid4())
    preview_dir = UPLOAD_DIR / "_preview" / preview_id
    preview_dir.mkdir(parents=True, exist_ok=True)

    declared_modalities = parse_modalities(modalities)

    try:
        saved_paths = []

        for image in images:
            if not image.filename:
                continue

            filename = Path(image.filename).name
            file_path = preview_dir / filename

            with file_path.open("wb") as buffer:
                shutil.copyfileobj(image.file, buffer)

            saved_paths.append(str(file_path))

        files_response = []

        for index, path in enumerate(saved_paths):
            file_validation = validate_file(path)
            raster_info = inspect_raster_file(path)

            declared = (
                declared_modalities[index]
                if index < len(declared_modalities)
                else None
            )

            modality_info = detect_modality(path, declared_modality=declared)

            files_response.append({
                **raster_info,
                "file_size_mb": file_validation.get("file_size_mb", 0.0),
                "valid": raster_info["valid"] and file_validation.get("valid", False),
                "errors": raster_info["errors"] + file_validation.get("errors", []),
                "modality": modality_info
            })

        dimension_status, crs_status = compute_pair_status(files_response)

        overall_valid = all(item["valid"] for item in files_response)

        optical_sar_check = (
            check_optical_sar_pair(saved_paths, declared_modalities=declared_modalities)
            if len(saved_paths) == 2
            else None
        )

        task_preview = None

        if query and query.strip():
            routing = route_query(
                query=query,
                number_of_images=len(saved_paths),
                modalities=[item["modality"]["modality"] for item in files_response]
            )

            task_preview = {
                **routing,
                "task_label": get_task_description(routing["task"])
            }

        if len(files_response) == 1:
            configuration = "Single Image"
        elif optical_sar_check and optical_sar_check.get("is_optical_sar"):
            configuration = "Optical + SAR Pair"
        else:
            configuration = "Two Images (Bi-temporal)"

        return {
            "valid": overall_valid,
            "files": files_response,
            "dimension_status": dimension_status,
            "crs_status": crs_status,
            "optical_sar_check": optical_sar_check,
            "task_preview": task_preview,
            "resolution_summary": summarize_resolution(files_response),
            "configuration": configuration
        }

    finally:
        shutil.rmtree(preview_dir, ignore_errors=True)


@app.post("/analysis")
async def create_analysis(
    query: str = Form(...),
    images: list[UploadFile] = File(...),
    modalities: str = Form("")
):
    if not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query is required."
        )

    if not images:
        raise HTTPException(
            status_code=400,
            detail="At least one image is required."
        )

    analysis_id = str(uuid.uuid4())

    analysis_dir = UPLOAD_DIR / analysis_id

    analysis_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    image_paths = []
    display_paths = []

    for image in images:
        if not image.filename:
            continue

        filename = Path(
            image.filename
        ).name

        file_path = analysis_dir / filename

        with file_path.open("wb") as buffer:
            shutil.copyfileobj(
                image.file,
                buffer
            )

        resolved_path = str(file_path.resolve())

        image_paths.append(
            resolved_path
        )

        # Browsers can't render .tif/.tiff directly - generate a
        # PNG preview (saved alongside the original in the same
        # analysis_dir, so the /uploads static mount can serve it)
        # for display purposes only. The original file is still
        # what gets passed to the model below.
        display_paths.append(
            generate_display_image(
                resolved_path,
                str(analysis_dir)
            ) or resolved_path
        )

    if not image_paths:
        raise HTTPException(
            status_code=400,
            detail="No valid images were uploaded."
        )

    modality_list = parse_modalities(
        modalities
    )

    analysis_jobs[analysis_id] = {
        "analysis_id": analysis_id,
        "status": "queued",
        "step": "Preparing Analysis",
        "progress": 0,
        "steps": create_job_steps(),
        "result": None,
        "error": None
    }

    thread = Thread(
        target=run_analysis,
        args=(
            analysis_id,
            query,
            image_paths,
            modality_list,
            display_paths
        ),
        daemon=True
    )

    thread.start()

    return {
        "analysis_id": analysis_id,
        "status": "queued"
    }


@app.get("/analysis/{analysis_id}/status")
def analysis_status(
    analysis_id: str
):
    job = analysis_jobs.get(
        analysis_id
    )

    if not job:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found."
        )

    return {
        "analysis_id": analysis_id,
        "status": job.get("status"),
        "step": job.get("step"),
        "progress": job.get("progress"),
        "steps": job.get("steps", []),
        "error": job.get("error")
    }


@app.get("/analysis/{analysis_id}")
def get_analysis(
    analysis_id: str,
    request: Request
):
    job = analysis_jobs.get(
        analysis_id
    )

    if not job:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found."
        )

    if job.get("status") != "completed":
        return {
            "analysis_id": analysis_id,
            "status": job.get("status")
        }

    result = job["result"].copy()

    result["images"] = [
        str(request.base_url).rstrip("/")
        + image
        for image in result.get(
            "images",
            []
        )
    ]

    result["evidence"] = prepare_evidence(
        result.get("evidence", []),
        request
    )

    return result