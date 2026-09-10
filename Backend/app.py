import os
import tempfile
from pathlib import Path
from datetime import datetime

# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

HF_CACHE = PROJECT_ROOT / "hf_cache"

os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE / "transformers")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ============================================================
# IMPORTS
# ============================================================

import streamlit as st
from PIL import Image

from agent.controller import SatQueryController


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="SatQuery AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL UI CSS
#
# CSS ONLY.
# NO HTML CONTENT IS USED FOR UI ELEMENTS.
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 15% 0%,
                rgba(0, 190, 255, 0.08),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 10%,
                rgba(40, 90, 200, 0.10),
                transparent 30%
            ),
            #07111f;
    }

    .main .block-container {
        max-width: 1280px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }

    section[data-testid="stSidebar"] {
        background: #081522;
        border-right: 1px solid rgba(130, 190, 225, 0.12);
    }

    h1 {
        font-size: 3.2rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.04em;
    }

    h2 {
        font-size: 1.75rem !important;
        margin-top: 2rem !important;
    }

    h3 {
        font-size: 1.15rem !important;
    }

    p {
        color: #a9c0cc;
    }

    [data-testid="stMetric"] {
        background: rgba(11, 29, 44, 0.82);
        border: 1px solid rgba(90, 160, 200, 0.16);
        border-radius: 14px;
        padding: 1rem;
    }

    [data-testid="stMetricLabel"] {
        color: #7895a8 !important;
    }

    [data-testid="stMetricValue"] {
        color: #edf8fb !important;
    }

    [data-testid="stFileUploader"] {
        background: rgba(9, 24, 38, 0.70);
        border: 1px dashed rgba(36, 199, 255, 0.32);
        border-radius: 14px;
        padding: 0.5rem;
    }

    textarea,
    input {
        background-color: #0c1d2c !important;
        color: #edf7fa !important;
    }

    textarea::placeholder {
        color: #617c8e !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #0c1d2c !important;
        color: #edf7fa !important;
        border-color: rgba(120, 190, 225, 0.18) !important;
    }

    div[data-baseweb="select"] span {
        color: #edf7fa !important;
    }

    .stButton > button {
        min-height: 46px;
        border-radius: 11px;
        font-weight: 750;
        background: #0d2131;
        color: #edf8fb;
        border: 1px solid rgba(36, 199, 255, 0.22);
    }

    .stButton > button:hover {
        border-color: #24c7ff;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(
            135deg,
            #087fa8,
            #155ea2
        );
        color: white;
        border: none;
    }

    [data-testid="stExpander"] {
        background: rgba(10, 27, 42, 0.70);
        border: 1px solid rgba(120, 185, 220, 0.13);
        border-radius: 14px;
    }

    hr {
        border-color: rgba(120, 180, 220, 0.10);
    }

    .footer {
        text-align: center;
        color: #526d80;
        font-size: 0.75rem;
        padding-top: 2rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "controller" not in st.session_state:
    st.session_state.controller = SatQueryController()

if "result" not in st.session_state:
    st.session_state.result = None

if "uploaded_paths" not in st.session_state:
    st.session_state.uploaded_paths = []


# ============================================================
# HELPERS
# ============================================================

def save_uploaded_file(uploaded_file):

    suffix = Path(uploaded_file.name).suffix.lower()

    temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    )

    temp.write(
        uploaded_file.getbuffer()
    )

    temp.close()

    return temp.name


def confidence_label(value):

    if value is None:
        return "Not available"

    try:

        value = float(value)

        if value >= 0.80:
            return "High"

        if value >= 0.55:
            return "Moderate"

        return "Low"

    except Exception:

        return "Not available"


def confidence_percent(value):

    if value is None:
        return "—"

    try:
        return f"{float(value) * 100:.0f}%"
    except Exception:
        return "—"


def find_evidence(result, evidence_type):

    return [
        item
        for item in result.get("evidence", [])
        if item.get("type") == evidence_type
    ]


def show_image(path):

    if not path:
        return

    path = Path(path)

    if not path.exists():

        st.warning(
            "The generated visual evidence could not be found."
        )

        return

    try:

        image = Image.open(path)

        st.image(
            image,
            use_container_width=True,
        )

    except Exception:

        st.warning(
            "This image could not be displayed."
        )


def create_human_report(result, query):

    task = result.get(
        "task",
        "analysis",
    )

    model = result.get(
        "model",
        "SatQuery AI",
    )

    confidence = result.get(
        "confidence"
    )

    answer = result.get(
        "answer",
        "No result was generated.",
    )

    modalities = result.get(
        "modalities",
        [],
    )

    modality_names = [
        str(x.get("modality", "")).upper()
        for x in modalities
        if x.get("modality")
    ]

    lines = []

    lines.append(
        "SATQUERY AI — ANALYSIS REPORT"
    )

    lines.append(
        "=" * 50
    )

    lines.append(
        f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}"
    )

    lines.append("")

    lines.append(
        "YOUR QUESTION"
    )

    lines.append(
        query
    )

    lines.append("")

    lines.append(
        "ANALYSIS PERFORMED"
    )

    readable_task = {
        "vqa": "Satellite Image Question Answering",
        "captioning": "Satellite Scene Description",
        "change_detection": "Change Detection",
        "change_vqa": "Change Analysis with Question Answering",
        "optical_sar": "Optical + SAR Analysis",
    }.get(
        task,
        task.replace("_", " ").title(),
    )

    lines.append(
        readable_task
    )

    lines.append("")

    if modality_names:

        lines.append(
            "IMAGERY USED"
        )

        lines.append(
            ", ".join(modality_names)
        )

        lines.append("")

    lines.append(
        "SATQUERY AI FINDING"
    )

    lines.append(
        answer
    )

    lines.append("")

    lines.append(
        "CONFIDENCE"
    )

    lines.append(
        confidence_label(confidence)
    )

    lines.append("")

    lines.append(
        "CONFIDENCE NOTE"
    )

    lines.append(
        "Confidence indicates how strongly the system "
        "supports the generated result. It should be "
        "treated as an analytical indicator rather than "
        "a guarantee of correctness."
    )

    lines.append("")

    lines.append(
        "END OF REPORT"
    )

    return "\n".join(lines)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🛰️ SatQuery AI")

    st.caption(
        "Agentic Remote-Sensing Intelligence"
    )

    st.divider()

    st.subheader(
        "Analysis Configuration"
    )

    modality_mode = st.selectbox(
        "Input modality",
        [
            "Auto Detect",
            "Optical",
            "SAR",
            "Optical + SAR",
        ],
    )

    st.divider()

    st.subheader(
        "Available Intelligence"
    )

    st.markdown(
        """
        👁️ **Visual Q&A**

        Ask questions about satellite imagery.

        📝 **Scene Captioning**

        Generate an understandable description.

        🔄 **Change Detection**

        Identify differences between two images.

        🔍 **Change Q&A**

        Ask what changed and where.

        🛰️ **Optical + SAR**

        Analyze complementary satellite modalities.
        """
    )

    st.divider()

    st.caption(
        "5 analysis capabilities available"
    )

    if st.button(
        "Clear Analysis",
        use_container_width=True,
    ):

        st.session_state.result = None
        st.session_state.uploaded_paths = []

        st.rerun()


# ============================================================
# HERO
# ============================================================

st.caption(
    "AGENTIC EARTH OBSERVATION"
)

st.title(
    "SatQuery AI"
)

st.write(
    "Understand, compare and question satellite imagery "
    "using an intelligent multi-model analysis system."
)

st.success(
    "System ready for analysis"
)

st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.header(
    "New Satellite Analysis"
)

st.write(
    "Upload one image for visual analysis, or two images "
    "for change and multi-modal analysis."
)


uploaded_files = st.file_uploader(
    "Satellite imagery",
    type=[
        "tif",
        "tiff",
        "png",
        "jpg",
        "jpeg",
    ],
    accept_multiple_files=True,
)


# ============================================================
# IMAGE PREVIEW
# ============================================================

if uploaded_files:

    st.subheader(
        "Uploaded Imagery"
    )

    preview_columns = st.columns(
        min(len(uploaded_files), 2)
    )

    for i, uploaded_file in enumerate(
        uploaded_files
    ):

        with preview_columns[
            i % len(preview_columns)
        ]:

            st.caption(
                f"Image {i + 1}: {uploaded_file.name}"
            )

            try:

                preview = Image.open(
                    uploaded_file
                )

                st.image(
                    preview,
                    use_container_width=True,
                )

            except Exception:

                st.info(
                    "Raster image uploaded successfully."
                )


# ============================================================
# QUESTION
# ============================================================

st.subheader(
    "Ask SatQuery AI"
)

query = st.text_area(
    "Question",
    placeholder=(
        "Examples:\n"
        "• What type of land cover is visible?\n"
        "• Describe this satellite scene.\n"
        "• What changed between these two images?\n"
        "• Did the urban area increase?\n"
        "• Analyze these optical and SAR images together."
    ),
    height=150,
    label_visibility="collapsed",
)


# ============================================================
# MODALITY
# ============================================================

declared_modalities = None

if modality_mode == "Optical":

    declared_modalities = ["optical"]

elif modality_mode == "SAR":

    declared_modalities = ["sar"]

elif modality_mode == "Optical + SAR":

    declared_modalities = [
        "optical",
        "sar",
    ]


# ============================================================
# ANALYZE
# ============================================================

if st.button(
    "✦  ANALYZE IMAGERY",
    type="primary",
    use_container_width=True,
):

    if not uploaded_files:

        st.error(
            "Please upload at least one image."
        )

        st.stop()

    if not query.strip():

        st.error(
            "Please enter a question."
        )

        st.stop()

    if len(uploaded_files) > 2:

        st.error(
            "Please upload no more than two images."
        )

        st.stop()

    if (
        modality_mode == "Optical + SAR"
        and len(uploaded_files) != 2
    ):

        st.error(
            "Optical + SAR analysis requires two images."
        )

        st.stop()

    image_paths = []

    for file in uploaded_files:

        image_paths.append(
            save_uploaded_file(file)
        )

    st.session_state.uploaded_paths = image_paths

    with st.spinner(
        "SatQuery AI is analyzing the imagery..."
    ):

        try:

            result = (
                st.session_state.controller.analyze(
                    query=query,
                    images=image_paths,
                    modalities=declared_modalities,
                )
            )

            st.session_state.result = result

        except Exception as error:

            st.session_state.result = {
                "success": False,
                "error": str(error),
            }


# ============================================================
# RESULTS
# ============================================================

result = st.session_state.result


if result:

    st.divider()

    if not result.get("success", False):

        st.error(
            result.get(
                "error",
                "SatQuery AI could not complete the analysis.",
            )
        )

        st.stop()


    # ========================================================
    # RESULT SUMMARY
    # ========================================================

    st.caption(
        "ANALYSIS COMPLETE"
    )

    task = result.get(
        "task",
        "unknown",
    )

    readable_task = {
        "vqa": "Visual Question Answering",
        "captioning": "Scene Description",
        "change_detection": "Change Detection",
        "change_vqa": "Change Analysis",
        "optical_sar": "Optical + SAR Analysis",
    }.get(
        task,
        task.replace("_", " ").title(),
    )

    st.header(
        "What SatQuery AI found"
    )

    st.info(
        result.get(
            "answer",
            "No result was generated.",
        )
    )


    # ========================================================
    # SIMPLE SUMMARY CARDS
    # ========================================================

    confidence = result.get(
        "confidence"
    )

    routing_confidence = result.get(
        "routing_confidence"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Analysis",
            readable_task,
        )

    with c2:

        st.metric(
            "Result confidence",
            confidence_label(
                confidence
            ),
        )

    with c3:

        st.metric(
            "Images analyzed",
            len(uploaded_files)
            if uploaded_files
            else "—",
        )


    # ========================================================
    # HUMAN INTERPRETATION
    # ========================================================

    st.header(
        "Understanding the result"
    )

    if task == "vqa":

        st.write(
            "SatQuery AI examined the uploaded satellite "
            "image and used its visual understanding capability "
            "to answer your question."
        )

    elif task == "captioning":

        st.write(
            "SatQuery AI examined the satellite scene and "
            "generated a description of the visible landscape "
            "and major features."
        )

    elif task == "change_detection":

        st.write(
            "SatQuery AI compared the two images pixel-by-pixel "
            "to identify areas that appear different between "
            "the two observations."
        )

    elif task == "change_vqa":

        st.write(
            "SatQuery AI compared the two observations and "
            "used its visual-language model to answer the "
            "specific question about the detected change."
        )

    elif task == "optical_sar":

        st.write(
            "SatQuery AI examined the optical and SAR imagery "
            "together. Optical imagery provides visual information "
            "about the surface, while SAR provides complementary "
            "radar-based information."
        )


    # ========================================================
    # VISUAL EVIDENCE
    # ========================================================

    st.header(
        "Visual Evidence"
    )

    # --------------------------------------------------------
    # CHANGE TASKS
    # --------------------------------------------------------

    if task in [
        "change_detection",
        "change_vqa",
    ]:

        inputs = find_evidence(
            result,
            "input_image",
        )

        change_maps = find_evidence(
            result,
            "change_map",
        )

        columns = st.columns(3)

        if len(inputs) >= 1:

            with columns[0]:

                st.subheader(
                    "Before"
                )

                show_image(
                    inputs[0].get("path")
                )

        if len(inputs) >= 2:

            with columns[1]:

                st.subheader(
                    "After"
                )

                show_image(
                    inputs[1].get("path")
                )

        if change_maps:

            with columns[2]:

                st.subheader(
                    "Detected Difference"
                )

                show_image(
                    change_maps[0].get("path")
                )


    # --------------------------------------------------------
    # OPTICAL + SAR
    # --------------------------------------------------------

    elif task == "optical_sar":

        inputs = find_evidence(
            result,
            "input_image",
        )

        fusion = find_evidence(
            result,
            "optical_sar_fusion",
        )

        columns = st.columns(3)

        if len(inputs) >= 1:

            with columns[0]:

                st.subheader(
                    "Optical Image"
                )

                show_image(
                    inputs[0].get("path")
                )

        if len(inputs) >= 2:

            with columns[1]:

                st.subheader(
                    "SAR Image"
                )

                show_image(
                    inputs[1].get("path")
                )

        if fusion:

            with columns[2]:

                st.subheader(
                    "Combined Analysis"
                )

                show_image(
                    fusion[0].get("path")
                )


    # --------------------------------------------------------
    # SINGLE IMAGE
    # --------------------------------------------------------

    else:

        inputs = find_evidence(
            result,
            "input_image",
        )

        if inputs:

            st.subheader(
                "Analyzed Image"
            )

            show_image(
                inputs[0].get("path")
            )


    # ========================================================
    # CHANGE INFORMATION
    # ========================================================

    if task == "change_detection":

        st.header(
            "Change Summary"
        )

        model_result = result.get(
            "model_result",
            {},
        )

        change_percentage = (
            model_result.get(
                "change_percentage"
            )
            if model_result
            else result.get(
                "change_percentage"
            )
        )

        changed_pixels = (
            model_result.get(
                "changed_pixels"
            )
            if model_result
            else result.get(
                "changed_pixels"
            )
        )

        total_pixels = (
            model_result.get(
                "total_pixels"
            )
            if model_result
            else result.get(
                "total_pixels"
            )
        )

        image_size = (
            model_result.get(
                "image_size"
            )
            if model_result
            else result.get(
                "image_size"
            )
        )

        a, b, c = st.columns(3)

        with a:

            if change_percentage is not None:

                st.metric(
                    "Area that appears different",
                    f"{change_percentage:.2f}%",
                )

            else:

                st.metric(
                    "Area that appears different",
                    "—",
                )

        with b:

            if changed_pixels is not None:

                st.metric(
                    "Detected changed pixels",
                    f"{changed_pixels:,}",
                )

            else:

                st.metric(
                    "Detected changed pixels",
                    "—",
                )

        with c:

            if image_size:

                width = image_size.get(
                    "width",
                    "?"
                )

                height = image_size.get(
                    "height",
                    "?"
                )

                st.metric(
                    "Comparison area",
                    f"{width} × {height}",
                )

            else:

                st.metric(
                    "Comparison area",
                    "—",
                )

        st.caption(
            "Note: the detected percentage represents "
            "pixels identified as different by the current "
            "change-detection method. It does not automatically "
            "mean that the entire area represents a real-world "
            "land-cover change."
        )


    # ========================================================
    # OPTICAL + SAR SUMMARY
    # ========================================================

    if task == "optical_sar":

        st.header(
            "Optical + SAR Summary"
        )

        st.write(
            "The two imagery sources were successfully processed "
            "together."
        )

        st.info(
            "This analysis combines information from optical "
            "imagery and radar-based SAR imagery. The current "
            "fusion output should be considered an analytical "
            "feature representation rather than a direct "
            "real-world classification."
        )


    # ========================================================
    # HOW THE SYSTEM WORKED
    # ========================================================

    with st.expander(
        "How SatQuery AI reached this result",
        expanded=False,
    ):

        st.write(
            f"**1. Your request**  \n"
            f"SatQuery AI received your question and imagery."
        )

        st.write(
            f"**2. Analysis selected**  \n"
            f"The system selected: **{readable_task}**."
        )

        st.write(
            "**3. Imagery examined**  \n"
            f"{len(uploaded_files) if uploaded_files else 'The uploaded'} "
            "image(s) were processed for the analysis."
        )

        st.write(
            "**4. Result generated**  \n"
            "The selected analysis model produced the result "
            "shown above."
        )

        st.write(
            "**5. Visual evidence**  \n"
            "Supporting imagery is displayed above so that the "
            "result can be visually inspected."
        )


    # ========================================================
    # REPORT
    # ========================================================

    st.header(
        "Analysis Report"
    )

    human_report = create_human_report(
        result,
        query,
    )

    st.download_button(
        label="⬇ Download Human-Readable Report",
        data=human_report,
        file_name="SatQuery_AI_Analysis_Report.txt",
        mime="text/plain",
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">'
    'SATQUERY AI · Agentic Remote-Sensing Intelligence'
    '</div>',
    unsafe_allow_html=True,
)