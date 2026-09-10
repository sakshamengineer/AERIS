import { useEffect, useMemo, useRef, useState } from 'react'
import { useForm } from 'react-hook-form'
import {
    UploadCloud,
    X,
    Send,
    Plus,
    CheckCircle2,
    XCircle,
    MinusCircle,
    BookOpen,
    Satellite,
    MessageSquareText,
    Layers3,
    Gauge,
    MapPin,
    Timer,
    Info,
    Pencil,
    HardDrive,
    Ruler,
    CalendarDays,
    SlidersHorizontal,
    ChevronDown,
    RotateCcw,
    Loader2
} from 'lucide-react'
import { toast } from 'sonner'
import { startAnalysis, inspectAnalysis } from '../../lib/api'

const SUPPORTED_TYPES = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
const MAX_IMAGES = 4 // Optical-SAR uses four inputs: optical T1/T2 + SAR T1/T2.

const MODALITY_OPTIONS = ["optical", "sar", "unknown"]

const MODALITY_STYLES = {
    optical: "bg-emerald-500/90",
    sar: "bg-purple-500/90",
    unknown: "bg-slate-600/90"
}

const TASK_OPTIONS = [
    { value: "vqa", label: "Visual Question Answering" },
    { value: "captioning", label: "Image Captioning" },
    { value: "change_detection", label: "Change Detection" },
    { value: "optical_sar", label: "Optical-SAR Change Analysis" }
]

const EXAMPLE_QUERIES = [
    "What changed between these two images? Estimate the extent of change.",
    "What type of land is visible in this image?",
    "Caption this image",
    "Analyse both sar and optical image pair and give me information about changes"
]

const getExtension = (name) => "." + name.split(".").pop().toLowerCase()

const formatFileDate = (timestamp) => {
    if (!timestamp) return null

    const date = new Date(timestamp)

    if (Number.isNaN(date.getTime())) return null

    return date.toISOString().slice(0, 10)
}

const formatMB = (bytes) => (bytes / 1024 / 1024).toFixed(2)

const InputState = (props) => {
    const {
        register,
        handleSubmit,
        watch,
        setValue,
        formState: { errors }
    } = useForm()

    const [files, setFiles] = useState([]) // [{ file, id, modality, url }]
    const [dragging, setDragging] = useState(false)
    const [showExamples, setShowExamples] = useState(false)
    const [submitting, setSubmitting] = useState(false)

    const [inspection, setInspection] = useState(null)
    const [inspecting, setInspecting] = useState(false)

    const [taskOverride, setTaskOverride] = useState("auto")
    const [showTaskMenu, setShowTaskMenu] = useState(false)
    const [showAdvanced, setShowAdvanced] = useState(false)

    const inspectTimer = useRef(null)
    const inspectToken = useRef(0)

    const query = watch("query") || ""

    const addFiles = (incoming) => {
        const incomingArray = Array.from(incoming)

        const valid = incomingArray.filter((file) =>
            SUPPORTED_TYPES.includes(getExtension(file.name))
        )

        if (!valid.length) {
            toast.error("Please upload a supported image format.")
            return
        }

        setFiles((previous) => {
            const room = MAX_IMAGES - previous.length

            if (room <= 0) {
                toast.error(`Maximum ${MAX_IMAGES} images allowed.`)
                return previous
            }

            const accepted = valid.slice(0, room)

            if (valid.length > accepted.length) {
                toast.warning(`Only ${MAX_IMAGES} images are supported per analysis.`)
            }

            const merged = [
                ...previous,
                ...accepted.map((file, index) => ({
                    file,
                    id: `${file.name}-${Date.now()}-${index}`,
                    modality: "unknown",
                    url: URL.createObjectURL(file)
                }))
            ]

            // Optical-SAR is a four-input contract. For four uploaded
            // images, the UI defines the canonical order as:
            //   1 = Optical T1, 2 = Optical T2,
            //   3 = SAR T1,    4 = SAR T2.
            // This avoids trying to infer SAR from PNG/JPEG appearance,
            // which is inherently unreliable for exported/pseudo-RGB SAR.
            if (merged.length === 4) {
                return merged.map((item, index) => ({
                    ...item,
                    modality: index < 2 ? "optical" : "sar"
                }))
            }

            return merged
        })
    }

    const removeFile = (id) => {
        setFiles((previous) => previous.filter((item) => item.id !== id))
    }

    const cycleModality = (id) => {
        setFiles((previous) =>
            previous.map((item) => {
                if (item.id !== id) return item

                const currentIndex = MODALITY_OPTIONS.indexOf(item.modality)
                const next = MODALITY_OPTIONS[(currentIndex + 1) % MODALITY_OPTIONS.length]

                return { ...item, modality: next }
            })
        )
    }

    const resetModalities = () => {
        setFiles((previous) =>
            previous.map((item, index) => ({
                ...item,
                modality:
                    previous.length === 4
                        ? index < 2
                            ? "optical"
                            : "sar"
                        : "unknown"
            }))
        )
        setShowAdvanced(false)
    }

    // ------------------------------------------------------------
    // Live backend inspection: whenever files or the question
    // change, ask the backend what it actually sees (format,
    // geospatial metadata, modality, task routing) instead of
    // guessing on the frontend. Debounced so we don't spam the
    // API while the person is still typing.
    // ------------------------------------------------------------

    useEffect(() => {
        if (inspectTimer.current) {
            clearTimeout(inspectTimer.current)
        }

        if (!files.length) {
            setInspection(null)
            setInspecting(false)
            return
        }

        const token = ++inspectToken.current

        inspectTimer.current = setTimeout(async () => {
            try {
                setInspecting(true)

                const result = await inspectAnalysis({
                    query,
                    files: files.map((item) => item.file),
                    modalities: files.map((item) =>
                        item.modality === "unknown" ? "" : item.modality
                    )
                })

                if (token === inspectToken.current) {
                    setInspection(result)
                }
            } catch (error) {
                if (token === inspectToken.current) {
                    setInspection(null)
                }
            } finally {
                if (token === inspectToken.current) {
                    setInspecting(false)
                }
            }
        }, 600)

        return () => clearTimeout(inspectTimer.current)
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [files, query])

    const getFileMeta = (index) => inspection?.files?.[index] || null

    const displayModality = (item, index) => {
        if (item.modality !== "unknown") return item.modality

        const detected = getFileMeta(index)?.modality?.modality

        return detected && detected !== "unknown" ? detected : "unknown"
    }

    // ------------------------------------------------------------
    // Validation Status panel — backed by live inspection results
    // once available, falling back to lightweight client checks
    // (or a neutral "Pending" state) before that.
    // ------------------------------------------------------------

    const validation = useMemo(() => {
        const checks = []
        const hasFiles = files.length > 0
        const trimmedQuery = query.trim()

        const formatValid =
            hasFiles &&
            files.every((item) => SUPPORTED_TYPES.includes(getExtension(item.file.name)))

        checks.push({
            label: "File Format",
            ok: hasFiles ? formatValid : null,
            text: !hasFiles ? "Pending" : formatValid ? "Valid" : "Invalid"
        })

        const expectedCount =
            taskOverride === "vqa" || taskOverride === "captioning"
                ? 1
                : taskOverride === "change_detection"
                    ? 2
                    : taskOverride === "optical_sar"
                        ? 4
                        : null

        const countValid = expectedCount
            ? files.length === expectedCount
            : [1, 2, 4].includes(files.length)

        checks.push({
            label: "Number of Images",
            ok: hasFiles ? countValid : null,
            text: !hasFiles
                ? "Pending"
                : countValid
                    ? expectedCount ? `${expectedCount} required` : "Valid"
                    : expectedCount ? `${expectedCount} required` : "Invalid"
        })

        if (!hasFiles) {
            checks.push({ label: "Image Dimensions", ok: null, text: "Pending" })
        } else if (!inspection) {
            checks.push({ label: "Image Dimensions", ok: null, text: inspecting ? "Checking…" : "Pending" })
        } else if (inspection.dimension_status === "different_but_alignable") {
            checks.push({ label: "Image Dimensions", ok: true, text: "Alignable" })
        } else if (inspection.dimension_status === "compatible" || inspection.dimension_status === "single_image") {
            checks.push({ label: "Image Dimensions", ok: true, text: "Valid" })
        } else {
            checks.push({ label: "Image Dimensions", ok: null, text: "Unverified" })
        }

        if (!hasFiles) {
            checks.push({ label: "Sensor / Modality", ok: null, text: "Pending" })
        } else if (!inspection) {
            checks.push({ label: "Sensor / Modality", ok: null, text: inspecting ? "Detecting…" : "Pending" })
        } else {
            const known = inspection.files.every(
                (item) => item.modality?.modality && item.modality.modality !== "unknown"
            )

            checks.push({
                label: "Sensor / Modality",
                ok: known ? true : null,
                text: known ? "Detected" : "Needs Input"
            })
        }

        if (!hasFiles) {
            checks.push({ label: "Geospatial Metadata", ok: null, text: "Pending" })
        } else if (!inspection) {
            checks.push({ label: "Geospatial Metadata", ok: null, text: inspecting ? "Reading…" : "Pending" })
        } else {
            const allValid = inspection.files.every((item) => item.valid)
            const anyGeo = inspection.files.some((item) => item.georeferenced)

            checks.push({
                label: "Geospatial Metadata",
                ok: allValid ? true : false,
                text: !allValid ? "Invalid" : anyGeo ? "Valid" : "Not Georeferenced"
            })
        }

        if (files.length < 2) {
            checks.push({
                label: "Spatial Compatibility",
                ok: hasFiles ? true : null,
                text: hasFiles ? "N/A" : "Pending"
            })
        } else if (!inspection) {
            checks.push({ label: "Spatial Compatibility", ok: null, text: inspecting ? "Checking…" : "Pending" })
        } else if (inspection.crs_status === "matching") {
            checks.push({ label: "Spatial Compatibility", ok: true, text: "Compatible" })
        } else if (inspection.crs_status === "different") {
            checks.push({ label: "Spatial Compatibility", ok: false, text: "Mismatch" })
        } else {
            checks.push({ label: "Spatial Compatibility", ok: null, text: "Unverified" })
        }

        if (files.length < 2) {
            checks.push({
                label: "Temporal Metadata",
                ok: hasFiles ? true : null,
                text: hasFiles ? "N/A" : "Pending"
            })
        } else {
            // Acquisition timestamps aren't extracted from the
            // rasters yet, so two images are honestly "unverified"
            // rather than a fabricated pass.
            checks.push({ label: "Temporal Metadata", ok: null, text: "Unverified" })
        }

        if (!hasFiles || !trimmedQuery) {
            checks.push({ label: "Task Compatibility", ok: null, text: "Pending" })
        } else if (!inspection?.task_preview) {
            checks.push({ label: "Task Compatibility", ok: null, text: inspecting ? "Analyzing…" : "Pending" })
        } else if (inspection.task_preview.task === "unknown") {
            checks.push({ label: "Task Compatibility", ok: false, text: "Uncertain" })
        } else if (inspection.task_preview.confidence >= 0.55) {
            checks.push({ label: "Task Compatibility", ok: true, text: "Likely Match" })
        } else {
            checks.push({ label: "Task Compatibility", ok: true, text: "Possible Match" })
        }

        const allGood =
            hasFiles &&
            trimmedQuery.length > 0 &&
            checks.every((check) => check.ok !== false)

        return { checks, allGood }
    }, [files, query, inspection, inspecting, taskOverride])

    const configuration = useMemo(() => {
        if (files.length === 0) return "Not set"
        if (inspection?.configuration) return inspection.configuration
        if (files.length === 1) return "Single Image"
        if (files.length === 4) return "Optical + SAR Bi-temporal"

        const modalities = files.map((item, index) => displayModality(item, index))

        if (modalities.includes("optical") && modalities.includes("sar")) {
            return "Optical + SAR Pair"
        }

        return "Two Images (Bi-temporal)"
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [files, inspection])

    const resolutionSummary = inspection?.resolution_summary
        || (files.length ? "Not available" : "—")

    const onSubmit = async (data) => {
        if (!files.length) {
            toast.error("Please upload at least one image.")
            return
        }

        if (files.length > MAX_IMAGES) {
            toast.error(`Maximum ${MAX_IMAGES} images allowed.`)
            return
        }

        if (taskOverride === "optical_sar" && files.length !== 4) {
            toast.error("Optical-SAR analysis requires 4 images: optical T1, optical T2, SAR T1, SAR T2.")
            return
        }

        if (taskOverride === "change_detection" && files.length !== 2) {
            toast.error("Change Detection requires exactly 2 images.")
            return
        }

        if (["vqa", "captioning"].includes(taskOverride) && files.length !== 1) {
            toast.error(`${selectedTaskLabel} requires exactly 1 image.`)
            return
        }

        if (taskOverride === "auto" && ![1, 2, 4].includes(files.length)) {
            toast.error("Use 1 image, 2 images, or 4 images depending on the analysis type.")
            return
        }

        if (taskOverride === "auto" && files.length === 4) {
            const detected = files.map((item, index) => displayModality(item, index))
            if (detected.filter((m) => m === "optical").length !== 2 || detected.filter((m) => m === "sar").length !== 2) {
                toast.error("Four-image Optical-SAR analysis needs 2 optical and 2 SAR images. Correct the modality tags before analyzing.")
                return
            }
        }

        if (taskOverride === "auto" && files.length === 2) {
            const detected = files.map((item, index) => displayModality(item, index))
            if (detected.includes("optical") && detected.includes("sar")) {
                toast.error("The updated Optical-SAR model needs four images: optical T1/T2 and SAR T1/T2.")
                return
            }
        }

        try {
            setSubmitting(true)

            const payload = {
                query: data.query,
                files: files.map((item) => item.file),
                modalities: files.map((item) =>
                    item.modality === "unknown" ? "" : item.modality
                ),
                task: taskOverride === "auto" ? undefined : taskOverride
            }

            const response = await startAnalysis(payload)

            props.setAnalysisId(response.analysis_id)
            props.setResult(null)

            props.setInputData({
                query: data.query,
                previews: files.map((item, index) => ({
                    name: item.file.name,
                    url: item.url,
                    modality: displayModality(item, index),
                    size: item.file.size
                })),
                configuration,
                task: taskOverride
            })

            props.setStatus("processing")

            toast.success("Analysis started.")
        } catch (error) {
            toast.error(error.message || "Unable to start analysis.")
        } finally {
            setSubmitting(false)
        }
    }

    const selectedTaskLabel = TASK_OPTIONS.find((option) => option.value === taskOverride)?.label
        ?? "Auto (Recommended)"

    return (
        <div className='relative px-4 pb-6 pt-16 sm:px-6 lg:px-8 lg:pt-6 xl:px-10'>

            {/* Ambient background glow */}
            <div className='pointer-events-none absolute inset-x-0 top-0 -z-10 h-72 bg-[radial-gradient(ellipse_at_top,_rgba(34,211,238,0.08),_transparent_70%)]' />

            <div className='mx-auto max-w-7xl'>

                <div className='mb-7 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between'>
                    <div>
                        <h1 className='flex items-center gap-2 text-2xl font-semibold sm:text-3xl'>
                            New Analysis
                            <span className='text-cyan-300'>✨</span>
                        </h1>
                        <p className='mt-2 max-w-2xl text-sm text-slate-400'>
                            Upload satellite image(s) and ask anything about your Earth.
                        </p>
                    </div>
                </div>

                <form
                    onSubmit={handleSubmit(onSubmit)}
                    className='grid gap-6 xl:grid-cols-[1.7fr_1fr]'
                >
                    <div className='space-y-6'>

                        {/* Upload card */}
                        <section className='overflow-hidden rounded-2xl border border-white/10 bg-linear-to-b from-white/[0.05] to-white/[0.02] p-5 shadow-xl shadow-black/20 sm:p-7'>
                            <div className='mb-5 flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between'>
                                <div className='flex items-center gap-2.5'>
                                    <div className='flex h-8 w-8 items-center justify-center rounded-lg bg-cyan-400/10 text-cyan-300'>
                                        <UploadCloud size={16} />
                                    </div>
                                    <h2 className='flex items-center gap-1.5 text-sm font-semibold text-white'>
                                        1. Upload Satellite Image(s)
                                        <Info
                                            size={13}
                                            className='text-slate-600'
                                            title='Upload 1 image for VQA/Captioning, 2 for Change Detection, or 4 for Optical-SAR (Optical T1/T2 + SAR T1/T2).'
                                        />
                                    </h2>
                                </div>
                                <span className='flex items-center gap-1.5 text-xs text-slate-500'>
                                    Supported formats: GeoTIFF, TIFF, PNG, JPEG
                                    <Info
                                        size={13}
                                        className='text-slate-600'
                                        title='GeoTIFF/TIFF files are inspected for CRS and resolution. PNG/JPEG are treated as non-georeferenced.'
                                    />
                                </span>
                            </div>

                            <label
                                onDragOver={(event) => {
                                    event.preventDefault()
                                    setDragging(true)
                                }}
                                onDragLeave={() => setDragging(false)}
                                onDrop={(event) => {
                                    event.preventDefault()
                                    setDragging(false)
                                    addFiles(event.dataTransfer.files)
                                }}
                                className={`group relative flex min-h-56 cursor-pointer flex-col items-center justify-center overflow-hidden rounded-2xl border border-dashed p-6 text-center transition-all duration-300 ${
                                    dragging
                                        ? "border-cyan-400 bg-cyan-400/10 scale-[1.01]"
                                        : "border-white/15 bg-black/25 hover:border-cyan-400/40 hover:bg-black/35"
                                }`}
                            >
                                <div className='pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_center,_rgba(34,211,238,0.06),_transparent_65%)] opacity-0 transition-opacity duration-300 group-hover:opacity-100' />

                                <input
                                    type='file'
                                    multiple
                                    accept={SUPPORTED_TYPES.join(",")}
                                    className='hidden'
                                    onChange={(event) => {
                                        addFiles(event.target.files)
                                        event.target.value = ""
                                    }}
                                />

                                <div className='relative flex h-16 w-16 items-center justify-center rounded-2xl bg-cyan-400/5 ring-1 ring-cyan-400/30'>
                                    <UploadCloud className='h-7 w-7 text-cyan-300' />
                                </div>

                                <p className='mt-5 text-sm font-medium text-white'>
                                    Drag &amp; drop your satellite image(s) here
                                </p>

                                <p className='mt-1 text-xs text-slate-500'>or</p>

                                <span className='mt-4 rounded-xl bg-linear-to-r from-cyan-500 to-blue-600 px-5 py-2.5 text-xs font-semibold text-white shadow-lg shadow-cyan-500/20 transition group-hover:shadow-cyan-500/30'>
                                    Browse Files
                                </span>
                            </label>

                            <div className='mt-4 flex flex-wrap gap-2'>
                                {[
                                    "Single Image (Optical / SAR)",
                                    "Optical + SAR Pair",
                                    "Two Images (Different Dates)"
                                ].map((chip) => (
                                    <span
                                        key={chip}
                                        className='rounded-full border border-white/10 bg-white/[0.03] px-3 py-1.5 text-[11px] text-slate-400'
                                    >
                                        {chip}
                                    </span>
                                ))}
                            </div>

                            {files.length > 0 && (
                                <div className='mt-6 border-t border-white/5 pt-5'>
                                    <div className='mb-3 flex items-center justify-between'>
                                        <h3 className='text-sm font-medium text-slate-200'>
                                            Uploaded Images
                                            <span className='ml-1.5 text-slate-500'>({files.length})</span>
                                            {inspecting && (
                                                <Loader2 className='ml-2 inline h-3.5 w-3.5 animate-spin text-cyan-300' />
                                            )}
                                        </h3>
                                        <button
                                            type='button'
                                            onClick={() => setFiles([])}
                                            className='text-xs font-medium text-red-400 hover:text-red-300'
                                        >
                                            Clear All
                                        </button>
                                    </div>

                                    <div className='grid grid-cols-2 gap-3 sm:grid-cols-3'>
                                        {files.map((item, index) => {
                                            const meta = getFileMeta(index)
                                            const modalityLabel = displayModality(item, index)
                                            const dateLabel = formatFileDate(item.file.lastModified)

                                            return (
                                                <div
                                                    key={item.id}
                                                    className='group relative overflow-hidden rounded-2xl border border-white/10 bg-black/30 shadow-lg shadow-black/30'
                                                >
                                                    <div className='absolute left-2.5 top-2.5 z-10 flex h-6 w-6 items-center justify-center rounded-full bg-blue-500 text-[11px] font-semibold text-white shadow'>
                                                        {index + 1}
                                                    </div>

                                                    <button
                                                        type='button'
                                                        onClick={() => removeFile(item.id)}
                                                        className='absolute right-2.5 top-2.5 z-10 rounded-lg bg-black/70 p-1.5 text-slate-300 opacity-0 transition group-hover:opacity-100 hover:text-white'
                                                    >
                                                        <X className='h-3.5 w-3.5' />
                                                    </button>

                                                    <img
                                                        src={item.url}
                                                        alt={item.file.name}
                                                        className='h-36 w-full object-cover transition duration-300 group-hover:scale-105'
                                                    />

                                                    <div className='absolute inset-0 bg-linear-to-t from-black/80 via-transparent to-transparent' />

                                                    <button
                                                        type='button'
                                                        onClick={() => cycleModality(item.id)}
                                                        title='Click to override the detected modality'
                                                        className={`absolute bottom-16 left-2.5 rounded-md px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-white shadow ${MODALITY_STYLES[modalityLabel]}`}
                                                    >
                                                        {modalityLabel}
                                                    </button>

                                                    <div className='absolute inset-x-0 bottom-0 p-2.5'>
                                                        <p className='truncate text-xs font-medium text-slate-100'>
                                                            {item.file.name}
                                                        </p>
                                                        <div className='mt-1 flex flex-wrap items-center gap-x-2.5 gap-y-0.5 text-[10px] text-slate-400'>
                                                            <span className='flex items-center gap-1'>
                                                                <HardDrive size={10} />
                                                                {formatMB(item.file.size)} MB
                                                            </span>
                                                            {meta?.resolution_x && (
                                                                <span className='flex items-center gap-1'>
                                                                    <Ruler size={10} />
                                                                    {meta.resolution_x}
                                                                    {meta.resolution_unit === "m" ? "m" : "°"}/px
                                                                </span>
                                                            )}
                                                            {dateLabel && (
                                                                <span className='flex items-center gap-1'>
                                                                    <CalendarDays size={10} />
                                                                    {dateLabel}
                                                                </span>
                                                            )}
                                                        </div>
                                                    </div>
                                                </div>
                                            )
                                        })}

                                        {files.length < MAX_IMAGES && (
                                            <label className='flex h-full min-h-40 cursor-pointer flex-col items-center justify-center gap-2 rounded-2xl border border-dashed border-white/15 bg-black/20 text-slate-500 transition hover:border-cyan-400/40 hover:bg-black/30 hover:text-cyan-300'>
                                                <input
                                                    type='file'
                                                    multiple
                                                    accept={SUPPORTED_TYPES.join(",")}
                                                    className='hidden'
                                                    onChange={(event) => {
                                                        addFiles(event.target.files)
                                                        event.target.value = ""
                                                    }}
                                                />
                                                <div className='flex h-9 w-9 items-center justify-center rounded-full border border-current'>
                                                    <Plus className='h-4 w-4' />
                                                </div>
                                                <span className='text-xs font-medium'>Add more images</span>
                                            </label>
                                        )}
                                    </div>

                                    <p className='mt-3 flex items-center gap-1.5 text-[11px] text-slate-600'>
                                        <Info size={11} />
                                        Up to {MAX_IMAGES} images · Optical-SAR uses 4 inputs: Optical T1, Optical T2, SAR T1, SAR T2 · use the modality tag to override detection
                                    </p>
                                </div>
                            )}
                        </section>

                        {/* Question card */}
                        <section className='rounded-2xl border border-white/10 bg-linear-to-b from-white/5 to-white/2 p-5 shadow-xl shadow-black/20 sm:p-7'>
                            <div className='mb-4 flex items-center justify-between'>
                                <div className='flex items-center gap-2.5'>
                                    <div className='flex h-8 w-8 items-center justify-center rounded-lg bg-purple-400/10 text-purple-300'>
                                        <MessageSquareText size={16} />
                                    </div>
                                    <h2 className='flex items-center gap-1.5 text-sm font-semibold text-white'>
                                        2. Ask Your Question
                                        <Info
                                            size={13}
                                            className='text-slate-600'
                                            title='Your question is used to auto-route the analysis to the right model (VQA, captioning, change detection, etc).'
                                        />
                                    </h2>
                                </div>
                                <button
                                    type='button'
                                    onClick={() => setShowExamples((value) => !value)}
                                    className='flex items-center gap-1 text-xs font-medium text-cyan-300 hover:text-cyan-200'
                                >
                                    Examples
                                    <ChevronDown size={13} className={`transition ${showExamples ? "rotate-180" : ""}`} />
                                </button>
                            </div>

                            {showExamples && (
                                <div className='mb-4 space-y-1 rounded-xl border border-white/10 bg-black/25 p-2'>
                                    {EXAMPLE_QUERIES.map((example) => (
                                        <button
                                            type='button'
                                            key={example}
                                            onClick={() => {
                                                setValue("query", example, { shouldValidate: true })
                                                setShowExamples(false)
                                            }}
                                            className='block w-full rounded-lg px-3 py-2 text-left text-xs text-slate-400 transition hover:bg-white/5 hover:text-slate-200'
                                        >
                                            {example}
                                        </button>
                                    ))}
                                </div>
                            )}

                            <textarea
                                {...register("query", {
                                    required: "Please enter a question.",
                                    maxLength: { value: 1000, message: "Question must be under 1000 characters." }
                                })}
                                placeholder='e.g., What changed between these two images? Is there any new construction? Estimate the extent of change.'
                                rows={5}
                                maxLength={1000}
                                className='w-full resize-none rounded-xl border border-white/10 bg-black/25 p-4 text-sm text-white outline-none transition placeholder:text-slate-600 focus:border-cyan-400/50 focus:ring-2 focus:ring-cyan-400/10'
                            />

                            {errors.query && (
                                <p className='mt-2 text-xs text-red-400'>{errors.query.message}</p>
                            )}

                            <div className='mt-2 text-right text-xs text-slate-600'>
                                {query.length} / 1000
                            </div>

                            <div className='mt-3 flex flex-col gap-2.5 sm:flex-row sm:items-center'>
                                

                                <button
                                    type='submit'
                                    disabled={submitting}
                                    className='flex flex-1 items-center justify-center gap-2 rounded-xl bg-linear-to-r from-cyan-500 to-blue-600 px-5 py-3.5 text-sm font-semibold text-white shadow-lg shadow-cyan-500/20 transition hover:scale-[1.01] hover:shadow-cyan-500/30 disabled:cursor-not-allowed disabled:opacity-60 disabled:hover:scale-100 cursor-pointer'
                                >
                                    <Send className='h-4 w-4' />
                                    {submitting ? "Starting…" : "Analyze Images"}
                                </button>
                            </div>

                            <p className='mt-2 text-center text-xs text-slate-600'>
                                This may take a few minutes depending on the task.
                            </p>
                        </section>
                    </div>


                    <div className='space-y-6'>
                        <div className='relative overflow-hidden rounded-2xl border border-white/10 bg-linear-to-b from-white/5 to-white/2 p-5 shadow-xl shadow-black/20'>
                            <h2 className='mb-4 flex items-center gap-2 text-sm font-semibold text-white'>
                                <Layers3 size={15} className='text-cyan-300' />
                                Input Summary
                            </h2>

                            <div className='relative z-10 space-y-3.5 text-sm'>
                                <SummaryRow icon={Layers3} label='Images' value={files.length || "—"} />
                                <SummaryRow icon={Gauge} label='Configuration' value={configuration} />
                                <SummaryRow icon={Ruler} label='Spatial Resolution' value={resolutionSummary} />
                                <SummaryRow
                                    icon={MapPin}
                                    label='Area of Interest'
                                    value={(
                                        <button
                                            type='button'
                                            onClick={() =>
                                                toast.info("Area-of-interest selection is coming in a future update.")
                                            }
                                            className='flex items-center gap-1 text-slate-200 hover:text-cyan-300'
                                        >
                                            Not set
                                            <Pencil size={11} />
                                        </button>
                                    )}
                                />
                                <SummaryRow icon={Timer} label='Est. Processing Time' value='2 – 4 min' />
                            </div>

                            <Satellite className='pointer-events-none absolute -bottom-4 -right-4 h-24 w-24 rotate-12 text-cyan-400/10' />
                        </div>

                        <div className='rounded-2xl border border-white/10 bg-linear-to-b from-white/5 to-white/2 p-5 shadow-xl shadow-black/20'>
                            <h2 className='mb-4 text-sm font-semibold text-white'>Validation Status</h2>

                            <div className='space-y-3'>
                                {validation.checks.map((check) => (
                                    <div key={check.label} className='flex items-center justify-between text-sm'>
                                        <span className='text-slate-400'>{check.label}</span>

                                        {check.ok === true && (
                                            <span className='flex items-center gap-1.5 text-xs font-medium text-emerald-300'>
                                                <CheckCircle2 className='h-4 w-4' /> {check.text}
                                            </span>
                                        )}
                                        {check.ok === false && (
                                            <span className='flex items-center gap-1.5 text-xs font-medium text-red-400'>
                                                <XCircle className='h-4 w-4' /> {check.text}
                                            </span>
                                        )}
                                        {check.ok === null && (
                                            <span className='flex items-center gap-1.5 text-xs text-slate-500'>
                                                <MinusCircle className='h-3.5 w-3.5' /> {check.text}
                                            </span>
                                        )}
                                    </div>
                                ))}
                            </div>

                            <div
                                className={`mt-4 rounded-xl border px-3.5 py-3 text-xs font-medium ${
                                    validation.allGood
                                        ? "border-emerald-400/25 bg-emerald-400/10 text-emerald-300"
                                        : "border-white/10 bg-black/20 text-slate-500"
                                }`}
                            >
                                {validation.allGood
                                    ? "✓ All good! You're ready to analyze."
                                    : "Upload image(s) and enter a question to continue."}
                            </div>
                        </div>

                        <div className='rounded-2xl border border-purple-400/15 bg-linear-to-b from-purple-500/6 to-transparent p-5'>
                            <div className='flex items-start gap-3'>
                                <div className='flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-purple-400/10 text-purple-300'>
                                    <BookOpen size={16} />
                                </div>
                                <div>
                                    <h2 className='text-sm font-semibold text-white'>Need Help?</h2>
                                    <p className='mt-2 text-xs leading-5 text-slate-500'>
                                        VQA and Captioning need exactly 1 image. Change Detection needs exactly 2.
                                        Optical-SAR uses four images: Optical T1/T2 and SAR T1/T2.
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                </form>
            </div>
        </div>
    )
}

const SummaryRow = ({ icon: Icon, label, value }) => (
    <div className='flex items-center justify-between gap-3'>
        <dt className='flex items-center gap-2 text-slate-500'>
            <Icon size={13} className='text-slate-600' />
            {label}
        </dt>
        <dd className='max-w-40 truncate text-right font-medium text-slate-200'>{value}</dd>
    </div>
)

export default InputState