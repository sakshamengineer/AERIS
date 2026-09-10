import { useEffect, useRef, useState } from 'react'
import {
    CheckCircle2,
    Loader2,
    Circle,
    XCircle,
    Bot
} from 'lucide-react'
import { toast } from 'sonner'
import { getAnalysisStatus, getAnalysisResult } from '../../lib/api'

const POLL_INTERVAL_MS = 1200

const STEP_DESCRIPTIONS = {
    "Input Validation": "Checking file formats, metadata and compatibility.",
    "Metadata Extraction": "Extracting CRS, resolution and sensor info.",
    "Query Understanding": "Understanding your question and identifying the task.",
    "Model Selection": "Selecting the most suitable model(s) and tools.",
    "Model Execution": "Running the specialist model(s) to generate results.",
    "Evidence Generation": "Preparing visual evidence for the result.",
    "Result Generation": "Assembling the final answer and confidence score."
}

const formatTime = (date) =>
    date.toLocaleTimeString("en-IN", {
        hour: "numeric",
        minute: "2-digit",
        second: "2-digit"
    })

const ProcessingState = ({ analysisId, inputData, setStatus, setResult }) => {
    const [job, setJob] = useState(null)
    const [trace, setTrace] = useState([])
    const [error, setError] = useState("")

    const seenSteps = useRef(new Set())
    const finished = useRef(false)

    useEffect(() => {
        if (!analysisId) return

        finished.current = false
        seenSteps.current = new Set()

        const poll = async () => {
            try {
                const data = await getAnalysisStatus(analysisId)

                setJob(data);
                (data.steps || []).forEach((step) => {
                    const key = `${step.name}:${step.status}`

                    if (!seenSteps.current.has(key) && step.status !== "pending") {
                        seenSteps.current.add(key)

                        setTrace((previous) => [
                            ...previous,
                            {
                                key,
                                label: step.name,
                                status: step.status,
                                time: formatTime(new Date())
                            }
                        ])
                    }
                })

                if (data.status === "completed" && !finished.current) {
                    finished.current = true

                    const result = await getAnalysisResult(analysisId)

                    setResult(result)
                    setStatus("result")
                }

                if (data.status === "failed" && !finished.current) {
                    finished.current = true
                    setError(data.error || "Analysis failed.")
                    toast.error(data.error || "Analysis failed.")
                }
            } catch (err) {
                setError(err.message || "Unable to reach the backend.")
            }
        }

        poll()
        const interval = setInterval(() => {
            if (!finished.current) poll()
        }, POLL_INTERVAL_MS)

        return () => clearInterval(interval)
    }, [analysisId, setResult, setStatus])

    const steps = job?.steps || []
    const progress = job?.progress ?? 0
    const currentStepName = job?.step || "Preparing Analysis"

    return (
        <div className='px-4 pb-5 pt-16 sm:px-6 lg:px-8 lg:pt-5 xl:px-10'>
            <div className='mx-auto max-w-7xl'>

                <div className='mb-6'>
                    <h1 className='flex items-center gap-2 text-2xl font-semibold sm:text-3xl'>
                        Analysis in Progress
                        <span className='text-cyan-300'>✨</span>
                    </h1>
                    <p className='mt-2 text-sm text-slate-400'>
                        Our AI agent is analyzing your satellite data. This may take a few minutes.
                    </p>
                </div>

                {error && (
                    <div className='mb-5 flex flex-col gap-2 rounded-xl border border-red-500/20 bg-red-500/5 px-4 py-3 sm:flex-row sm:items-center sm:justify-between'>
                        <p className='text-sm text-red-300'>{error}</p>
                        <button
                            onClick={() => setStatus("input")}
                            className='w-fit rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-slate-300 hover:bg-white/10'
                        >
                            Back to New Analysis
                        </button>
                    </div>
                )}

                <div className='grid gap-5 xl:grid-cols-[1.7fr_1fr]'>
                    <div className='space-y-5'>

                        {/* Input summary */}
                        <div className='rounded-2xl border border-white/10 bg-white/[0.03] p-5'>
                            <h2 className='mb-4 text-sm font-medium text-white'>Input Summary</h2>

                            <div className='flex flex-col gap-4 sm:flex-row'>
                                <div className='flex gap-3'>
                                    {(inputData?.previews || []).map((preview) => (
                                        <div
                                            key={preview.name}
                                            className='relative h-20 w-20 shrink-0 overflow-hidden rounded-xl border border-white/10'
                                        >
                                            <img
                                                src={preview.url}
                                                alt={preview.name}
                                                className='h-full w-full object-cover'
                                            />
                                            <span className='absolute bottom-0 left-0 right-0 bg-black/60 px-1 py-0.5 text-center text-[9px] uppercase text-slate-200'>
                                                {preview.modality}
                                            </span>
                                        </div>
                                    ))}
                                </div>

                                <div className='grid flex-1 grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-2'>
                                    <div>
                                        <p className='text-xs text-slate-500'>Configuration</p>
                                        <p className='text-slate-200'>{inputData?.configuration || "—"}</p>
                                    </div>
                                    <div>
                                        <p className='text-xs text-slate-500'>Images</p>
                                        <p className='text-slate-200'>{inputData?.previews?.length ?? "—"}</p>
                                    </div>
                                    <div className='col-span-2'>
                                        <p className='text-xs text-slate-500'>Your Question</p>
                                        <p className='mt-1 text-slate-300'>{inputData?.query || "—"}</p>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Agent activity */}
                        <div className='rounded-2xl border border-white/10 bg-white/[0.03] p-5'>
                            <div className='mb-4 flex items-center justify-between'>
                                <h2 className='text-sm font-medium text-white'>Agent Activity</h2>
                                <span className='flex items-center gap-1.5 text-xs text-cyan-300'>
                                    <span className='h-1.5 w-1.5 animate-pulse rounded-full bg-cyan-400' />
                                    {job?.status === "completed" ? "Done" : "Working…"}
                                </span>
                            </div>

                            <div className='grid grid-cols-2 gap-3 sm:grid-cols-4'>
                                {steps.map((step) => (
                                    <div
                                        key={step.name}
                                        className={`rounded-xl border p-3 text-center transition ${
                                            step.status === "completed"
                                                ? "border-emerald-400/20 bg-emerald-400/5"
                                                : step.status === "running"
                                                ? "border-purple-400/30 bg-purple-400/10"
                                                : step.status === "failed"
                                                ? "border-red-400/30 bg-red-400/10"
                                                : "border-white/10 bg-black/20"
                                        }`}
                                    >
                                        <div className='mb-2 flex justify-center'>
                                            {step.status === "completed" && (
                                                <CheckCircle2 className='h-5 w-5 text-emerald-400' />
                                            )}
                                            {step.status === "running" && (
                                                <Loader2 className='h-5 w-5 animate-spin text-purple-300' />
                                            )}
                                            {step.status === "failed" && (
                                                <XCircle className='h-5 w-5 text-red-400' />
                                            )}
                                            {step.status === "pending" && (
                                                <Circle className='h-5 w-5 text-slate-600' />
                                            )}
                                        </div>
                                        <p className='text-[11px] text-slate-300'>{step.name}</p>
                                    </div>
                                ))}
                            </div>

                            <div className='mt-5'>
                                <div className='mb-1.5 flex items-center justify-between text-xs text-slate-500'>
                                    <span>{currentStepName}</span>
                                    <span>{progress}%</span>
                                </div>
                                <div className='h-1.5 overflow-hidden rounded-full bg-white/5'>
                                    <div
                                        className='h-full rounded-full bg-linear-to-r from-cyan-400 to-purple-500 transition-all duration-500'
                                        style={{ width: `${progress}%` }}
                                    />
                                </div>
                            </div>
                        </div>

                        {/* Current step details */}
                        <div className='rounded-2xl border border-white/10 bg-white/[0.03] p-5'>
                            <div className='flex items-center gap-2'>
                                <Bot className='h-5 w-5 text-purple-400' />
                                <h2 className='font-medium'>{currentStepName}</h2>
                            </div>
                            <p className='mt-3 text-sm leading-6 text-slate-400'>
                                {STEP_DESCRIPTIONS[currentStepName] ||
                                    "The agent is working through your request."}
                            </p>
                        </div>
                    </div>

                    {/* Execution trace */}
                    <div className='space-y-5'>
                        <div className='rounded-2xl border border-white/10 bg-white/[0.03] p-5'>
                            <div className='mb-4 flex items-center justify-between'>
                                <h2 className='text-sm font-medium text-white'>Execution Trace</h2>
                                <span className='flex items-center gap-1.5 text-xs text-emerald-300'>
                                    <span className='h-1.5 w-1.5 rounded-full bg-emerald-400' />
                                    Live
                                </span>
                            </div>

                            <div className='space-y-4'>
                                {trace.length === 0 && (
                                    <p className='text-xs text-slate-600'>Waiting for the agent to start…</p>
                                )}

                                {trace.map((item, index) => (
                                    <div key={`${item.key}-${index}`} className='flex gap-3'>
                                        <div className='flex flex-col items-center'>
                                            <div
                                                className={`h-2.5 w-2.5 rounded-full ${
                                                    item.status === "completed"
                                                        ? "bg-emerald-400"
                                                        : item.status === "failed"
                                                        ? "bg-red-400"
                                                        : "bg-purple-400"
                                                }`}
                                            />
                                            {index < trace.length - 1 && (
                                                <div className='mt-1 h-full w-px flex-1 bg-white/10' />
                                            )}
                                        </div>

                                        <div className='pb-4'>
                                            <p className='text-[11px] text-slate-600'>{item.time}</p>
                                            <p className='text-sm text-slate-200'>
                                                {item.label}
                                                <span className='ml-2 text-xs text-slate-500'>
                                                    ({item.status})
                                                </span>
                                            </p>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default ProcessingState
