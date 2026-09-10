import { useEffect, useState } from "react"
import {
    AlertTriangle,
    CheckCircle2,
    Clock3,
    FlaskConical,
    Info,
    RefreshCw,
    Play,
    XCircle
} from "lucide-react"
import { getModelMetrics, getModelMetricsStatus, runModelMetrics } from "../../lib/api"
import { formatMetricLabel, formatMetricValue } from "../../utils/formatMetric"

const STATUS_CONFIG = {
    evaluated: {
        label: "Evaluated",
        icon: CheckCircle2,
        classes: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
    },
    evaluated_baseline: {
        label: "Baseline (untrained)",
        icon: FlaskConical,
        classes: "border-amber-500/30 bg-amber-500/10 text-amber-300"
    },
    pending: {
        label: "Evaluation pending",
        icon: Clock3,
        classes: "border-slate-500/30 bg-slate-500/10 text-slate-300"
    },
    error: {
        label: "Evaluator error",
        icon: XCircle,
        classes: "border-red-500/30 bg-red-500/10 text-red-300"
    }
}

const ModelMetricsContent = () => {
    const [data, setData] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState("")
    const [running, setRunning] = useState(false)

    const fetchMetrics = async () => {
        try {
            setLoading(true)
            setError("")

            const result = await getModelMetrics()

            setData(result)
        } catch (fetchError) {
            console.error(fetchError)
            setError(fetchError.message || "Backend unavailable")
        } finally {
            setLoading(false)
        }
    }

    const runEvaluation = async () => {
        try {
            setRunning(true)
            setError("")
            await runModelMetrics()

            // Poll while the backend evaluates the current checkpoints.
            const startedAt = Date.now()
            const poll = async () => {
                await fetchMetrics()
                const status = await getModelMetricsStatus()

                if (status.status === "running" && Date.now() - startedAt < 10 * 60 * 1000) {
                    window.setTimeout(poll, 1500)
                } else {
                    setRunning(false)
                    await fetchMetrics()
                }
            }

            window.setTimeout(poll, 500)
        } catch (runError) {
            setRunning(false)
            setError(runError.message || "Could not start evaluation")
        }
    }

    useEffect(() => {
        fetchMetrics()
    }, [])

    const models = data?.models || []

    const evaluatedCount = models.filter(
        (model) => model.status === "evaluated" || model.status === "evaluated_baseline"
    ).length

    const pendingCount = models.filter((model) => model.status === "pending").length

    const formatTimestamp = (value) => {
        if (!value) return "Never run"

        const date = new Date(value)

        if (Number.isNaN(date.getTime())) return "Never run"

        return date.toLocaleString("en-IN", {
            day: "numeric",
            month: "short",
            year: "numeric",
            hour: "numeric",
            minute: "2-digit"
        })
    }

    return (
        <main className="min-w-0 overflow-hidden bg-[#020611] p-3 pt-16 text-white sm:p-4 sm:pt-16 lg:p-6">
            <div className="mb-5 flex flex-col gap-4 sm:mb-6 lg:flex-row lg:items-center lg:justify-between">
                <div>
                    <div className="mb-1 flex items-center gap-2 text-xs text-slate-500 sm:text-sm">
                        <span>Model Hub</span>
                        <span>/</span>
                        <span className="text-slate-400">Performance</span>
                    </div>

                    <h1 className="text-xl font-semibold tracking-tight sm:text-2xl lg:text-3xl">
                        Model Performance
                    </h1>

                    <p className="mt-1 text-sm text-slate-400">
                        Evaluation metrics for every model in the AERIS pipeline.
                    </p>
                </div>

                <div className="flex flex-wrap items-center gap-3">
                    <span className="text-xs text-slate-500">
                        Last run: {formatTimestamp(data?.generated_at)}
                    </span>

                    <button
                        onClick={fetchMetrics}
                        className="flex cursor-pointer items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/3 px-3 py-2 text-sm text-slate-300 transition hover:bg-white/[0.07] hover:text-white"
                    >
                        <RefreshCw size={16} className={loading ? "animate-spin" : ""} />
                        Refresh
                    </button>

                    <button
                        onClick={runEvaluation}
                        disabled={running}
                        className="flex cursor-pointer items-center justify-center gap-2 rounded-xl border border-cyan-400/30 bg-cyan-400/10 px-3 py-2 text-sm font-medium text-cyan-300 transition hover:bg-cyan-400/15 disabled:cursor-not-allowed disabled:opacity-60"
                    >
                        <Play size={15} className={running ? "animate-pulse" : ""} />
                        {running ? "Evaluating…" : "Run Evaluation"}
                    </button>
                </div>
            </div>

            {error && (
                <div className="mb-5 flex flex-col gap-2 rounded-xl border border-amber-500/20 bg-amber-500/5 px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
                    <div className="flex items-center gap-2 text-sm text-amber-200">
                        <AlertTriangle size={16} />
                        {error}
                    </div>
                </div>
            )}

            {!loading && !error && models.length === 0 && (
                <div className="rounded-2xl border border-white/10 bg-white/2.5 p-8 text-center">
                    <p className="text-sm text-slate-400">
                        No evaluation results are available yet. Use <span className="text-cyan-300">Run Evaluation</span> to score the current AERIS checkpoints.
                    </p>
                </div>
            )}

            <div className="mb-5 grid grid-cols-1 gap-3 sm:grid-cols-3">
                <div className="rounded-2xl border border-white/10 bg-white/2.5 p-4 sm:p-5">
                    <p className="text-xs text-slate-500">Models tracked</p>
                    <p className="mt-1 text-2xl font-semibold">{models.length}</p>
                </div>

                <div className="rounded-2xl border border-white/10 bg-white/2.5 p-4 sm:p-5">
                    <p className="text-xs text-slate-500">Evaluated</p>
                    <p className="mt-1 text-2xl font-semibold text-emerald-300">
                        {evaluatedCount}
                    </p>
                </div>

                <div className="rounded-2xl border border-white/10 bg-white/2.5 p-4 sm:p-5">
                    <p className="text-xs text-slate-500">Pending</p>
                    <p className="mt-1 text-2xl font-semibold text-slate-300">
                        {pendingCount}
                    </p>
                </div>
            </div>

            {data?.evaluation_status === "running" || running ? (
                <div className="mb-5 flex items-center gap-3 rounded-2xl border border-cyan-400/20 bg-cyan-400/5 p-4 text-sm text-cyan-200">
                    <RefreshCw size={16} className="animate-spin shrink-0" />
                    Evaluating the current AERIS checkpoints. This page will update when the run finishes.
                </div>
            ) : null}

            <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
                {models.map((model) => (
                    <ModelCard key={model.model} model={model} />
                ))}
            </div>
        </main>
    )
}

const ModelCard = ({ model }) => {
    const status = STATUS_CONFIG[model.status] || STATUS_CONFIG.pending
    const StatusIcon = status.icon
    const metricEntries = Object.entries(model.metrics || {}).filter(
        ([, value]) => value !== null && value !== undefined
    )

    return (
        <div className="min-w-0 rounded-2xl border border-white/10 bg-white/2.5 p-4 sm:p-5">
            <div className="mb-3 flex flex-wrap items-start justify-between gap-2">
                <div>
                    <h3 className="text-base font-medium text-white">
                        {model.display_name || model.model}
                    </h3>
                    <p className="mt-0.5 text-xs text-slate-500">{model.model_type}</p>
                </div>

                <span
                    className={`flex shrink-0 items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium ${status.classes}`}
                >
                    <StatusIcon size={13} />
                    {status.label}
                </span>
            </div>

            {model.methodology && (
                <div className="mb-4 flex gap-2 rounded-xl border border-white/5 bg-black/20 p-3 text-xs leading-relaxed text-slate-400">
                    <Info size={14} className="mt-0.5 shrink-0 text-slate-500" />
                    <span>{model.methodology}</span>
                </div>
            )}

            {model.checkpoint && (
                <div className="mb-3 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-slate-500">
                    <span>Checkpoint: <span className="text-slate-400">{model.checkpoint}</span></span>
                    {model.best_epoch != null && <span>Best epoch: <span className="text-slate-400">{model.best_epoch}</span></span>}
                </div>
            )}

            {model.evaluation_basis && (
                <div className="mb-4 rounded-xl border border-cyan-400/10 bg-cyan-400/5 px-3 py-2 text-xs leading-relaxed text-slate-400">
                    <span className="text-slate-300">Evaluation basis:</span> {model.evaluation_basis}
                </div>
            )}

            {metricEntries.length > 0 && (
                <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3">
                    {metricEntries.map(([key, value]) => (
                        <div
                            key={key}
                            className="rounded-xl border border-white/5 bg-black/20 p-3"
                        >
                            <p className="text-[11px] text-slate-500">
                                {formatMetricLabel(key)}
                            </p>
                            <p className="mt-1 text-lg font-semibold text-cyan-300">
                                {formatMetricValue(key, value)}
                            </p>
                        </div>
                    ))}
                </div>
            )}

            {model.status === "pending" && model.pending_reason && (
                <div className="mt-3 flex gap-2 rounded-xl border border-slate-500/20 bg-slate-500/5 p-3 text-xs leading-relaxed text-slate-400">
                    <Clock3 size={14} className="mt-0.5 shrink-0 text-slate-500" />
                    <span>{model.pending_reason}</span>
                </div>
            )}

            {model.status === "error" && model.error && (
                <div className="mt-3 flex gap-2 rounded-xl border border-red-500/20 bg-red-500/5 p-3 text-xs leading-relaxed text-red-300">
                    <XCircle size={14} className="mt-0.5 shrink-0" />
                    <span>{model.error}</span>
                </div>
            )}

            <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-slate-500">
                {typeof model.sample_size === "number" && (
                    <span>Sample size: {model.sample_size}</span>
                )}
                {typeof model.skipped === "number" && model.skipped > 0 && (
                    <span>Skipped: {model.skipped}</span>
                )}
                {typeof model.failed_runs === "number" && model.failed_runs > 0 && (
                    <span>Failed runs: {model.failed_runs}</span>
                )}
            </div>
        </div>
    )
}

export default ModelMetricsContent
