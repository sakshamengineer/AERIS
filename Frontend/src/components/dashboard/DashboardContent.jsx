import { useEffect, useState } from "react"
import {
    Activity,
    Brain,
    CheckCircle2,
    Clock3,
    Eye,
    FileText,
    Image,
    Layers,
    MessageSquare,
    Play,
    RefreshCw,
    ShieldCheck,
    Sparkles,
    Target,
    Upload,
    FolderOpen,
    Grid2X2
} from "lucide-react"
import { useNavigate } from "react-router-dom"
import earthSatellite from "../../assets/earth-satellite.jpg"
import { getDashboard } from "../../lib/api"
import { getDisplayConfidence } from "../../utils/Confidence"

const TYPE_COLORS = [
    "#22d3ee", // cyan
    "#a78bfa", // purple
    "#34d399", // emerald
    "#fbbf24", // amber
    "#f472b6"  // pink
]

const DashboardContent = () => {
    const navigate = useNavigate()

    const [dashboard, setDashboard] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState("")

    const fetchDashboard = async () => {
        try {
            setLoading(true)
            setError("")

            const data = await getDashboard()

            setDashboard(data)
        } catch (error) {
            console.error(error)
            setError(error.message || "Backend unavailable")
        } finally {
            setLoading(false)
        }
    }

    useEffect(() => {
        fetchDashboard()
    }, [])

    const stats = dashboard?.stats || {
        total_analyses: 0,
        successful_analyses: 0,
        average_confidence: 0,
        average_processing_time: 0
    }

    const analysisTypes = dashboard?.analysis_types || []
    const recentAnalyses = dashboard?.recent_analyses || []
    const activities = dashboard?.activities || []

    const totalTypeCount = analysisTypes.reduce(
        (sum, item) => sum + item.count,
        0
    )

    const getGreeting = () => {
        const hour = new Date().getHours()

        if (hour < 12) return "Good Morning"
        if (hour < 17) return "Good Afternoon"

        return "Good Evening"
    }

    const getTypeIcon = (task) => {
        if (task?.includes("change")) return Layers
        if (task?.includes("vqa")) return MessageSquare
        if (task?.includes("caption")) return FileText
        if (task?.includes("optical")) return Eye

        return Brain
    }

    const formatDate = (date) => {
        if (!date) return "Unknown"

        const value = new Date(date)

        if (Number.isNaN(value.getTime())) {
            return "Unknown"
        }

        return value.toLocaleString("en-IN", {
            day: "numeric",
            month: "short",
            hour: "numeric",
            minute: "2-digit"
        })
    }

    return (
        <main className="min-w-0 overflow-hidden bg-[#020611] p-3 pt-16 text-white sm:p-4 sm:pt-16 lg:p-6">
            <div className="mb-5 flex flex-col gap-4 sm:mb-6 lg:flex-row lg:items-center lg:justify-between">
                <div>
                    <div className="mb-1 flex items-center gap-2 text-xs text-slate-500 sm:text-sm">
                        <span>Dashboard</span>
                        <span>/</span>
                        <span className="text-slate-400">
                            Overview
                        </span>
                    </div>

                    <h1 className="text-xl font-semibold tracking-tight sm:text-2xl lg:text-3xl">
                        {getGreeting()}
                    </h1>

                    <p className="mt-1 text-sm text-slate-400">
                        Monitor your satellite intelligence workspace.
                    </p>
                </div>

                <div className="flex flex-wrap gap-2">
                    <button
                        onClick={fetchDashboard}
                        className="flex flex-1 cursor-pointer items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/3 px-3 py-2 text-sm text-slate-300 transition hover:bg-white/[0.07] hover:text-white sm:flex-none"
                    >
                        <RefreshCw
                            size={16}
                            className={loading ? "animate-spin" : ""}
                        />
                        Refresh
                    </button>

                    <button
                        onClick={() => navigate("/analysis/new")}
                        className="flex flex-1 cursor-pointer items-center justify-center gap-2 rounded-xl bg-blue-500 px-4 py-2 text-sm font-medium text-white transition hover:bg-blue-400 sm:flex-none"
                    >
                        <PlusIcon />
                        New Analysis
                    </button>
                </div>
            </div>

            {/* Backend status */}
            {error && (
                <div className="mb-5 flex flex-col gap-2 rounded-xl border border-amber-500/20 bg-amber-500/5 px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
                    <div>
                        <p className="text-sm text-amber-300">
                            Backend unavailable
                        </p>

                        <p className="mt-0.5 text-xs text-slate-500">
                            Dashboard layout is available, but live data
                            cannot be loaded right now.
                        </p>
                    </div>

                    <button
                        onClick={fetchDashboard}
                        className="w-fit shrink-0 cursor-pointer rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-slate-300 hover:bg-white/10"
                    >
                        Try Again
                    </button>
                </div>
            )}

            {/* Stats */}
            <div className="mb-5 grid grid-cols-1 gap-3 xs:grid-cols-2 sm:grid-cols-2 xl:grid-cols-4">

                {/* Total Analyses */}
                <div className="rounded-2xl border border-white/10 bg-white/2.5 p-4 sm:p-5">
                    <div className="mb-4 flex items-center justify-between">
                        <div className="rounded-xl bg-blue-500/10 p-2.5 text-blue-300">
                            <Activity size={20} />
                        </div>

                        <span className="text-xs text-slate-500">
                            All time
                        </span>
                    </div>

                    <p className="text-sm text-slate-400">
                        Total Analyses
                    </p>

                    {loading ? (
                        <div className="mt-2 h-8 w-16 animate-pulse rounded bg-white/10" />
                    ) : (
                        <h2 className="mt-1 text-xl font-semibold sm:text-2xl">
                            {stats.total_analyses}
                        </h2>
                    )}
                </div>

                {/* Successful */}
                <div className="rounded-2xl border border-white/10 bg-white/2.5 p-4 sm:p-5">
                    <div className="mb-4 flex items-center justify-between">
                        <div className="rounded-xl bg-emerald-500/10 p-2.5 text-emerald-300">
                            <CheckCircle2 size={20} />
                        </div>

                        <span className="text-xs text-slate-500">
                            Successful
                        </span>
                    </div>

                    <p className="text-sm text-slate-400">
                        Completed Analyses
                    </p>

                    {loading ? (
                        <div className="mt-2 h-8 w-16 animate-pulse rounded bg-white/10" />
                    ) : (
                        <h2 className="mt-1 text-xl font-semibold sm:text-2xl">
                            {stats.successful_analyses}
                        </h2>
                    )}
                </div>

                {/* Confidence */}
                <div className="rounded-2xl border border-white/10 bg-white/2.5 p-4 sm:p-5">
                    <div className="mb-4 flex items-center justify-between">
                        <div className="rounded-xl bg-purple-500/10 p-2.5 text-purple-300">
                            <ShieldCheck size={20} />
                        </div>

                        <span className="text-xs text-slate-500">
                            Average
                        </span>
                    </div>

                    <p className="text-sm text-slate-400">
                        Confidence Score
                    </p>

                    {loading ? (
                        <div className="mt-2 h-8 w-16 animate-pulse rounded bg-white/10" />
                    ) : (
                        <h2 className="mt-1 text-xl font-semibold sm:text-2xl">
                            {stats.average_confidence}%
                        </h2>
                    )}
                </div>

                {/* Processing Time */}
                <div className="rounded-2xl border border-white/10 bg-white/2.5 p-4 sm:p-5">
                    <div className="mb-4 flex items-center justify-between">
                        <div className="rounded-xl bg-cyan-500/10 p-2.5 text-cyan-300">
                            <Clock3 size={20} />
                        </div>

                        <span className="text-xs text-slate-500">
                            Average
                        </span>
                    </div>

                    <p className="text-sm text-slate-400">
                        Processing Time
                    </p>

                    {loading ? (
                        <div className="mt-2 h-8 w-20 animate-pulse rounded bg-white/10" />
                    ) : (
                        <h2 className="mt-1 text-xl font-semibold sm:text-2xl">
                            {stats.average_processing_time}s
                        </h2>
                    )}
                </div>

            </div>

            {/* Hero + Analysis Types */}
            <div className="mb-5 grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_330px]">

                {/* Hero */}
                <div className="relative min-h-60 overflow-hidden rounded-2xl border border-white/10 bg-[#07111f] sm:min-h-70">
                    <img
                        src={earthSatellite}
                        alt="Earth satellite"
                        className="absolute inset-0 h-full w-full object-cover opacity-40"
                    />

                    <div className="absolute inset-0 bg-linear-to-r from-[#020611] via-[#020611]/75 to-transparent" />

                    <div className="relative z-10 flex min-h-60 max-w-2xl flex-col justify-center p-5 sm:min-h-70 sm:p-8">
                        <div className="mb-4 flex w-fit items-center gap-2 rounded-full border border-cyan-400/20 bg-cyan-400/5 px-3 py-1.5 text-xs text-cyan-300">
                            <Sparkles size={14} />
                            AI Earth Intelligence
                        </div>

                        <h2 className="max-w-xl text-xl font-semibold leading-tight sm:text-2xl lg:text-3xl">
                            Turn satellite imagery into actionable intelligence.
                        </h2>

                        <p className="mt-3 max-w-xl text-sm leading-6 text-slate-400">
                            Ask natural-language questions about your imagery
                            and let SatQuery AI select and execute the right
                            remote-sensing workflow.
                        </p>

                        <button
                            onClick={() => navigate("/analysis/new")}
                            className="mt-6 flex w-fit cursor-pointer items-center gap-2 rounded-xl bg-white px-4 py-2.5 text-sm font-medium text-slate-900 transition hover:bg-slate-200"
                        >
                            <Play size={16} />
                            Start Analysis
                        </button>
                    </div>
                </div>

                {/* Analysis Types */}
                <div className="rounded-2xl border border-white/10 bg-white/2.5 p-4 sm:p-5">
                    <div className="mb-5 flex items-center justify-between">
                        <div>
                            <h3 className="font-medium">
                                Analysis Types
                            </h3>

                            <p className="mt-1 text-xs text-slate-500">
                                Your analysis distribution
                            </p>
                        </div>

                    </div>

                    {loading ? (
                        <div className="space-y-5">
                            <SkeletonBar />
                            <SkeletonBar />
                            <SkeletonBar />
                        </div>
                    ) : analysisTypes.length === 0 ? (
                        <div className="flex min-h-45 flex-col items-center justify-center gap-2 text-center text-sm text-slate-500">
                            <Target size={22} className="text-slate-600" />
                            No analyses yet.
                        </div>
                    ) : (
                        <>
                            <div className="mb-5 flex items-center justify-center">
                                <DonutChart
                                    items={analysisTypes}
                                    total={totalTypeCount}
                                />
                            </div>

                            <div className="space-y-3">
                                {analysisTypes.map((item, index) => {
                                    const percentage = totalTypeCount
                                        ? Math.round((item.count / totalTypeCount) * 100)
                                        : 0

                                    return (
                                        <div
                                            key={item.task}
                                            className="flex items-center justify-between gap-3 text-sm"
                                        >
                                            <div className="flex min-w-0 items-center gap-2">
                                                <span
                                                    className="h-2 w-2 shrink-0 rounded-full"
                                                    style={{
                                                        backgroundColor:
                                                            TYPE_COLORS[index % TYPE_COLORS.length]
                                                    }}
                                                />
                                                <span className="truncate text-slate-300">
                                                    {item.name}
                                                </span>
                                            </div>
                                            <span className="shrink-0 text-xs text-slate-500">
                                                {percentage}%
                                            </span>
                                        </div>
                                    )
                                })}
                            </div>
                        </>
                    )}
                </div>

            </div>

            {/* System status + Quick actions */}
            <div className="mb-5 grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div className="rounded-2xl border border-white/10 bg-white/2.5 p-4 sm:p-5">
                    <h3 className="mb-4 font-medium">AI System Status</h3>

                    <div className="space-y-3 text-sm">
                        <StatusRow label="Agent Status" ok={!error} />
                        <StatusRow label="Model Engine" ok={!error} okLabel="Operational" />
                        <StatusRow label="Data Pipeline" ok={!error} okLabel="Healthy" />
                        <div className="flex items-center justify-between text-slate-500">
                            <span>Last Updated</span>
                            <span className="text-xs">Just now</span>
                        </div>
                    </div>
                </div>

                <div className="rounded-2xl border border-white/10 bg-white/2.5 p-4 sm:p-5">
                    <h3 className="mb-4 font-medium">Quick Actions</h3>

                    <div className="grid grid-cols-2 gap-2.5 sm:gap-3">
                        <QuickAction
                            icon={Upload}
                            label="New Analysis"
                            onClick={() => navigate("/analysis/new")}
                        />
                        <QuickAction
                            icon={Clock3}
                            label="View History"
                            onClick={() => navigate("/history")}
                            disabled

                        />
                        <QuickAction icon={FolderOpen} label="Datasets" disabled />
                        <QuickAction icon={Grid2X2} label="Model Hub" disabled />
                    </div>
                </div>
            </div>

            {/* Recent + Activity */}
            <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">

                {/* Recent Analyses */}
                <div className="min-w-0 rounded-2xl border border-white/10 bg-white/2.5">
                    <div className="flex items-center justify-between border-b border-white/10 p-4 sm:p-5">
                        <div>
                            <h3 className="font-medium">
                                Recent Analyses
                            </h3>

                            <p className="mt-1 text-xs text-slate-500">
                                Latest satellite intelligence runs
                            </p>
                        </div>
                    </div>

                    {loading ? (
                        <div className="space-y-3 p-4 sm:p-5">
                            <TableSkeleton />
                            <TableSkeleton />
                            <TableSkeleton />
                        </div>
                    ) : recentAnalyses.length === 0 ? (
                        <div className="flex min-h-55 items-center justify-center p-6 text-center">
                            <div>
                                <Image
                                    size={30}
                                    className="mx-auto mb-3 text-slate-600"
                                />

                                <p className="text-sm text-slate-400">
                                    No analyses yet
                                </p>

                                <button
                                    onClick={() => navigate("/analysis/new")}
                                    className="mt-3 text-sm text-cyan-300 hover:text-cyan-200"
                                >
                                    Create your first analysis →
                                </button>
                            </div>
                        </div>
                    ) : (
                        <div className="overflow-x-auto">
                            <table className="w-full min-w-170">
                                <thead>
                                    <tr className="border-b border-white/5 text-left text-xs text-slate-500">
                                        <th className="px-4 py-3 font-medium sm:px-5">
                                            Analysis
                                        </th>

                                        <th className="px-4 py-3 font-medium">
                                            Sensor
                                        </th>

                                        <th className="px-4 py-3 font-medium">
                                            Model
                                        </th>

                                        <th className="px-4 py-3 font-medium">
                                            Confidence
                                        </th>

                                        <th className="px-4 py-3 font-medium">
                                            Status
                                        </th>
                                    </tr>
                                </thead>

                                <tbody>
                                    {recentAnalyses.map((item) => (
                                        <tr
                                            key={item.id}
                                            className="border-b border-white/5 last:border-0"
                                        >
                                            <td className="px-4 py-4 sm:px-5">
                                                <div className="flex items-center gap-3">
                                                    <div className="rounded-lg bg-blue-500/10 p-2 text-blue-300">
                                                        <Brain size={16} />
                                                    </div>

                                                    <div className="min-w-0">
                                                        <p className="truncate text-sm text-slate-200">
                                                            {item.task}
                                                        </p>

                                                        <p className="mt-0.5 text-xs text-slate-500">
                                                            {formatDate(item.created_at)}
                                                        </p>
                                                    </div>
                                                </div>
                                            </td>

                                            <td className="px-4 py-4 text-sm text-slate-400">
                                                {item.sensor || "Unknown"}
                                            </td>

                                            <td className="max-w-45 truncate px-4 py-4 text-sm text-slate-400">
                                                {item.model || "Unknown"}
                                            </td>

                                            <td className="px-4 py-4 text-sm text-slate-300">
                                                {(getDisplayConfidence(item)*100).toFixed(2)}%
                                            </td>

                                            <td className="px-4 py-4">
                                                <span className="rounded-full border border-emerald-400/20 bg-emerald-400/5 px-2.5 py-1 text-xs text-emerald-300">
                                                    {item.status}
                                                </span>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}
                </div>

                {/* Activity Feed */}
                <div className="rounded-2xl border border-white/10 bg-white/2.5">
                    <div className="border-b border-white/10 p-4 sm:p-5">
                        <h3 className="font-medium">
                            Activity Feed
                        </h3>

                        <p className="mt-1 text-xs text-slate-500">
                            Latest workspace activity
                        </p>
                    </div>

                    {loading ? (
                        <div className="space-y-4 p-4">
                            <ActivitySkeleton />
                            <ActivitySkeleton />
                            <ActivitySkeleton />
                        </div>
                    ) : activities.length === 0 ? (
                        <div className="flex min-h-55 items-center justify-center p-6 text-center text-sm text-slate-500">
                            No activity yet.
                        </div>
                    ) : (
                        <div className="divide-y divide-white/5">
                            {activities.map((item) => (
                                <div
                                    key={item.id}
                                    className="flex gap-3 p-4"
                                >
                                    <div className="mt-0.5 rounded-lg bg-cyan-400/10 p-2 text-cyan-300">
                                        <CheckCircle2 size={15} />
                                    </div>

                                    <div className="min-w-0">
                                        <p className="text-sm text-slate-300">
                                            {item.title}
                                        </p>

                                        <p className="mt-1 truncate text-xs text-slate-500">
                                            {item.description}
                                        </p>

                                        <p className="mt-2 text-[11px] text-slate-600">
                                            {formatDate(item.created_at)}
                                        </p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>

            </div>

        </main>
    )
}

const SkeletonBar = () => {
    return (
        <div>
            <div className="mb-2 flex justify-between">
                <div className="h-4 w-32 animate-pulse rounded bg-white/10" />
                <div className="h-4 w-6 animate-pulse rounded bg-white/10" />
            </div>

            <div className="h-1.5 animate-pulse rounded-full bg-white/10" />
        </div>
    )
}

const TableSkeleton = () => {
    return (
        <div className="flex items-center gap-3 rounded-xl border border-white/5 p-3">
            <div className="h-9 w-9 animate-pulse rounded-lg bg-white/10" />

            <div className="flex-1 space-y-2">
                <div className="h-3 w-40 animate-pulse rounded bg-white/10" />
                <div className="h-2.5 w-24 animate-pulse rounded bg-white/5" />
            </div>

            <div className="h-3 w-16 animate-pulse rounded bg-white/10" />
        </div>
    )
}

const ActivitySkeleton = () => {
    return (
        <div className="flex gap-3">
            <div className="h-8 w-8 shrink-0 animate-pulse rounded-lg bg-white/10" />

            <div className="flex-1 space-y-2">
                <div className="h-3 w-40 animate-pulse rounded bg-white/10" />
                <div className="h-2.5 w-52 max-w-full animate-pulse rounded bg-white/5" />
                <div className="h-2 w-20 animate-pulse rounded bg-white/5" />
            </div>
        </div>
    )
}

const DonutChart = ({ items, total }) => {
    const size = 140
    const stroke = 16
    const radius = (size - stroke) / 2
    const circumference = 2 * Math.PI * radius

    let offsetAccumulator = 0

    return (
        <div className="relative h-32 w-32 shrink-0 sm:h-35 sm:w-35">
            <svg width="100%" height="100%" viewBox={`0 0 ${size} ${size}`}>
                <circle
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    fill="none"
                    stroke="rgba(255,255,255,0.05)"
                    strokeWidth={stroke}
                />

                {items.map((item, index) => {
                    const fraction = total ? item.count / total : 0
                    const dash = fraction * circumference
                    const gap = circumference - dash

                    const circle = (
                        <circle
                            key={item.task}
                            cx={size / 2}
                            cy={size / 2}
                            r={radius}
                            fill="none"
                            stroke={TYPE_COLORS[index % TYPE_COLORS.length]}
                            strokeWidth={stroke}
                            strokeDasharray={`${dash} ${gap}`}
                            strokeDashoffset={-offsetAccumulator}
                            transform={`rotate(-90 ${size / 2} ${size / 2})`}
                        />
                    )

                    offsetAccumulator += dash

                    return circle
                })}
            </svg>

            <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span className="text-lg font-semibold sm:text-xl">{total}</span>
                <span className="text-[10px] text-slate-500">Total</span>
            </div>
        </div>
    )
}

const StatusRow = ({ label, ok, okLabel = "Active" }) => (
    <div className="flex items-center justify-between gap-3">
        <span className="text-slate-400">{label}</span>
        <span
            className={`flex shrink-0 items-center gap-1.5 text-xs ${
                ok ? "text-emerald-300" : "text-amber-300"
            }`}
        >
            <span
                className={`h-1.5 w-1.5 rounded-full ${
                    ok ? "bg-emerald-400" : "bg-amber-400"
                }`}
            />
            {ok ? okLabel : "Unavailable"}
        </span>
    </div>
)

const QuickAction = ({ icon: Icon, label, onClick, disabled }) => (
    <button
        onClick={disabled ? undefined : onClick}
        disabled={disabled}
        className={`flex flex-col items-center justify-center gap-2 rounded-xl border border-white/10 bg-black/20 py-3 text-[11px] transition sm:py-4 sm:text-xs ${
            disabled
                ? "cursor-not-allowed text-slate-600"
                : "cursor-pointer text-slate-300 hover:bg-white/5 hover:text-white"
        }`}
    >
        <Icon size={18} />
        <span className="text-center leading-tight">{label}</span>
    </button>
)

const PlusIcon = () => (
    <svg
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
    >
        <path d="M12 5v14" />
        <path d="M5 12h14" />
    </svg>
)

export default DashboardContent