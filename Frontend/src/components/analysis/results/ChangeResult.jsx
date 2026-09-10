import { useMemo, useState } from 'react'
import {
    Activity,
    CheckCircle2,
    Columns2
} from 'lucide-react'
import { getDisplayConfidence } from '../../../utils/Confidence'

const TABS = ["Change Map", "Probability", "Before", "After", "Side by Side"]

const ChangeResult = (props) => {
    const data = props.result || {}
    const confidence = getDisplayConfidence(data)

    const images = data.images || []
    const evidence = data.evidence || []
    const trace = data.trace || data.execution_trace || []

    const changeEvidence = evidence.find(
        (item) =>
            String(item.type || "").toLowerCase().includes("change") ||
            String(item.description || "").toLowerCase().includes("change")
    )

    const changeMap = changeEvidence?.url
    const probabilityEvidence = evidence.find(
        (item) => String(item.type || "").toLowerCase().includes("probability")
    )
    const probabilityMap = probabilityEvidence?.url
    const before = images[0]
    const after = images[1]

    const [activeTab, setActiveTab] = useState(changeMap ? "Change Map" : "Before")

    const confidencePercent = Math.round((data.confidence || 0) * 100)

    const activeImage = useMemo(() => {
        if (activeTab === "Change Map") return changeMap || before
        if (activeTab === "Probability") return probabilityMap || changeMap || before
        if (activeTab === "Before") return before
        if (activeTab === "After") return after
        return null
    }, [activeTab, before, after, changeMap])

    return (
        <main className='min-w-0 px-4 py-5 sm:px-6 lg:px-8'>
            <div className='mx-auto max-w-7xl'>

                <div className='mb-6'>
                    <p className='text-sm text-cyan-400'>BI-TEMPORAL ANALYSIS</p>
                    <h1 className='mt-1 text-2xl font-semibold sm:text-3xl'>Change Detection</h1>
                </div>

                <div className='grid gap-5 xl:grid-cols-[1.6fr_0.9fr]'>

                    <section className='space-y-5'>

                        {/* Tabs + viewer */}
                        <div className='overflow-hidden rounded-2xl border border-white/10 bg-black/20'>
                            <div className='flex flex-wrap items-center gap-1 border-b border-white/5 px-3 py-2'>
                                {TABS.map((tab) => (
                                    <button
                                        key={tab}
                                        onClick={() => setActiveTab(tab)}
                                        disabled={tab === "Change Map" && !changeMap}
                                        className={`rounded-lg px-3 py-1.5 text-xs transition disabled:cursor-not-allowed disabled:opacity-30 ${
                                            activeTab === tab
                                                ? "bg-white/10 text-white"
                                                : "text-slate-500 hover:text-slate-300"
                                        }`}
                                    >
                                        {tab}
                                    </button>
                                ))}
                            </div>

                            {activeTab === "Side by Side" ? (
                                <div className='grid grid-cols-2 gap-px bg-white/5'>
                                    <div className='bg-black/30'>
                                        <p className='px-3 py-2 text-xs text-slate-500'>Before</p>
                                        {before && (
                                            <img src={before} alt='Before' className='h-72 w-full object-contain sm:h-96' />
                                        )}
                                    </div>
                                    <div className='bg-black/30'>
                                        <p className='px-3 py-2 text-xs text-slate-500'>After</p>
                                        {after && (
                                            <img src={after} alt='After' className='h-72 w-full object-contain sm:h-96' />
                                        )}
                                    </div>
                                </div>
                            ) : activeImage ? (
                                <img
                                    src={activeImage}
                                    alt={activeTab}
                                    className='max-h-[560px] w-full object-contain'
                                />
                            ) : (
                                <div className='flex h-72 items-center justify-center text-sm text-slate-600'>
                                    No image available for this view.
                                </div>
                            )}
                        </div>

                        <div className='rounded-2xl border border-white/10 bg-white/[0.03] p-5 sm:p-6'>
                            <h2 className='font-medium'>AI Answer</h2>
                            <p className='mt-4 text-sm leading-7 text-slate-300'>
                                {data.answer || "No change analysis returned."}
                            </p>
                        </div>

                        {evidence.length > 0 && (
                            <div className='rounded-2xl border border-white/10 bg-white/[0.03] p-5'>
                                <h2 className='mb-4 font-medium'>Visual Evidence</h2>
                                <div className='grid grid-cols-2 gap-3 sm:grid-cols-4'>
                                    {evidence.map((item, index) => (
                                        <button
                                            key={index}
                                            onClick={() => {
                                                if (item === changeEvidence) setActiveTab("Change Map")
                                                else if (item === probabilityEvidence) setActiveTab("Probability")
                                                else if (index === 0) setActiveTab("Before")
                                                else setActiveTab("After")
                                            }}
                                            className='overflow-hidden rounded-xl border border-white/10 bg-black/20 text-left'
                                        >
                                            {item.url && (
                                                <img src={item.url} alt={item.description} className='h-20 w-full object-cover' />
                                            )}
                                            <p className='truncate px-2 py-1.5 text-[10px] text-slate-500'>
                                                {item.description || item.type}
                                            </p>
                                        </button>
                                    ))}
                                </div>
                            </div>
                        )}
                    </section>

                    <section className='space-y-5'>

                        <div className='rounded-2xl border border-white/10 bg-white/[0.03] p-5'>
                            <p className='text-xs text-slate-500'>Prediction Confidence</p>
                            <p className='mt-1 text-3xl font-semibold text-cyan-300'>
                                {confidencePercent}%
                            </p>
                            <div className='mt-3 h-2 overflow-hidden rounded-full bg-white/5'>
                                <div
                                    className='h-full rounded-full bg-linear-to-r from-emerald-400 to-cyan-400'
                                    style={{ width: `${confidencePercent}%` }}
                                />
                            </div>
                            <p className='mt-3 text-xs leading-5 text-slate-500'>
                                {data.confidence_type === "unavailable"
                                    ? "Confidence score not available for this run."
                                    : "Average confidence of the pixel-level prediction."}
                            </p>
                        </div>

                        <div className='grid grid-cols-2 gap-3'>
                            <div className='rounded-2xl border border-white/10 bg-white/[0.03] p-4'>
                                <p className='text-xs text-slate-500'>Model</p>
                                <p className='mt-2 text-xs leading-5 text-white'>{data.model || "Unknown"}</p>
                            </div>
                            <div className='rounded-2xl border border-white/10 bg-white/[0.03] p-4'>
                                <p className='text-xs text-slate-500'>Sensor</p>
                                <p className='mt-2 text-xs leading-5 text-white'>{data.sensor || "Unknown"}</p>
                            </div>
                        </div>

                        <div className='grid grid-cols-2 gap-3'>
                            <div className='rounded-2xl border border-white/10 bg-white/[0.03] p-4'>
                                <p className='text-xs text-slate-500'>Detected Change</p>
                                <p className='mt-2 text-xl font-semibold text-cyan-300'>
                                    {data.change_percentage ?? 0}%
                                </p>
                            </div>
                            <div className='rounded-2xl border border-white/10 bg-white/[0.03] p-4'>
                                <p className='text-xs text-slate-500'>Max Change Probability</p>
                                <p className='mt-2 text-xl font-semibold text-purple-300'>
                                    {data.max_change_probability != null
                                        ? `${Math.round(data.max_change_probability * 100)}%`
                                        : "—"}
                                </p>
                            </div>
                        </div>

                        <div className='rounded-2xl border border-white/10 bg-white/[0.03] p-5'>
                            <div className='flex items-center gap-2'>
                                <Columns2 className='h-5 w-5 text-purple-400' />
                                <p className='text-xs text-slate-500'>Routing Confidence</p>
                            </div>
                            <p className='mt-2 text-xl font-semibold text-purple-300'>
                                {Math.round(confidence * 100)}%
                            </p>
                            {data.routing_reason && (
                                <p className='mt-3 text-xs leading-5 text-slate-500'>{data.routing_reason}</p>
                            )}
                        </div>

                        <div className='rounded-2xl border border-white/10 bg-white/[0.03] p-5'>
                            <div className='flex items-center gap-2'>
                                <Activity className='h-5 w-5 text-emerald-400' />
                                <h2 className='font-medium'>Execution Trace</h2>
                            </div>
                            <div className='mt-4 space-y-2'>
                                {trace.map((item, index) => (
                                    <div key={index} className='flex gap-3 rounded-lg bg-black/20 p-3'>
                                        <CheckCircle2 className='h-4 w-4 shrink-0 text-emerald-400' />
                                        <p className='text-xs text-slate-400'>
                                            {item.step || item.name || `Step ${index + 1}`}
                                        </p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </section>
                </div>
            </div>
        </main>
    )
}

export default ChangeResult
