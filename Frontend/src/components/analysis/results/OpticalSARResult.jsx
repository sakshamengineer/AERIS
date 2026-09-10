import { useMemo, useState } from 'react'
import { Activity, Layers, Satellite, Radio, Map } from 'lucide-react'
import { getDisplayConfidence } from '../../../utils/Confidence'

const OpticalSARResult = ({ result }) => {
    const data = result || {}
    const confidence = getDisplayConfidence(data)
    const images = data.images || []
    const evidence = data.evidence || []
    const trace = data.trace || data.execution_trace || []
    const [view, setView] = useState('change_map')

    const evidenceByType = useMemo(() => {
        const map = {}
        for (const item of evidence) {
            if (item?.type && item?.url) map[item.type] = item
        }
        return map
    }, [evidence])

    const views = [
        ['change_map', 'Change Map'],
        ['probability_map', 'Probability'],
        ['optical_t1', 'Optical T1'],
        ['optical_t2', 'Optical T2'],
        ['sar_t1', 'SAR T1'],
        ['sar_t2', 'SAR T2'],
        ['sar_fusion_map', 'SAR Fusion']
    ]

    const imageFor = (key) => {
        if (evidenceByType[key]?.url) return evidenceByType[key].url
        const indexMap = { optical_t1: 0, optical_t2: 1, sar_t1: 2, sar_t2: 3 }
        return images[indexMap[key]] || ''
    }

    const currentImage = imageFor(view)
    const changed = data.change_percentage
    const meanProbability = data.mean_change_probability
    const maxProbability = data.max_change_probability

    return (
        <main className='min-w-0 px-4 py-5 sm:px-6 lg:px-8'>
            <div className='mx-auto max-w-7xl'>
                <div className='mb-6'>
                    <p className='text-sm text-cyan-400'>MULTI-MODAL BI-TEMPORAL ANALYSIS</p>
                    <h1 className='mt-1 text-2xl font-semibold sm:text-3xl'>Optical + SAR Change Detection</h1>
                    <p className='mt-2 text-sm text-slate-500'>OpticalSARChangeNet using Optical T1/T2 and SAR T1/T2.</p>
                </div>

                <div className='grid gap-5 xl:grid-cols-[1.45fr_0.55fr]'>
                    <section className='space-y-5'>
                        <div className='overflow-hidden rounded-2xl border border-white/10 bg-black/20'>
                            <div className='flex flex-wrap gap-1 border-b border-white/5 p-2'>
                                {views.map(([key, label]) => (
                                    <button
                                        key={key}
                                        type='button'
                                        onClick={() => setView(key)}
                                        className={`rounded-lg px-3 py-2 text-xs transition ${view === key ? 'bg-white/10 text-white' : 'text-slate-500 hover:text-slate-300'}`}
                                    >
                                        {label}
                                    </button>
                                ))}
                            </div>
                            <div className='flex min-h-[420px] items-center justify-center bg-black/30 p-3 sm:p-5'>
                                {currentImage ? (
                                    <img src={currentImage} alt={view} className='max-h-[620px] w-full rounded-xl object-contain' />
                                ) : (
                                    <p className='text-sm text-slate-500'>No visual evidence returned.</p>
                                )}
                            </div>
                        </div>

                        <div className='grid gap-3 sm:grid-cols-2'>
                            {[
                                ['Optical T1', imageFor('optical_t1'), Satellite],
                                ['Optical T2', imageFor('optical_t2'), Satellite],
                                ['SAR T1', imageFor('sar_t1'), Radio],
                                ['SAR T2', imageFor('sar_t2'), Radio]
                            ].map(([label, src, Icon]) => (
                                <div key={label} className='overflow-hidden rounded-2xl border border-white/10 bg-black/20'>
                                    <div className='flex items-center gap-2 border-b border-white/5 px-4 py-3 text-xs text-slate-400'>
                                        <Icon className='h-4 w-4 text-cyan-300' /> {label}
                                    </div>
                                    {src && <img src={src} alt={label} className='h-52 w-full object-contain' />}
                                </div>
                            ))}
                        </div>

                        <div className='rounded-2xl border border-white/10 bg-white/[0.03] p-5 sm:p-6'>
                            <h2 className='font-medium'>AI Answer</h2>
                            <p className='mt-4 text-sm leading-7 text-slate-300'>{data.answer || 'No Optical-SAR result returned.'}</p>
                        </div>
                    </section>

                    <section className='space-y-5'>
                        <div className='rounded-2xl border border-white/10 bg-white/[0.03] p-5'>
                            <p className='text-xs text-slate-500'>Prediction Confidence</p>
                            <p className='mt-2 text-2xl font-semibold text-cyan-300'>{Math.round(confidence * 100)}%</p>
                            <p className='mt-2 text-xs text-slate-500'>Diagnostic confidence of the pixel-level model prediction.</p>
                        </div>

                        <div className='grid gap-3 sm:grid-cols-2 xl:grid-cols-1'>
                            <Metric icon={Map} label='Detected Change' value={changed != null ? `${changed}%` : '—'} />
                            <Metric icon={Activity} label='Mean Change Probability' value={meanProbability != null ? `${meanProbability}%` : '—'} />
                            <Metric icon={Activity} label='Max Change Probability' value={maxProbability != null ? `${maxProbability}%` : '—'} />
                        </div>

                        <div className='rounded-2xl border border-white/10 bg-white/[0.03] p-5'>
                            <p className='text-xs text-slate-500'>Model</p>
                            <p className='mt-2 text-sm text-white'>{data.model || 'OpticalSARChangeNet'}</p>
                            <p className='mt-5 text-xs text-slate-500'>Sensor</p>
                            <p className='mt-2 text-sm text-white'>{data.sensor || 'optical + SAR (bi-temporal)'}</p>
                            {data.threshold != null && <><p className='mt-5 text-xs text-slate-500'>Decision Threshold</p><p className='mt-2 text-sm text-white'>{Number(data.threshold).toFixed(2)}</p></>}
                        </div>

                        <div className='rounded-2xl border border-white/10 bg-white/[0.03] p-5'>
                            <div className='flex items-center gap-2'><Activity className='h-5 w-5 text-emerald-400' /><h2 className='font-medium'>Execution Trace</h2></div>
                            <div className='mt-4 space-y-2'>
                                {trace.map((item, index) => <div key={index} className='rounded-lg bg-black/20 p-3 text-xs text-slate-400'>{item.step || item.name || `Step ${index + 1}`}</div>)}
                            </div>
                        </div>
                    </section>
                </div>
            </div>
        </main>
    )
}

const Metric = ({ icon: Icon, label, value }) => (
    <div className='rounded-2xl border border-white/10 bg-white/[0.03] p-5'>
        <div className='flex items-center gap-2 text-xs text-slate-500'><Icon className='h-4 w-4 text-cyan-300' />{label}</div>
        <p className='mt-2 text-xl font-semibold text-cyan-300'>{value}</p>
    </div>
)

export default OpticalSARResult
