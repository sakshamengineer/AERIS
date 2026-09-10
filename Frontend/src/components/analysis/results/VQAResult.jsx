import {
    Brain,
    CheckCircle2,
    Image as ImageIcon,
    Activity
} from 'lucide-react'
import { getDisplayConfidence } from '../../../utils/Confidence'

const VQAResult = (props) => {
    const data = props.result || {}
    const confidence = getDisplayConfidence(data)

    const image =
        data.images?.[0]

    const trace =
        data.trace ||
        data.execution_trace ||
        []

    const evidence =
        data.evidence || []

    return (
        <main className='min-w-0 px-4 py-5 sm:px-6 lg:px-8'>
            <div className='mx-auto max-w-6xl'>

                <div className='mb-6'>
                    <p className='text-sm text-cyan-400'>
                        ANALYSIS RESULT
                    </p>

                    <h1 className='mt-1 text-2xl font-semibold sm:text-3xl'>
                        Visual Question Answering
                    </h1>
                </div>

                <div className='grid gap-5 lg:grid-cols-[1.4fr_0.6fr]'>

                    <section className='space-y-5'>

                        {image && (
                            <div className='overflow-hidden rounded-2xl border border-white/10 bg-black/20'>
                                <img
                                    src={image}
                                    alt='Satellite analysis'
                                    className='max-h-[550px] w-full object-contain'
                                />
                            </div>
                        )}

                        <div className='rounded-2xl border border-white/10 bg-white/[0.03] p-5 sm:p-6'>
                            <div className='flex items-center gap-2'>
                                <Brain className='h-5 w-5 text-cyan-400' />

                                <h2 className='font-medium'>
                                    AI Answer
                                </h2>
                            </div>

                            <p className='mt-4 text-base leading-7 text-slate-200'>
                                {data.answer ||
                                    "No answer returned."}
                            </p>
                        </div>

                    </section>

                    <section className='space-y-5'>

                        <div className='grid grid-cols-2 gap-3'>
                            <div className='rounded-2xl border border-white/10 bg-white/[0.03] p-4'>
                                <p className='text-xs text-slate-500'>
                                    Confidence
                                </p>

                                <p className='mt-2 text-xl font-semibold text-cyan-300'>
                                    {Math.round(confidence * 100)}%
                                </p>
                            </div>

                            <div className='rounded-2xl border border-white/10 bg-white/[0.03] p-4'>
                                <p className='text-xs text-slate-500'>
                                    Model
                                </p>

                                <p className='mt-2 text-sm font-medium text-white'>
                                    {data.model ||
                                        "Unknown"}
                                </p>
                            </div>
                        </div>

                        <div className='rounded-2xl border border-white/10 bg-white/[0.03] p-5'>
                            <div className='flex items-center gap-2'>
                                <ImageIcon className='h-5 w-5 text-purple-400' />

                                <h2 className='font-medium'>
                                    Visual Evidence
                                </h2>
                            </div>

                            <div className='mt-4 space-y-3'>
                                {evidence.length ? (
                                    evidence.map(
                                        (item, index) => (
                                            <div
                                                key={index}
                                                className='rounded-xl border border-white/5 bg-black/20 p-3'
                                            >
                                                {item.url && (
                                                    <img
                                                        src={item.url}
                                                        alt={
                                                            item.description ||
                                                            "Evidence"
                                                        }
                                                        className='mb-3 max-h-52 w-full rounded-lg object-contain'
                                                    />
                                                )}

                                                <p className='text-xs text-slate-400'>
                                                    {item.description ||
                                                        item.type ||
                                                        "Evidence"}
                                                </p>
                                            </div>
                                        )
                                    )
                                ) : (
                                    <p className='text-sm text-slate-600'>
                                        No visual evidence returned.
                                    </p>
                                )}
                            </div>
                        </div>

                        <div className='rounded-2xl border border-white/10 bg-white/[0.03] p-5'>
                            <div className='flex items-center gap-2'>
                                <Activity className='h-5 w-5 text-emerald-400' />

                                <h2 className='font-medium'>
                                    Execution Trace
                                </h2>
                            </div>

                            <div className='mt-4 space-y-2'>
                                {trace.map(
                                    (item, index) => (
                                        <div
                                            key={index}
                                            className='flex gap-3 rounded-lg bg-black/20 p-3'
                                        >
                                            <CheckCircle2 className='h-4 w-4 shrink-0 text-emerald-400' />

                                            <div className='min-w-0'>
                                                <p className='text-xs text-slate-300'>
                                                    {item.step ||
                                                        item.name ||
                                                        `Step ${index + 1}`}
                                                </p>

                                                {item.message && (
                                                    <p className='mt-1 text-[11px] text-slate-600'>
                                                        {item.message}
                                                    </p>
                                                )}
                                            </div>
                                        </div>
                                    )
                                )}
                            </div>
                        </div>

                    </section>
                </div>
            </div>
        </main>
    )
}

export default VQAResult