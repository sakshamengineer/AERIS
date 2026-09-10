import { Share2, Download, Plus } from 'lucide-react'
import { toast } from 'sonner'
import VQAResult from './results/VQAResult'
import CaptioningResult from './results/CaptioningResult'
import ChangeResult from './results/ChangeResult'
import ChangeVQAResult from './results/ChangeVQAResult'
import OpticalSARResult from './results/OpticalSARResult'

const TASK_TITLES = {
    vqa: "Visual Question Answering",
    captioning: "Image Captioning",
    change_detection: "Change Detection",
    change_vqa: "Change VQA",
    optical_sar: "Optical-SAR Fusion"
}

const ResultState = (props) => {
    const data = props.result?.result || props.result || {}

    const downloadReport = () => {
        const report = {
            analysis_id: data.analysis_id,
            task: data.task,
            model: data.model,
            confidence: data.confidence,
            answer: data.answer,
            created_at: data.created_at
        }

        const blob = new Blob(
            [JSON.stringify(report, null, 2)],
            { type: "application/json" }
        )

        const url = URL.createObjectURL(blob)
        const link = document.createElement("a")

        link.href = url
        link.download = `${data.analysis_id || "analysis"}-report.json`
        link.click()

        URL.revokeObjectURL(url)
    }

    const shareResult = async () => {
        const shareText = `SatQuery AI — ${TASK_TITLES[data.task] || "Analysis"}: ${
            data.answer || ""
        }`.slice(0, 500)

        try {
            if (navigator.share) {
                await navigator.share({ title: "SatQuery AI Result", text: shareText })
            } else {
                await navigator.clipboard.writeText(shareText)
                toast.success("Result copied to clipboard.")
            }
        } catch (error) {
            // user cancelled share sheet — no-op
        }
    }

    const renderResult = () => {
        if (data.task === "vqa") return <VQAResult result={data} />
        if (data.task === "captioning") return <CaptioningResult result={data} />
        if (data.task === "change_detection") return <ChangeResult result={data} />
        if (data.task === "change_vqa") return <ChangeVQAResult result={data} />
        if (data.task === "optical_sar") return <OpticalSARResult result={data} />

        return (
            <div className='mx-auto max-w-4xl rounded-2xl border border-white/10 bg-white/[0.03] p-6 text-center'>
                <h1 className='text-xl font-semibold'>Analysis Result</h1>
                <p className='mt-2 text-sm text-slate-500'>
                    {data.error || "No supported result type was returned."}
                </p>
            </div>
        )
    }

    return (
        <div>
            <div className='flex flex-col gap-4 px-4 pt-16 sm:flex-row sm:items-center sm:justify-between sm:px-6 lg:px-8 lg:pt-5 xl:px-10'>
                <div>
                    <div className='flex flex-wrap items-center gap-3'>
                        <h1 className='flex items-center gap-2 text-2xl font-semibold sm:text-3xl'>
                            Analysis Result
                            <span className='text-cyan-300'>✨</span>
                        </h1>
                        <span
                            className={`rounded-full border px-2.5 py-1 text-xs ${
                                data.success === false
                                    ? "border-red-400/30 bg-red-400/10 text-red-300"
                                    : "border-emerald-400/30 bg-emerald-400/10 text-emerald-300"
                            }`}
                        >
                            {data.status || (data.success === false ? "Failed" : "Completed")}
                        </span>
                    </div>

                    <p className='mt-1 text-xs text-slate-500'>
                        {data.created_at
                            ? `Completed on ${new Date(data.created_at).toLocaleString("en-IN")}`
                            : ""}
                        {data.analysis_id ? ` • ID: ${data.analysis_id}` : ""}
                    </p>
                </div>

                <div className='flex flex-wrap gap-2'>
                    <button
                        onClick={shareResult}
                        className='flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-300 hover:bg-white/10'
                    >
                        <Share2 size={15} />
                        Share Result
                    </button>

                    <button
                        onClick={downloadReport}
                        className='flex items-center gap-2 rounded-xl bg-linear-to-r from-cyan-500 to-blue-600 px-3 py-2 text-sm font-medium text-white'
                    >
                        <Download size={15} />
                        Download Report
                    </button>

                    <button
                        onClick={props.onNewAnalysis}
                        className='flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-300 hover:bg-white/10'
                    >
                        <Plus size={15} />
                        New Analysis
                    </button>
                </div>
            </div>

            {renderResult()}
        </div>
    )
}

export default ResultState
