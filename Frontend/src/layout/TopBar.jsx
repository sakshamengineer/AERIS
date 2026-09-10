import { Bell } from "lucide-react"

const STEPS = [
    { id: 1, label: "Input" },
    { id: 2, label: "Analysis" },
    { id: 3, label: "Results" }
]

const TopBar = ({ crumbs = [], activeStep = null }) => {
    return (
        <div className="flex flex-col gap-4 border-b border-white/5 px-4 py-4 sm:px-6 sm:py-5 lg:flex-row lg:items-center lg:justify-between lg:px-8">
            <div className="flex items-center gap-2 text-sm text-slate-500">
                {crumbs.map((crumb, index) => (
                    <span key={crumb} className="flex items-center gap-2">
                        {index > 0 && <span>/</span>}
                        <span
                            className={
                                index === crumbs.length - 1
                                    ? "text-slate-200"
                                    : "text-slate-500"
                            }
                        >
                            {crumb}
                        </span>
                    </span>
                ))}
            </div>

            <div className="flex items-center gap-6">
                {activeStep && (
                    <div className="flex items-center">
                        {STEPS.map((step, index) => (
                            <div key={step.id} className="flex items-center">
                                <div className="flex flex-col items-center gap-1.5">
                                    <div
                                        className={`flex h-8 w-8 items-center justify-center rounded-full border text-xs font-medium transition ${
                                            step.id === activeStep
                                                ? "border-cyan-400 bg-cyan-400/10 text-cyan-300"
                                                : step.id < activeStep
                                                ? "border-cyan-400/40 bg-cyan-400/5 text-cyan-400"
                                                : "border-white/10 text-slate-600"
                                        }`}
                                    >
                                        {step.id}
                                    </div>
                                    <span
                                        className={`text-[11px] ${
                                            step.id === activeStep
                                                ? "text-slate-300"
                                                : "text-slate-600"
                                        }`}
                                    >
                                        {step.label}
                                    </span>
                                </div>

                                {index < STEPS.length - 1 && (
                                    <div
                                        className={`mx-2 mb-4 h-px w-10 sm:w-16 ${
                                            step.id < activeStep
                                                ? "bg-cyan-400/40"
                                                : "bg-white/10"
                                        }`}
                                    />
                                )}
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    )
}

export default TopBar
