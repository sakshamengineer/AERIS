import { useState } from 'react'
import Sidebar from '../layout/Sidebar'
import TopBar from '../layout/TopBar'
import InputState from './analysis/InputState'
import ProcessingState from './analysis/ProcessingState'
import ResultState from './analysis/ResultState'

const CRUMBS = {
    input: ["Dashboard", "New Analysis"],
    processing: ["Dashboard", "New Analysis", "Analysis in Progress"],
    result: ["Dashboard", "History", "Analysis Result"]
}

const STEP_BY_STATUS = {
    input: 1,
    processing: 2,
    result: 3
}

const NewAnalysis = () => {
    const [status, setStatus] = useState("input")
    const [result, setResult] = useState(null)
    const [analysisId, setAnalysisId] = useState(null)
    const [inputData, setInputData] = useState(null)

    const reset = () => {
        setStatus("input")
        setResult(null)
        setAnalysisId(null)
        setInputData(null)
    }

    return (
        <div className='min-h-screen bg-[#020611] text-white lg:grid lg:grid-cols-[240px_1fr]'>
            <Sidebar />

            <main className="min-w-0">
                <TopBar
                    crumbs={CRUMBS[status]}
                    activeStep={STEP_BY_STATUS[status]}
                />

                {status === "input" && (
                    <InputState
                        setStatus={setStatus}
                        setResult={setResult}
                        setAnalysisId={setAnalysisId}
                        setInputData={setInputData}
                    />
                )}

                {status === "processing" && (
                    <ProcessingState
                        analysisId={analysisId}
                        inputData={inputData}
                        setStatus={setStatus}
                        setResult={setResult}
                    />
                )}

                {status === "result" && (
                    <ResultState
                        result={result}
                        onNewAnalysis={reset}
                    />
                )}
            </main>
        </div>
    )
}

export default NewAnalysis
