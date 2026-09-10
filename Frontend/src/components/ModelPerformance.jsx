import Sidebar from "../layout/Sidebar"
import ModelMetricsContent from "./models/ModelMetricsContent"

const ModelPerformance = () => {
    return (
        <div className="min-h-screen bg-[#020611] text-white lg:grid lg:grid-cols-[250px_1fr]">
            <Sidebar />
            <ModelMetricsContent />
        </div>
    )
}

export default ModelPerformance
