import { Routes, Route } from "react-router-dom"
import Dashboard from "../components/Dashboard"
import NewAnalysis from "../components/NewAnalysis"
import ModelPerformance from "../components/ModelPerformance"
import History from "../components/History"


const AppRoutes = () => {
    return (
        <Routes>
            <Route
                path="/"
                element={<Dashboard />}
            />

            <Route
                path="/analysis/new"
                element={<NewAnalysis />}
            />

            <Route
                path="/model-performance"
                element={<ModelPerformance />}
            />

            <Route
                path="/history"
                element={<History />}
            />
        </Routes>
    )
}

export default AppRoutes
