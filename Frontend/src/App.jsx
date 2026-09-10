import AppRoutes from "./routes/AppRoutes"
import { Toaster } from "sonner"

const App = () => {
    return (
        <div>
            <AppRoutes />
            <Toaster theme="dark" />
        </div>
    )
}

export default App