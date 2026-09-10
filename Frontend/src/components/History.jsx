import Sidebar from "../layout/Sidebar"

const History = () => {
    return (
        <div className="min-h-screen bg-[#020611] text-white lg:grid lg:grid-cols-[250px_1fr]">
            <Sidebar />

            <main className="flex min-w-0 items-center justify-center p-6 pt-16">
                <div className="max-w-sm rounded-2xl border border-white/10 bg-white/2.5 p-8 text-center">
                    <h1 className="text-lg font-semibold text-white">History</h1>
                    <p className="mt-2 text-sm text-slate-400">
                        Past analyses will show up here. This page isn't built yet.
                    </p>
                </div>
            </main>
        </div>
    )
}

export default History
