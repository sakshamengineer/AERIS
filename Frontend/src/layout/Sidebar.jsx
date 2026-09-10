import React, { useState } from "react"
import { NavLink } from "react-router-dom"
import {
    BarChart3,
    CircleHelp,
    Clock3,
    FileText,
    FolderOpen,
    Grid2X2,
    LayoutDashboard,
    Menu,
    Settings,
    Sparkles,
    X,
    Zap
} from "lucide-react"
import logo1 from '../assets/logo1.png'

const Sidebar = () => {
    const [open, setOpen] = useState(false)

    const menuItems = [
        { path: "/", label: "Dashboard", icon: LayoutDashboard },
        { path: "/analysis/new", label: "New Analysis", icon: Sparkles },
        { path: "/history", label: "History", icon: Clock3, disabled: true },
        { path: "/reports", label: "Reports", icon: FileText, disabled: true },
        { path: "/datasets", label: "Datasets", icon: FolderOpen, disabled: true },
        { path: "/model-performance", label: "Model Performance", icon: BarChart3 }
    ];

    return (
        <>
            {/* Hamburger toggle - mobile/tablet only, hidden once the sidebar is pinned open at lg */}
            <button
                type="button"
                onClick={() => setOpen(true)}
                aria-label="Open menu"
                className={`fixed left-4 top-4 z-50 flex h-10 w-10 items-center justify-center rounded-lg border border-white/10 bg-[#030814] text-slate-300 shadow-lg lg:hidden ${
                    open ? "hidden" : "flex"
                }`}
            >
                <Menu className="h-5 w-5" />
            </button>

            {/* Backdrop - only rendered on mobile while the drawer is open */}
            {open && (
                <div
                    onClick={() => setOpen(false)}
                    className="fixed inset-0 z-40 bg-black/60 lg:hidden"
                />
            )}

            <aside
                className={`fixed inset-y-0 left-0 z-50 flex w-72 max-w-[80vw] flex-col border-b border-white/[0.07] bg-[#030814] transition-transform duration-300 ease-in-out lg:static lg:z-auto lg:h-screen lg:w-62.5 lg:max-w-none lg:translate-x-0 lg:border-b-0 lg:border-r lg:sticky lg:top-0 ${
                    open ? "translate-x-0" : "-translate-x-full"
                }`}
            >
                <div className="flex h-23 items-center justify-between gap-3 px-7 py-5">
                    <div className="flex items-center gap-3">
                        <div className="relative flex h-33 w-33 items-center justify-center">
                            <img src={logo1} alt="logo" />
                        </div>

                        <div>
                            <h1 className="text-[20px] font-semibold tracking-wide text-white">AERIS</h1>
                            <p className="text-[10px] text-slate-400">Adaptive Earth-observation Reasoning & Intelligence System</p>
                        </div>
                    </div>

                    {/* Close button - mobile/tablet only */}
                    <button
                        type="button"
                        onClick={() => setOpen(false)}
                        aria-label="Close menu"
                        className="flex h-9 w-9 items-center justify-center rounded-lg text-slate-400 hover:bg-white/5 hover:text-white lg:hidden"
                    >
                        <X className="h-5 w-5" />
                    </button>
                </div>

                <nav className="space-y-2 px-4">
                    {menuItems.map((item) => {
                        const Icon = item.icon
                        if (item.disabled) {
                            return (
                                <div
                                    key={item.path}
                                    className="flex h-11 w-full cursor-not-allowed items-center gap-4 rounded-lg px-4 text-sm text-slate-600"
                                >
                                    <Icon className="h-4.5 w-4.5" />
                                    <span>{item.label}</span>
                                </div>
                            )
                        }
                            return (
                                <NavLink
                                    key={item.path}
                                    to={item.path}
                                    end={item.path === "/"}
                                    onClick={() => setOpen(false)}
                                    className={({ isActive }) =>
                                        `group flex h-11 w-full items-center gap-4 rounded-lg px-4 text-sm transition-all ${isActive
                                            ? "border border-blue-500/40 bg-linear-to-r from-blue-500/15 to-purple-500/10 text-cyan-300"
                                            : "text-slate-400 hover:bg-white/3 hover:text-white"
                                        }`
                                    }
                                >
                                    <Icon className="h-4.5 w-4.5" />
                                    <span>{item.label}</span>
                                </NavLink>
                            );
                        })}
                </nav>

                <div className="mx-5 my-6 border-t border-white/6" />

                <nav className="space-y-2 px-4">
                    <button className="cursor-not-allowed group flex h-11 w-full items-center gap-4 rounded-lg px-4 text-sm text-slate-400 transition hover:bg-white/3 hover:text-white">
                        <Settings className="h-4.5 w-4.5" />
                        <span>Settings</span>
                    </button>

                    <button className="cursor-not-allowed group flex h-11 w-full items-center gap-4 rounded-lg px-4 text-sm text-slate-400 transition hover:bg-white/3 hover:text-white">
                        <CircleHelp className="h-4.5 w-4.5" />
                        <span>Help & Support</span>
                    </button>
                </nav>
            </aside>
        </>
    )
}

export default Sidebar;