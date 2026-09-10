import Sidebar from "../layout/Sidebar"
import DashboardContent from "./dashboard/DashboardContent"

const Dashboard = () => {
  return (
    <div className="min-h-screen bg-[#020611] text-white lg:grid lg:grid-cols-[250px_1fr]">
      <Sidebar />
      <DashboardContent />
    </div>
  );
};

export default Dashboard;