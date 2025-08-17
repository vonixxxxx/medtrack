export default function DashboardCard({ title, children, className = "" }) {
  return (
    <div className={`bg-white rounded-3xl shadow-lg p-6 transition hover:shadow-lg ${className}`}>
      <h3 className="text-lg font-semibold mb-4 text-gray-800 animated-gradient-text">{title}</h3>
      {children}
    </div>
  );
}
