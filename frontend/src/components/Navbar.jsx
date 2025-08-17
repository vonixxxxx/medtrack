import { Link } from 'react-router-dom';

export default function Navbar() {
  return (
    <nav className="bg-white shadow mb-4">
      <div className="container mx-auto px-4 py-3 flex justify-between">
        <h1 className="font-bold text-xl">MedTrack</h1>
        <div className="space-x-4">
          <Link to="/" className="hover:underline">
            Dashboard
          </Link>
          <Link to="/add-medication" className="hover:underline">
            Add Medication
          </Link>
          <Link to="/add-metric" className="hover:underline">
            Add Metric
          </Link>
          <button
            onClick={() => {
              localStorage.removeItem('token');
              window.location.href = '/login';
            }}
            className="text-red-500 hover:underline"
          >
            Logout
          </button>
        </div>
      </div>
    </nav>
  );
}
