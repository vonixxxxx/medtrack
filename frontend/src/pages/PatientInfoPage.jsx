import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

export default function PatientInfoPage() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = async () => {
    setLoading(true);
    try {
      const resp = await fetch('/api/screening/results', {
        headers: { Authorization: `Bearer ${localStorage.getItem('token')}` }
      });
      const json = await resp.json();
      if (!resp.ok) throw new Error(json.error || 'Failed to fetch');
      setData(json.results);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchData(); }, []);

  const downloadCSV = async () => {
    const resp = await fetch('/api/screening/export?format=csv', {
      headers: { Authorization: `Bearer ${localStorage.getItem('token')}` }
    });
    const text = await resp.text();
    const blob = new Blob([text], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'patient-results.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  if (loading) return <div className="p-6">Loading...</div>;
  if (error) return <div className="p-6 text-red-600">{error}</div>;

  const Card = ({ title, children }) => (
    <motion.div whileHover={{ y: -2 }} className="bg-white rounded-xl shadow p-6 border border-gray-100">
      <h3 className="text-lg font-semibold text-gray-800 mb-3">{title}</h3>
      {children}
    </motion.div>
  );

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-6xl mx-auto px-4">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Patient Info</h1>
            <p className="text-sm text-gray-600">View and manage your clinical assessments</p>
          </div>
          <button onClick={downloadCSV} className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700">Export CSV</button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {screeningResults.map((result) => (
            <div key={result.id} className="bg-white border border-gray-200 rounded-xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-800">{result.assessmentType}</h3>
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${result.severityBgColor}`}>
                  {result.severity || result.riskCategory || 'Completed'}
                </span>
              </div>
              
              <div className="space-y-2 mb-4">
                <p className="text-sm text-gray-600">
                  <span className="font-medium">Score:</span> {result.totalScore}
                </p>
                <p className="text-sm text-gray-600">
                  <span className="font-medium">Completed:</span> {new Date(result.completedAt).toLocaleDateString()}
                </p>
              </div>
              
              <button
                onClick={() => handleRetake(result.assessmentType)}
                className="w-full px-4 py-2 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-md hover:from-blue-700 hover:to-indigo-700"
              >
                Retake Assessment
              </button>
            </div>
          ))}
        </div>
        
        <div className="flex justify-center mt-8">
          <button
            onClick={exportAllResults}
            className="px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-md hover:from-blue-700 hover:to-indigo-700 flex items-center gap-2"
          >
            <Download className="w-5 h-5" />
            Export All Results
          </button>
        </div>
      </div>
    </div>
  );
}


