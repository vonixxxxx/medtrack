/**
 * Adherence Dashboard Component
 * Simple UI for testing adherence engine
 */

import React, { useState, useEffect } from 'react';

const AdherenceDashboard = () => {
  const [adherenceData, setAdherenceData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchAdherence = async () => {
    setLoading(true);
    setError(null);
    try {
      const token = localStorage.getItem('token');
      const response = await fetch('http://localhost:4000/api/adherence', {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error('Failed to fetch adherence data');
      }

      const data = await response.json();
      setAdherenceData(data.data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAdherence();
  }, []);

  if (loading) return <div>Loading adherence data...</div>;
  if (error) return <div>Error: {error}</div>;
  if (!adherenceData) return <div>No adherence data available</div>;

  return (
    <div style={{ padding: '20px' }}>
      <h2>Medication Adherence Dashboard</h2>
      <button onClick={fetchAdherence} style={{ marginBottom: '20px' }}>
        Refresh
      </button>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>
        {adherenceData.map((medication) => (
          <div key={medication.medicationId} style={{ border: '1px solid #ccc', padding: '15px', borderRadius: '8px' }}>
            <h3>Medication {medication.medicationId}</h3>
            <div style={{ marginTop: '10px' }}>
              <p><strong>Adherence:</strong> {medication.adherencePercentage.toFixed(1)}%</p>
              <p><strong>Expected Doses/Day:</strong> {medication.expectedDosesPerDay}</p>
              <p><strong>Streak:</strong> {medication.streakCount} days</p>
              
              {medication.weeklyAdherenceHistory.length > 0 && (
                <div style={{ marginTop: '10px' }}>
                  <strong>Weekly History:</strong>
                  <ul>
                    {medication.weeklyAdherenceHistory.slice(0, 4).map((week, idx) => (
                      <li key={idx}>
                        Week: {week.adherencePercentage.toFixed(1)}% ({week.dosesTaken}/{week.dosesExpected})
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default AdherenceDashboard;







