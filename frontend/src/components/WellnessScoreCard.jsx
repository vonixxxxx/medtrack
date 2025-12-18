/**
 * Wellness Score Card Component
 * Simple UI for testing wellness engine
 */

import React, { useState, useEffect } from 'react';

const WellnessScoreCard = () => {
  const [wellnessData, setWellnessData] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchWellness = async () => {
    setLoading(true);
    setError(null);
    try {
      const token = localStorage.getItem('token');
      
      const [scoreResponse, breakdownResponse] = await Promise.all([
        fetch('http://localhost:4000/api/wellness', {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        }),
        fetch('http://localhost:4000/api/wellness/breakdown', {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        })
      ]);

      if (!scoreResponse.ok || !breakdownResponse.ok) {
        throw new Error('Failed to fetch wellness data');
      }

      const scoreData = await scoreResponse.json();
      const breakdownData = await breakdownResponse.json();

      setWellnessData(scoreData.data);
      setBreakdown(breakdownData.data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchWellness();
  }, []);

  if (loading) return <div>Loading wellness data...</div>;
  if (error) return <div>Error: {error}</div>;
  if (!wellnessData) return <div>No wellness data available</div>;

  const getScoreColor = (score) => {
    if (score >= 80) return '#4caf50';
    if (score >= 60) return '#ff9800';
    return '#f44336';
  };

  return (
    <div style={{ padding: '20px' }}>
      <h2>Wellness Score</h2>
      <button onClick={fetchWellness} style={{ marginBottom: '20px' }}>
        Refresh
      </button>

      <div style={{ 
        border: '2px solid #ccc', 
        padding: '20px', 
        borderRadius: '8px',
        marginBottom: '20px',
        textAlign: 'center'
      }}>
        <div style={{ fontSize: '48px', fontWeight: 'bold', color: getScoreColor(wellnessData.overallScore) }}>
          {wellnessData.overallScore.toFixed(1)}
        </div>
        <div style={{ fontSize: '18px', marginTop: '10px' }}>Overall Wellness Score</div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px' }}>
        <div style={{ border: '1px solid #ccc', padding: '15px', borderRadius: '8px' }}>
          <h4>Adherence</h4>
          <div style={{ fontSize: '24px', color: getScoreColor(wellnessData.breakdown.adherenceScore) }}>
            {wellnessData.breakdown.adherenceScore.toFixed(1)}
          </div>
          <div style={{ fontSize: '12px', color: '#666' }}>Weight: 30%</div>
        </div>

        <div style={{ border: '1px solid #ccc', padding: '15px', borderRadius: '8px' }}>
          <h4>Metrics</h4>
          <div style={{ fontSize: '24px', color: getScoreColor(wellnessData.breakdown.metricScore) }}>
            {wellnessData.breakdown.metricScore.toFixed(1)}
          </div>
          <div style={{ fontSize: '12px', color: '#666' }}>Weight: 40%</div>
        </div>

        <div style={{ border: '1px solid #ccc', padding: '15px', borderRadius: '8px' }}>
          <h4>Stability</h4>
          <div style={{ fontSize: '24px', color: getScoreColor(wellnessData.breakdown.stabilityScore) }}>
            {wellnessData.breakdown.stabilityScore.toFixed(1)}
          </div>
          <div style={{ fontSize: '12px', color: '#666' }}>Weight: 20%</div>
        </div>

        <div style={{ border: '1px solid #ccc', padding: '15px', borderRadius: '8px' }}>
          <h4>Energy/Sleep</h4>
          <div style={{ fontSize: '24px', color: getScoreColor(wellnessData.breakdown.energyOrSleepScore) }}>
            {wellnessData.breakdown.energyOrSleepScore.toFixed(1)}
          </div>
          <div style={{ fontSize: '12px', color: '#666' }}>Weight: 10%</div>
        </div>
      </div>

      {breakdown && (
        <div style={{ marginTop: '20px', border: '1px solid #ccc', padding: '15px', borderRadius: '8px' }}>
          <h3>Detailed Breakdown</h3>
          <p><strong>Medications:</strong> {breakdown.adherence.medicationsCount}</p>
          <p><strong>Average Adherence:</strong> {breakdown.adherence.averageAdherence.toFixed(1)}%</p>
          <p><strong>Metrics Tracked:</strong> {breakdown.metrics.normalizedMetrics.length}</p>
          <p><strong>Stability Variability:</strong> {breakdown.stability.averageVariability.toFixed(2)}</p>
        </div>
      )}
    </div>
  );
};

export default WellnessScoreCard;







