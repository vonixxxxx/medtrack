/**
 * Metric Trends Graph Component
 * Simple UI for testing trends engine
 */

import React, { useState, useEffect } from 'react';

const MetricTrendsGraph = () => {
  const [trendsData, setTrendsData] = useState(null);
  const [selectedMetric, setSelectedMetric] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchTrends = async () => {
    setLoading(true);
    setError(null);
    try {
      const token = localStorage.getItem('token');
      const response = await fetch('http://localhost:4000/api/metrics/trends', {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error('Failed to fetch trends data');
      }

      const data = await response.json();
      setTrendsData(data.data);
      
      // Select first metric by default
      if (data.data && Object.keys(data.data).length > 0) {
        setSelectedMetric(Object.keys(data.data)[0]);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTrends();
  }, []);

  if (loading) return <div>Loading trends data...</div>;
  if (error) return <div>Error: {error}</div>;
  if (!trendsData || Object.keys(trendsData).length === 0) {
    return <div>No trends data available</div>;
  }

  const currentTrend = selectedMetric ? trendsData[selectedMetric] : null;

  return (
    <div style={{ padding: '20px' }}>
      <h2>Metric Trends</h2>
      <button onClick={fetchTrends} style={{ marginBottom: '20px' }}>
        Refresh
      </button>

      <div style={{ marginBottom: '20px' }}>
        <label>
          Select Metric:
          <select
            value={selectedMetric || ''}
            onChange={(e) => setSelectedMetric(e.target.value)}
            style={{ marginLeft: '10px', padding: '5px' }}
          >
            {Object.keys(trendsData).map((metric) => (
              <option key={metric} value={metric}>
                {metric}
              </option>
            ))}
          </select>
        </label>
      </div>

      {currentTrend && (
        <div style={{ border: '1px solid #ccc', padding: '15px', borderRadius: '8px' }}>
          <h3>{selectedMetric}</h3>
          
          <div style={{ marginTop: '10px' }}>
            <p><strong>Trend:</strong> {currentTrend.trendClassification.trend}</p>
            <p><strong>Confidence:</strong> {(currentTrend.trendClassification.confidence * 100).toFixed(1)}%</p>
            <p><strong>Change:</strong> {currentTrend.trendClassification.changePercentage > 0 ? '+' : ''}
              {currentTrend.trendClassification.changePercentage.toFixed(1)}%</p>
            <p><strong>Current Value:</strong> {currentTrend.trajectory.current.toFixed(2)}</p>
            <p><strong>Projected Value:</strong> {currentTrend.trajectory.projected.toFixed(2)}</p>
            
            <div style={{ marginTop: '15px' }}>
              <strong>Moving Averages:</strong>
              <ul>
                <li>7-day: {currentTrend.movingAverages.sevenDay.length > 0 
                  ? currentTrend.movingAverages.sevenDay[currentTrend.movingAverages.sevenDay.length - 1].value.toFixed(2)
                  : 'N/A'}</li>
                <li>14-day: {currentTrend.movingAverages.fourteenDay.length > 0
                  ? currentTrend.movingAverages.fourteenDay[currentTrend.movingAverages.fourteenDay.length - 1].value.toFixed(2)
                  : 'N/A'}</li>
                <li>30-day: {currentTrend.movingAverages.thirtyDay.length > 0
                  ? currentTrend.movingAverages.thirtyDay[currentTrend.movingAverages.thirtyDay.length - 1].value.toFixed(2)
                  : 'N/A'}</li>
              </ul>
            </div>

            {currentTrend.anomalies.length > 0 && (
              <div style={{ marginTop: '15px' }}>
                <strong>Anomalies ({currentTrend.anomalies.length}):</strong>
                <ul>
                  {currentTrend.anomalies.slice(0, 5).map((anomaly, idx) => (
                    <li key={idx}>
                      {anomaly.timestamp.toLocaleString()}: {anomaly.value.toFixed(2)} ({anomaly.severity})
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default MetricTrendsGraph;







