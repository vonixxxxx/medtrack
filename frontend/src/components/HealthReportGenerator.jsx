/**
 * Health Report Generator Component
 * Simple UI for testing health report generation
 */

import React, { useState } from 'react';

const HealthReportGenerator = () => {
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [timeframe, setTimeframe] = useState('30d');

  const generateReport = async () => {
    setLoading(true);
    setError(null);
    setReport(null);

    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`http://localhost:4000/api/health-report?timeframe=${timeframe}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error('Failed to generate health report');
      }

      const data = await response.json();
      setReport(data.data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const downloadReport = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`http://localhost:4000/api/health-report/download?timeframe=${timeframe}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error('Failed to download report');
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `health-report-${Date.now()}.json`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      alert('Failed to download report: ' + err.message);
    }
  };

  return (
    <div style={{ padding: '20px' }}>
      <h2>Health Report Generator</h2>
      
      <div style={{ marginBottom: '20px' }}>
        <label>
          Timeframe:
          <select
            value={timeframe}
            onChange={(e) => setTimeframe(e.target.value)}
            style={{ marginLeft: '10px', padding: '5px' }}
          >
            <option value="7d">7 days</option>
            <option value="14d">14 days</option>
            <option value="30d">30 days</option>
            <option value="60d">60 days</option>
            <option value="90d">90 days</option>
          </select>
        </label>
      </div>

      <div style={{ marginBottom: '20px' }}>
        <button 
          onClick={generateReport} 
          disabled={loading}
          style={{ marginRight: '10px', padding: '10px 20px' }}
        >
          {loading ? 'Generating...' : 'Generate Health Report'}
        </button>
        
        {report && (
          <button 
            onClick={downloadReport}
            style={{ padding: '10px 20px' }}
          >
            Download JSON
          </button>
        )}
      </div>

      {error && (
        <div style={{ color: 'red', marginBottom: '20px' }}>
          Error: {error}
        </div>
      )}

      {report && (
        <div style={{ border: '1px solid #ccc', padding: '20px', borderRadius: '8px' }}>
          <h3>Health Report - {report.timeframe}</h3>
          <p><strong>Generated:</strong> {new Date(report.generatedAt).toLocaleString()}</p>

          <div style={{ marginTop: '20px', marginBottom: '20px' }}>
            <h4>Wellness Score: {report.wellnessScore.overallScore.toFixed(1)}/100</h4>
          </div>

          <div style={{ marginTop: '20px' }}>
            <h4>Narrative Summary</h4>
            <div style={{ backgroundColor: '#f5f5f5', padding: '15px', borderRadius: '5px', marginTop: '10px' }}>
              <p><strong>Overall Status:</strong> {report.narrativeSummary.overallStatus}</p>
              <p><strong>Progress:</strong> {report.narrativeSummary.progress}</p>
              <p><strong>Medication Adherence:</strong> {report.narrativeSummary.medicationAdherence}</p>
              <p><strong>Metric Trends:</strong> {report.narrativeSummary.metricTrends}</p>
              <p><strong>Notable Events:</strong> {report.narrativeSummary.notableEvents}</p>
              <p><strong>Wellness Score Interpretation:</strong> {report.narrativeSummary.wellnessScoreInterpretation}</p>
              
              {report.narrativeSummary.recommendations.length > 0 && (
                <div style={{ marginTop: '15px' }}>
                  <strong>Recommendations:</strong>
                  <ul>
                    {report.narrativeSummary.recommendations.map((rec, idx) => (
                      <li key={idx}>{rec}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>

          <div style={{ marginTop: '20px' }}>
            <h4>Adherence Summary</h4>
            <p>Overall Adherence: {report.adherenceSummary.overallAdherence.toFixed(1)}%</p>
            <p>Pattern: {report.adherenceSummary.pattern}</p>
            <p>Medications: {report.adherenceSummary.medications.length}</p>
          </div>

          <div style={{ marginTop: '20px' }}>
            <h4>Metric Trends</h4>
            <ul>
              {report.metricTrendSummaries.slice(0, 5).map((metric, idx) => (
                <li key={idx}>
                  {metric.metricName}: {metric.trend} ({metric.changePercentage > 0 ? '+' : ''}{metric.changePercentage.toFixed(1)}%)
                </li>
              ))}
            </ul>
          </div>

          {report.anomalies.length > 0 && (
            <div style={{ marginTop: '20px' }}>
              <h4>Anomalies ({report.anomalies.length})</h4>
              <ul>
                {report.anomalies.slice(0, 5).map((anomaly, idx) => (
                  <li key={idx}>
                    {anomaly.metricName}: {anomaly.value} ({anomaly.severity}) - {new Date(anomaly.timestamp).toLocaleString()}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default HealthReportGenerator;







