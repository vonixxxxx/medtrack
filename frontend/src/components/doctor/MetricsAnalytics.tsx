import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { api } from '../../utils/api';

interface MetricData {
  id: string;
  metric_name: string;
  value: number;
  unit: string;
  date: string;
  change_from_previous?: number;
  change_percentage?: number;
  trend_direction?: string;
  source_type: string;
}

interface LabResult {
  id: string;
  metric_name: string;
  value: number;
  unit: string;
  date: string;
  reference_range?: string;
  status?: string;
}

interface VitalSign {
  id: string;
  vital_type: string;
  value: number;
  unit: string;
  date: string;
  value_secondary?: number;
}

interface MetricsAnalyticsProps {
  patientId: string;
  patientName: string;
}

export const MetricsAnalytics: React.FC<MetricsAnalyticsProps> = ({ patientId, patientName }) => {
  const [metrics, setMetrics] = useState<MetricData[]>([]);
  const [labResults, setLabResults] = useState<LabResult[]>([]);
  const [vitalSigns, setVitalSigns] = useState<VitalSign[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedMetric, setSelectedMetric] = useState<string>('all');
  const [dateRange, setDateRange] = useState<{ start: string; end: string }>({
    start: new Date(Date.now() - 365 * 24 * 60 * 60 * 1000).toISOString().split('T')[0], // 1 year ago
    end: new Date().toISOString().split('T')[0] // today
  });
  const [viewType, setViewType] = useState<'line' | 'bar'>('line');

  useEffect(() => {
    fetchMetricsData();
  }, [patientId]);

  const fetchMetricsData = async () => {
    try {
      setLoading(true);
      
      // Fetch all metrics data
      const [metricsRes, labRes, vitalsRes] = await Promise.all([
        api.get(`metrics/patient/${patientId}`),
        api.get(`lab-results/patient/${patientId}`),
        api.get(`vital-signs/patient/${patientId}`)
      ]);

      setMetrics(metricsRes.data || []);
      setLabResults(labRes.data || []);
      setVitalSigns(vitalsRes.data || []);
    } catch (error) {
      console.error('Error fetching metrics data:', error);
    } finally {
      setLoading(false);
    }
  };

  // Combine all data sources
  const allData = [
    ...metrics.map(m => ({
      ...m,
      source: 'metric_trend',
      displayName: m.metric_name
    })),
    ...labResults.map(l => ({
      ...l,
      source: 'lab_result',
      displayName: l.metric_name,
      value_secondary: undefined
    })),
    ...vitalSigns.map(v => ({
      ...v,
      source: 'vital_sign',
      displayName: v.vital_type,
      metric_name: v.vital_type
    }))
  ];

  // Filter data by selected metric and date range
  const filteredData = allData.filter(item => {
    const itemDate = new Date(item.date);
    const startDate = new Date(dateRange.start);
    const endDate = new Date(dateRange.end);
    
    const metricMatch = selectedMetric === 'all' || item.metric_name === selectedMetric;
    const dateMatch = itemDate >= startDate && itemDate <= endDate;
    
    return metricMatch && dateMatch;
  });

  // Group data by metric for charting
  const chartData = filteredData.reduce((acc, item) => {
    const date = item.date.split('T')[0];
    const existing = acc.find(d => d.date === date);
    
    if (existing) {
      existing[item.metric_name] = item.value;
      if (item.value_secondary) {
        existing[`${item.metric_name}_secondary`] = item.value_secondary;
      }
    } else {
      const newData = { date };
      newData[item.metric_name] = item.value;
      if (item.value_secondary) {
        newData[`${item.metric_name}_secondary`] = item.value_secondary;
      }
      acc.push(newData);
    }
    
    return acc;
  }, [] as any[]).sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());

  // Get unique metrics for filter dropdown
  const availableMetrics = ['all', ...new Set(allData.map(item => item.metric_name))];

  // Calculate trend statistics
  const calculateTrend = (metricName: string) => {
    const metricData = filteredData.filter(item => item.metric_name === metricName);
    if (metricData.length < 2) return null;

    const firstValue = metricData[0].value;
    const lastValue = metricData[metricData.length - 1].value;
    const change = lastValue - firstValue;
    const changePercent = (change / firstValue) * 100;

    return {
      change,
      changePercent,
      direction: change > 0 ? 'increasing' : change < 0 ? 'decreasing' : 'stable',
      firstValue,
      lastValue
    };
  };

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="animate-pulse">
          <div className="h-6 bg-gray-700 rounded w-1/4 mb-4"></div>
          <div className="h-64 bg-gray-700 rounded"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <div className="flex justify-between items-center mb-6">
        <h3 className="text-xl font-semibold text-white">
          Metrics Analytics - {patientName}
        </h3>
        <div className="flex space-x-4">
          <select
            value={selectedMetric}
            onChange={(e) => setSelectedMetric(e.target.value)}
            className="bg-gray-700 text-white px-3 py-2 rounded border border-gray-600"
          >
            {availableMetrics.map(metric => (
              <option key={metric} value={metric}>
                {metric === 'all' ? 'All Metrics' : metric}
              </option>
            ))}
          </select>
          <select
            value={viewType}
            onChange={(e) => setViewType(e.target.value as 'line' | 'bar')}
            className="bg-gray-700 text-white px-3 py-2 rounded border border-gray-600"
          >
            <option value="line">Line Chart</option>
            <option value="bar">Bar Chart</option>
          </select>
        </div>
      </div>

      {/* Date Range Filter */}
      <div className="flex space-x-4 mb-6">
        <div>
          <label className="block text-sm text-gray-300 mb-1">Start Date</label>
          <input
            type="date"
            value={dateRange.start}
            onChange={(e) => setDateRange(prev => ({ ...prev, start: e.target.value }))}
            className="bg-gray-700 text-white px-3 py-2 rounded border border-gray-600"
          />
        </div>
        <div>
          <label className="block text-sm text-gray-300 mb-1">End Date</label>
          <input
            type="date"
            value={dateRange.end}
            onChange={(e) => setDateRange(prev => ({ ...prev, end: e.target.value }))}
            className="bg-gray-700 text-white px-3 py-2 rounded border border-gray-600"
          />
        </div>
      </div>

      {/* Metrics Table */}
      <div className="mb-6">
        <h4 className="text-lg font-medium text-white mb-3">Metrics Over Time</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-gray-300">
            <thead>
              <tr className="border-b border-gray-700">
                <th className="text-left py-2">Date</th>
                <th className="text-left py-2">Metric</th>
                <th className="text-left py-2">Value</th>
                <th className="text-left py-2">Unit</th>
                <th className="text-left py-2">Source</th>
                <th className="text-left py-2">Change</th>
              </tr>
            </thead>
            <tbody>
              {filteredData.slice(0, 10).map((item, index) => {
                const trend = calculateTrend(item.metric_name);
                return (
                  <tr key={index} className="border-b border-gray-800">
                    <td className="py-2">{new Date(item.date).toLocaleDateString()}</td>
                    <td className="py-2">{item.metric_name}</td>
                    <td className="py-2">{item.value.toFixed(2)}</td>
                    <td className="py-2">{item.unit}</td>
                    <td className="py-2">
                      <span className={`px-2 py-1 rounded text-xs ${
                        item.source === 'lab_result' ? 'bg-blue-900 text-blue-300' :
                        item.source === 'vital_sign' ? 'bg-green-900 text-green-300' :
                        'bg-purple-900 text-purple-300'
                      }`}>
                        {item.source.replace('_', ' ')}
                      </span>
                    </td>
                    <td className="py-2">
                      {trend && (
                        <span className={`px-2 py-1 rounded text-xs ${
                          trend.direction === 'increasing' ? 'bg-red-900 text-red-300' :
                          trend.direction === 'decreasing' ? 'bg-green-900 text-green-300' :
                          'bg-gray-700 text-gray-300'
                        }`}>
                          {trend.changePercent > 0 ? '+' : ''}{trend.changePercent.toFixed(1)}%
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Interactive Chart */}
      <div className="mb-6">
        <h4 className="text-lg font-medium text-white mb-3">Trend Visualization</h4>
        <div className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            {viewType === 'line' ? (
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis 
                  dataKey="date" 
                  stroke="#9CA3AF"
                  tickFormatter={(value) => new Date(value).toLocaleDateString()}
                />
                <YAxis stroke="#9CA3AF" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1F2937', 
                    border: '1px solid #374151',
                    color: '#F3F4F6'
                  }}
                  labelFormatter={(value) => new Date(value).toLocaleDateString()}
                />
                <Legend />
                {availableMetrics.filter(m => m !== 'all').map((metric, index) => (
                  <Line
                    key={metric}
                    type="monotone"
                    dataKey={metric}
                    stroke={`hsl(${index * 60}, 70%, 50%)`}
                    strokeWidth={2}
                    dot={{ fill: `hsl(${index * 60}, 70%, 50%)`, strokeWidth: 2, r: 4 }}
                    connectNulls={false}
                  />
                ))}
              </LineChart>
            ) : (
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis 
                  dataKey="date" 
                  stroke="#9CA3AF"
                  tickFormatter={(value) => new Date(value).toLocaleDateString()}
                />
                <YAxis stroke="#9CA3AF" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1F2937', 
                    border: '1px solid #374151',
                    color: '#F3F4F6'
                  }}
                  labelFormatter={(value) => new Date(value).toLocaleDateString()}
                />
                <Legend />
                {availableMetrics.filter(m => m !== 'all').map((metric, index) => (
                  <Bar
                    key={metric}
                    dataKey={metric}
                    fill={`hsl(${index * 60}, 70%, 50%)`}
                  />
                ))}
              </BarChart>
            )}
          </ResponsiveContainer>
        </div>
      </div>

      {/* Summary Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gray-700 rounded-lg p-4">
          <h5 className="text-sm font-medium text-gray-300 mb-2">Total Data Points</h5>
          <p className="text-2xl font-bold text-white">{filteredData.length}</p>
        </div>
        <div className="bg-gray-700 rounded-lg p-4">
          <h5 className="text-sm font-medium text-gray-300 mb-2">Metrics Tracked</h5>
          <p className="text-2xl font-bold text-white">
            {new Set(filteredData.map(item => item.metric_name)).size}
          </p>
        </div>
        <div className="bg-gray-700 rounded-lg p-4">
          <h5 className="text-sm font-medium text-gray-300 mb-2">Date Range</h5>
          <p className="text-sm text-white">
            {new Date(dateRange.start).toLocaleDateString()} - {new Date(dateRange.end).toLocaleDateString()}
          </p>
        </div>
      </div>
    </div>
  );
};
