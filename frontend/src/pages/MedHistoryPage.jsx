import React, { useState, useMemo, useCallback } from 'react';
import Navigation from '../components/Navigation';
import { useQuery } from '@tanstack/react-query';
import { fetcher } from '../api';
import { format, parseISO } from 'date-fns';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';

export default function MedHistoryPage() {
  const [selectedCycleId, setSelectedCycleId] = useState('');
  const [selectedMetric, setSelectedMetric] = useState('');
  const [graphType, setGraphType] = useState('line');

  const {
    data: cycles = [],
    isLoading: cyclesLoading,
    error: cyclesError
  } = useQuery({
    queryKey: ['medicationCycles'],
    queryFn: () => fetcher('/cycles'),
    staleTime: 5 * 60 * 1000,
    retry: 2
  });

  const {
    data: logs = [],
    isLoading: logsLoading,
    error: logsError
  } = useQuery({
    queryKey: ['metricLogs'],
    queryFn: () => fetcher('/metrics/logs'),
    staleTime: 5 * 60 * 1000,
    retry: 2
  });

  // Derive monitored metrics for selected cycle
  const monitoredMetrics = useMemo(() => {
    if (!selectedCycleId || !cycles.length) return [];
    const selectedCycle = cycles.find(cycle => cycle.id === selectedCycleId);
    return selectedCycle?.metrics || [];
  }, [selectedCycleId, cycles]);

  // Comprehensive log filtering
  const filteredLogs = useMemo(() => {
    if (!logs.length) return [];

    console.log('Filtering logs:', {
      totalLogs: logs.length,
      selectedCycleId,
      selectedMetric,
      cycles: cycles.map(c => ({ id: c.id, name: c.name })),
      sampleLogs: logs.slice(0, 3).map(l => ({ cycle: l.cycle, kind: l.kind }))
    });

    const filtered = logs.filter(log => {
      // If no cycle is selected, show all logs
      if (!selectedCycleId) return true;
      
      // Find the selected cycle to get its name
      const selectedCycle = cycles.find(c => c.id === selectedCycleId);
      if (!selectedCycle) return true;
      
      // Match logs by cycle name
      const cycleMatch = log.cycle === selectedCycle.name;
      
      // If metric is also selected, filter by both
      if (selectedMetric) {
        return cycleMatch && log.kind === selectedMetric;
      }
      
      return cycleMatch;
    });

    console.log('Filtered logs:', {
      filteredCount: filtered.length,
      selectedCycleName: cycles.find(c => c.id === selectedCycleId)?.name
    });

    return filtered;
  }, [logs, selectedCycleId, selectedMetric, cycles]);

  // Sorting logs by date (most recent first)
  const sortedLogs = useMemo(() =>
    [...filteredLogs].sort((a, b) => new Date(b.date) - new Date(a.date)),
    [filteredLogs]
  );

  // Prepare chart data - support multiple graph types
  const chartData = useMemo(() => {
    const numericLogs = filteredLogs
      .filter(r => r.valueFloat !== null && r.valueFloat !== undefined)
      .sort((a, b) => new Date(a.date) - new Date(b.date))
      .map(r => ({
        date: format(parseISO(r.date), 'MMM d'),
        value: r.valueFloat,
        kind: r.kind
      }));

    return numericLogs;
  }, [filteredLogs]);

  // Reset metric selection when cycle changes
  const handleCycleChange = useCallback((e) => {
    const newCycleId = e.target.value;
    setSelectedCycleId(newCycleId);
    setSelectedMetric('');
  }, []);

  // Render loading state
  if (cyclesLoading || logsLoading) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin text-4xl mb-4">🔄</div>
          <p className="text-gray-600 text-lg">Loading Medication History...</p>
        </div>
      </div>
    );
  }

  // Render error state
  if (cyclesError || logsError) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-center bg-white p-8 rounded-xl shadow-lg">
          <div className="text-6xl mb-4">😕</div>
          <h2 className="text-2xl font-bold text-red-600 mb-4">Unable to Load Metrics</h2>
          <p className="text-gray-600 mb-6">
            {cyclesError?.message || logsError?.message || 'An unexpected error occurred'}
          </p>
          <button
            onClick={() => window.location.reload()}
            className="bg-blue-500 text-white px-6 py-3 rounded-lg hover:bg-blue-600 transition"
          >
            Retry Loading
          </button>
        </div>
      </div>
    );
  }

  // Render empty state when user has no medication cycles
  if (!cyclesLoading && cycles.length === 0) {
    return (
      <div className="min-h-screen bg-gray-100">
        <Navigation />
        <div className="max-w-6xl mx-auto p-4">
          <div className="bg-white rounded-3xl shadow-lg p-6 mb-6">
            <h1 className="text-2xl font-bold mb-6 text-gray-800">Medication History</h1>
            
            <div className="text-center py-12">
              <div className="text-6xl mb-4">📋</div>
              <h2 className="text-xl font-semibold text-gray-700 mb-2">No Medications Yet</h2>
              <p className="text-gray-600 mb-6">
                You haven't added any medication cycles yet. Start tracking your health by adding your first medication.
              </p>
              <a
                href="/add-medication"
                className="inline-block bg-gradient-to-r from-blue-600 to-indigo-600 text-white px-6 py-3 rounded-xl font-medium hover:from-blue-700 hover:to-indigo-700 transition-all duration-200"
              >
                Add Your First Medication
              </a>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Render empty state when user has cycles but no metric logs
  if (!logsLoading && cycles.length > 0 && logs.length === 0) {
    return (
      <div className="min-h-screen bg-gray-100">
        <Navigation />
        <div className="max-w-6xl mx-auto p-4">
          <div className="bg-white rounded-3xl shadow-lg p-6 mb-6">
            <h1 className="text-2xl font-bold mb-6 text-gray-800">Medication History</h1>
            
            <div className="grid md:grid-cols-3 gap-4 mb-6">
              <select
                value={selectedCycleId}
                onChange={handleCycleChange}
                className="w-full rounded-xl border px-3 py-2 text-sm"
              >
                <option value="">All Medication Cycles</option>
                {cycles.map((cycle) => (
                  <option key={cycle.id} value={cycle.id}>
                    {cycle.name}
                  </option>
                ))}
              </select>
            </div>
            
            <div className="text-center py-12">
              <div className="text-6xl mb-4">📊</div>
              <h2 className="text-xl font-semibold text-gray-700 mb-2">No Health Metrics Yet</h2>
              <p className="text-gray-600 mb-6">
                You have medication cycles but haven't logged any health metrics yet. Start tracking your measurements to see your progress over time.
              </p>
              <a
                href="/add-metric"
                className="inline-block bg-gradient-to-r from-emerald-600 to-teal-600 text-white px-6 py-3 rounded-xl font-medium hover:from-emerald-700 hover:to-teal-700 transition-all duration-200"
              >
                Log Your First Metric
              </a>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100">
      <Navigation />
      <div className="max-w-6xl mx-auto p-4">
        <div className="bg-white rounded-3xl shadow-lg p-6 mb-6">
          <h1 className="text-2xl font-bold mb-6 text-gray-800">Medication History</h1>

          <div className="grid md:grid-cols-3 gap-4 mb-6">
            <select
              value={selectedCycleId}
              onChange={handleCycleChange}
              className="w-full rounded-xl border px-3 py-2 text-sm"
            >
              <option value="">All Medication Cycles</option>
              {cycles.map((cycle) => (
                <option key={cycle.id} value={cycle.id}>
                  {cycle.name}
                </option>
              ))}
            </select>

            {selectedCycleId && monitoredMetrics.length > 0 && (
              <select
                value={selectedMetric}
                onChange={(e) => setSelectedMetric(e.target.value)}
                className="w-full rounded-xl border px-3 py-2 text-sm"
              >
                <option value="">All Metrics</option>
                {monitoredMetrics.map((metric) => (
                  <option key={metric} value={metric}>
                    {metric}
                  </option>
                ))}
              </select>
            )}

            <div className="flex space-x-2">
              <button
                onClick={() => setGraphType('line')}
                className={`px-4 py-2 rounded-xl text-xs transition ${
                  graphType === 'line'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                Line Chart
              </button>
              <button
                onClick={() => setGraphType('bar')}
                className={`px-4 py-2 rounded-xl text-xs transition ${
                  graphType === 'bar'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                Bar Chart
              </button>
            </div>
          </div>

          {/* Chart Section */}
          {chartData.length > 1 && (
            <div className="mb-6 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl">
              <h4 className="font-medium text-sm mb-3">
                {selectedMetric ? `${selectedMetric} Trend` : 'Metric Trend'}
              </h4>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  {graphType === 'line' ? (
                    <LineChart data={chartData} margin={{ top: 10, right: 30, left: 20, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e0e7ff" />
                      <XAxis dataKey="date" tick={{ fontSize: 10 }} stroke="#6b7280" />
                      <YAxis tick={{ fontSize: 10 }} stroke="#6b7280" />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#f8fafc',
                          border: '1px solid #e2e8f0',
                          borderRadius: '8px',
                          fontSize: '12px'
                        }}
                        formatter={(value) => [value, selectedMetric || 'Metric']}
                      />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="value"
                        stroke="#3b82f6"
                        strokeWidth={2}
                        dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
                        activeDot={{ r: 6, stroke: '#3b82f6', strokeWidth: 2, fill: '#fff' }}
                      />
                    </LineChart>
                  ) : (
                    <BarChart data={chartData} margin={{ top: 10, right: 30, left: 20, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e0e7ff" />
                      <XAxis dataKey="date" tick={{ fontSize: 10 }} stroke="#6b7280" />
                      <YAxis tick={{ fontSize: 10 }} stroke="#6b7280" />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#f8fafc',
                          border: '1px solid #e2e8f0',
                          borderRadius: '8px',
                          fontSize: '12px'
                        }}
                        formatter={(value) => [value, selectedMetric || 'Metric']}
                      />
                      <Legend />
                      <Bar
                        dataKey="value"
                        fill="#3b82f6"
                        activeBar={{ fill: '#2563eb' }}
                      />
                    </BarChart>
                  )}
                </ResponsiveContainer>
              </div>
            </div>
          )}

          <div className="overflow-x-auto rounded-xl border border-gray-100">
            {sortedLogs.length > 0 ? (
              <table className="min-w-full text-xs">
                <thead className="bg-gray-100">
                  <tr>
                    <th className="p-3 text-left whitespace-nowrap">Date</th>
                    <th className="p-3 text-left whitespace-nowrap">Cycle</th>
                    <th className="p-3 text-left whitespace-nowrap">Type</th>
                    <th className="p-3 text-left break-words max-w-xs">Value</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedLogs.map((log) => (
                    <tr key={log.id} className="even:bg-gray-50 hover:bg-gray-100 transition">
                      <td className="p-3 whitespace-nowrap">
                        {format(parseISO(log.date), 'MMM d, yyyy')}
                      </td>
                      <td className="p-3 whitespace-nowrap">{log.cycle}</td>
                      <td className="p-3 whitespace-nowrap">{log.kind}</td>
                      <td className="p-3 break-words max-w-xs">
                        {log.valueFloat ?? log.valueText ?? '-'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="text-center py-6 text-gray-500">
                {selectedCycleId ? (
                  <div>
                    <p className="mb-2">
                      {selectedMetric 
                        ? `No logs found for ${cycles.find(c => c.id === selectedCycleId)?.name} - ${selectedMetric}`
                        : `No logs found for ${cycles.find(c => c.id === selectedCycleId)?.name}`
                      }
                    </p>
                    <p className="text-sm text-gray-400">
                      {logs.length > 0 
                        ? `Total logs available: ${logs.length}` 
                        : 'No logs available in the system'
                      }
                    </p>
                  </div>
                ) : (
                  "Select a medication to view logs."
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
