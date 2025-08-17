import { useState, useEffect, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetcher } from '../api';
import DashboardCard from './DashboardCard';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import { format } from 'date-fns';

export default function MetricHistoryTable() {
  const { data: cycles = [] } = useQuery({ queryKey: ['cycles'], queryFn: () => fetcher('/cycles') });
  const rows = cycles.flatMap((c) => c.metricLogs.map((m) => ({ ...m, cycle: c.name, cycleId: c.id })));
  const kinds = [...new Set(rows.map((r) => r.kind))];
  const [sortDir, setSortDir] = useState('desc');

  // State for tracking selected cycle and metric
  const [activeCycleId, setActiveCycleId] = useState('');
  const [activeKind, setActiveKind] = useState('');

  // Determine monitored metrics for the selected cycle
  const monitoredMetrics = useMemo(() => {
    if (!activeCycleId) return [];
    
    const selectedCycle = cycles.find(cycle => cycle.id === activeCycleId);
    return selectedCycle?.metrics || [];
  }, [activeCycleId, cycles]);

  // Reset metric selection when cycle changes
  useEffect(() => {
    if (activeCycleId) {
      // Reset active kind when cycle changes
      setActiveKind('');
    }
  }, [activeCycleId]);

  // Filter logs based on selected cycle and metric
  const filteredLogs = rows.filter(r => 
    (!activeCycleId || r.cycleId.toString() === activeCycleId) &&
    (!activeKind || r.kind === activeKind)
  );

  // Prepare chart data - only numeric values
  const chartData = filteredLogs
    .filter(r => r.valueFloat !== null && r.valueFloat !== undefined)
    .sort((a, b) => new Date(a.date) - new Date(b.date))
    .map(r => ({
      date: format(new Date(r.date), 'MMM d'),
      value: r.valueFloat
    }));

  // Determine if chart data is available
  const hasChartData = chartData.length > 1;

  return (
    <DashboardCard title="Metric History">
      <div className="h-full flex flex-col">
        <div className="space-y-3 mb-4">
          <select
            value={activeCycleId}
            onChange={(e) => setActiveCycleId(e.target.value)}
            className="w-full rounded-xl border px-3 py-2 text-sm"
          >
            <option value="">All Medication Cycles</option>
            {cycles.map((cycle) => (
              <option key={cycle.id} value={cycle.id}>
                {cycle.name}
              </option>
            ))}
          </select>
          
          {activeCycleId && monitoredMetrics.length > 0 && (
            <div className="flex gap-2 overflow-x-auto">
              {monitoredMetrics.map((k) => (
                <button
                  key={k}
                  onClick={() => setActiveKind(k === activeKind ? '' : k)}
                  className={`px-3 py-1 rounded-full text-xs whitespace-nowrap ${k === activeKind ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-700'}`}
                >
                  {k}
                </button>
              ))}
            </div>
          )}
        </div>
        
        {/* Conditional rendering based on selection */}
        {activeCycleId ? (
          <>
            {/* Chart Section */}
            {hasChartData && (
              <div className="mb-4 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl">
                <h4 className="font-medium text-sm mb-3">
                  {activeKind ? `${activeKind} Trend` : 'Metric Trend'}
                </h4>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} margin={{ top: 10, right: 30, left: 20, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e0e7ff" />
                      <XAxis 
                        dataKey="date" 
                        tick={{ fontSize: 10 }}
                        stroke="#6b7280"
                      />
                      <YAxis 
                        tick={{ fontSize: 10 }}
                        stroke="#6b7280"
                      />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: '#f8fafc', 
                          border: '1px solid #e2e8f0',
                          borderRadius: '8px',
                          fontSize: '12px'
                        }}
                        formatter={(value) => [value, activeKind || 'Metric']}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="value" 
                        stroke="#3b82f6" 
                        strokeWidth={2}
                        dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
                        activeDot={{ r: 6, stroke: '#3b82f6', strokeWidth: 2, fill: '#fff' }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* Table Section */}
            <div className="flex-1 overflow-y-auto custom-scrollbar">
              <div className="overflow-x-auto rounded-xl border border-gray-100">
                {filteredLogs.length > 0 ? (
                  <table className="min-w-full text-xs">
                    <thead className="bg-gray-100">
                      <tr>
                        <th className="p-2 text-left">Date</th>
                        <th className="p-2 text-left">Cycle</th>
                        <th className="p-2 text-left">Type</th>
                        <th className="p-2 text-left">Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredLogs.map((r) => (
                        <tr key={r.id} className="even:bg-gray-50">
                          <td className="p-2">
                            {new Date(r.date).toLocaleDateString()}
                          </td>
                          <td className="p-2">{r.cycle}</td>
                          <td className="p-2">{r.kind}</td>
                          <td className="p-2">
                            {r.valueFloat ?? r.valueText ?? '-'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <div className="text-center py-6 text-gray-500">
                    {!activeKind 
                      ? "Select a metric to view logs" 
                      : "No logs available for the selected medication and metric"}
                  </div>
                )}
              </div>
            </div>
          </>
        ) : (
          <div className="text-center py-6 text-gray-500">
            Select a medication cycle to view metrics
          </div>
        )}
      </div>
    </DashboardCard>
  );
}
