import DashboardCard from './DashboardCard';
import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetcher } from '../api';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { differenceInDays } from 'date-fns';

export default function CycleDetailCard() {
  const { data: cycles = [] } = useQuery({ queryKey: ['cycles'], queryFn: () => fetcher('/cycles') });
  const [activeCycleId, setActiveCycleId] = useState('');
  const [activeKind, setActiveKind] = useState('');

  // Get the currently selected cycle
  const cycle = cycles.find(c => c.id.toString() === activeCycleId) || cycles[0];

  // compute kinds every render (safe even if cycle null)
  const kinds = cycle ? [...new Set(cycle.metricLogs.map((m) => m.kind))] : [];

  // Initialize activeCycleId when cycles are loaded
  useEffect(() => {
    if (!activeCycleId && cycles.length > 0) {
      setActiveCycleId(cycles[0].id.toString());
    }
  }, [cycles, activeCycleId]);

  // initialise activeKind once kinds ready
  useEffect(() => {
    if (!activeKind && kinds.length) setActiveKind(kinds[0]);
  }, [kinds, activeKind]);

  if (!cycle) return <DashboardCard title="Cycle Detail">No Details to Display</DashboardCard>;

  const logsForKind = cycle.metricLogs
    .filter((m) => m.kind === activeKind)
    .sort((a, b) => new Date(a.date) - new Date(b.date));

  // Calculate cycle stats
  const daysOnMedication = differenceInDays(new Date(), new Date(cycle.startDate)) + 1;
  const intakesCompleted = cycle.doseLogs?.filter(d => d.taken).length || 0;
  const totalDoses = cycle.doseLogs?.length || 0;

  return (
    <DashboardCard title="Cycle Detail">
      {/* Cycle Selector */}
      {cycles.length > 1 && (
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Select Medication Cycle
          </label>
          <select
            value={activeCycleId}
            onChange={(e) => {
              setActiveCycleId(e.target.value);
              setActiveKind(''); // Reset kind when changing cycle
            }}
            className="w-full rounded-xl border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {cycles.map((cycle) => (
              <option key={cycle.id} value={cycle.id}>
                {cycle.name} - {cycle.dosage}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Cycle Stats */}
      <div className="mb-4 p-3 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl">
        <div className="flex justify-between items-start mb-2">
          <h3 className="font-semibold text-sm">{cycle.name}</h3>
          <span className="text-xs text-gray-500 bg-white/60 px-2 py-1 rounded-full">
            {cycle.dosage}
          </span>
        </div>
        <div className="grid grid-cols-2 gap-4 text-xs mb-2">
          <div>
            <span className="text-gray-500">Days on medication:</span>
            <span className="font-medium ml-1">{daysOnMedication}</span>
          </div>
          <div>
            <span className="text-gray-500">Intakes completed:</span>
            <span className="font-medium ml-1">{intakesCompleted}/{totalDoses}</span>
          </div>
        </div>
        <div className="text-xs text-gray-600">
          <span>Started: {new Date(cycle.startDate).toLocaleDateString()}</span>
          {cycle.endDate && (
            <span className="ml-3">
              Ends: {new Date(cycle.endDate).toLocaleDateString()}
            </span>
          )}
        </div>
      </div>
      {/* Kind Tabs */}
      {kinds.length > 0 ? (
        <div className="flex gap-2 mb-3 overflow-x-auto">
          {kinds.map((k) => (
            <button
              key={k}
              onClick={() => setActiveKind(k)}
              className={`px-3 py-1 rounded-full text-xs whitespace-nowrap ${k === activeKind ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-700'}`}
            >
              {k}
            </button>
          ))}
        </div>
      ) : (
        <div className="text-center py-8 text-gray-500">
          <div className="mb-2">📊</div>
          <p className="text-sm">No metrics logged for this cycle yet</p>
          <p className="text-xs mt-1">Add metrics using the "Add Metric" card</p>
        </div>
      )}

      {/* Table and Graph - only show if there are metrics */}
      {kinds.length > 0 && activeKind && (
        <>
          {/* Table */}
          <div className="overflow-x-auto max-h-48 rounded-2xl border border-gray-100">
            <table className="min-w-full text-xs">
              <thead className="bg-gray-50 sticky top-0">
                <tr>
                  <th className="p-2 text-left">Date</th>
                  <th className="p-2 text-left">Value</th>
                  <th className="p-2 text-left">Notes</th>
                </tr>
              </thead>
              <tbody>
                {logsForKind.map((m) => (
                  <tr key={m.id} className="even:bg-gray-50 hover:bg-gray-100">
                    <td className="p-2">{new Date(m.date).toLocaleDateString()}</td>
                    <td className="p-2">{m.valueFloat ?? m.valueText}</td>
                    <td className="p-2 text-gray-500">{m.notes || '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Graph preview */}
          {logsForKind.length > 1 && logsForKind.some(log => log.valueFloat !== null) && (
            <div className="mt-4 h-32">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart 
                  data={logsForKind
                    .filter(log => log.valueFloat !== null)
                    .map(log => ({
                      ...log,
                      dateFormatted: new Date(log.date).toLocaleDateString()
                    }))
                    .sort((a, b) => new Date(a.date) - new Date(b.date))
                  } 
                  margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
                >
                  <defs>
                    <linearGradient id="cycBlue" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.4} />
                      <stop offset="100%" stopColor="#3b82f6" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="dateFormatted" hide />
                  <YAxis hide />
                  <Tooltip 
                    formatter={(value, name) => [value, activeKind]}
                    labelFormatter={(label) => `Date: ${label}`}
                    contentStyle={{
                      backgroundColor: '#f8fafc',
                      border: '1px solid #e2e8f0',
                      borderRadius: '8px'
                    }}
                  />
                  <Area 
                    dataKey="valueFloat" 
                    type="monotone" 
                    stroke="#3b82f6" 
                    fill="url(#cycBlue)" 
                    strokeWidth={2} 
                    dot={{ fill: '#3b82f6', strokeWidth: 1, r: 3 }}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}

          <div className="flex gap-3 mt-4">
            <a
              href={`/api/cycles/${cycle.id}/report/csv?kind=${encodeURIComponent(activeKind)}`}
              className="btn-secondary flex-1 text-center"
              target="_blank"
              rel="noopener noreferrer"
            >
              Download CSV
            </a>
          </div>
        </>
      )}
    </DashboardCard>
  );
}
