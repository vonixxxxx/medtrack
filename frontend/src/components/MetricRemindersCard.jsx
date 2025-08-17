import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetcher } from '../api';
import DashboardCard from './DashboardCard';
import { getColorForName } from './colorUtil';

export default function MetricRemindersCard() {
  const { data: reminders = [] } = useQuery({
    queryKey: ['metric-reminders'],
    queryFn: () => fetcher('/cycles/metric-reminders'),
    refetchInterval: 30000 // Refetch every 30 seconds
  });

  // Filter out reminders that are no longer due (logic fix)
  const activeReminders = reminders.filter(reminder => {
    // If daysSinceLastLog is 0 or negative, it's not overdue
    return reminder.daysSinceLastLog > 0;
  });

    return (
    <DashboardCard title="Metric Update Reminders">
      <div className="space-y-3">
        {activeReminders.length === 0 ? (
          <div className="text-center py-6">
            <div className="text-2xl mb-2">📊</div>
            <p className="text-gray-600 text-sm">All metrics up to date! 🎉</p>
          </div>
        ) : (
          <div className="max-h-48 overflow-y-auto custom-scrollbar pr-2 space-y-3">
            {activeReminders.map((reminder, index) => (
              <div
                key={`${reminder.cycleId}-${reminder.metricType}-${index}`}
                className="group relative bg-gradient-to-r from-white to-gray-50 p-4 rounded-2xl border border-gray-100 shadow-sm hover:shadow-md transition-all duration-300 hover:border-gray-200"
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start space-x-3 flex-1">
                    <div className="relative">
                      <div
                        className="w-4 h-4 rounded-full shadow-sm"
                        style={{ backgroundColor: getColorForName(reminder.cycleName) }}
                      ></div>
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="font-semibold text-sm text-gray-900 mb-1">
                        {reminder.metricType}
                      </p>
                      <div className="flex items-center gap-2 text-xs text-gray-600">
                        <span className="font-medium">{reminder.cycleName}</span>
                        <span className="w-1 h-1 bg-gray-400 rounded-full"></span>
                        <span className="capitalize">{reminder.frequency}</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex flex-col items-end gap-2">
                    <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-gradient-to-r from-orange-100 to-red-100 text-orange-800 border border-orange-200">
                      <span className="w-2 h-2 bg-orange-500 rounded-full mr-2 animate-pulse"></span>
                      {reminder.daysSinceLastLog > 1 ? `${reminder.daysSinceLastLog} days overdue` : '1 day overdue'}
                    </span>
                  </div>
                </div>
                
                {/* Progress indicator */}
                <div className="mt-3 pt-3 border-t border-gray-100">
                  <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
                    <span>Last updated</span>
                    <span>{reminder.daysSinceLastLog} days ago</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-gradient-to-r from-orange-400 to-red-500 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${Math.min(reminder.daysSinceLastLog * 20, 100)}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
        
        {/* Helpful tip */}
        <div className="p-3 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl border border-blue-100">
          <div className="flex items-center justify-between text-blue-700">
            <div className="flex items-center gap-2">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
              </svg>
              <span className="text-xs font-medium">Metric Logging</span>
            </div>
            <span className="text-xs font-medium">
              Use "Update Metrics" card
            </span>
          </div>
        </div>
      </div>
    </DashboardCard>
  );
}
