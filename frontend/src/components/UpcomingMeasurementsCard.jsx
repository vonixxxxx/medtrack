import { useQuery } from '@tanstack/react-query';
import { fetcher } from '../api';
import DashboardCard from './DashboardCard';
import { colorDot } from './colorUtil';
import { format, isToday, isTomorrow } from 'date-fns';

function statusColor(date) {
  const today = new Date();
  const d = new Date(date);
  const diff = Math.floor((d - today) / (1000 * 60 * 60 * 24));
  if (diff < 0) return 'bg-red-500';
  if (diff === 0) return 'bg-amber-400';
  return 'bg-green-500';
}

export default function UpcomingMeasurementsCard() {
  const { data: items = [] } = useQuery({ 
    queryKey: ['metricReminders'], 
    queryFn: () => fetcher('/cycles/metric-reminders'),
    refetchInterval: 60000, // Refetch every minute
  });

  const formatDueDate = (dateStr, daysUntilDue) => {
    const date = new Date(dateStr);
    
    if (isToday(date)) {
      return 'Due today';
    } else if (isTomorrow(date)) {
      return 'Due tomorrow';
    } else if (daysUntilDue < 0) {
      return `Overdue by ${Math.abs(daysUntilDue)} day${Math.abs(daysUntilDue) > 1 ? 's' : ''}`;
    } else {
      return `Due ${format(date, 'MMM d')}`;
    }
  };

  const getStatusColor = (status, daysUntilDue) => {
    if (status === 'overdue' || daysUntilDue < 0) return 'text-red-600';
    if (status === 'due' || daysUntilDue === 0) return 'text-amber-600';
    return 'text-blue-600';
  };

  // Sort by days until due (overdue first, then due soon)
  const sortedItems = [...items].sort((a, b) => a.daysUntilDue - b.daysUntilDue);

  return (
    <DashboardCard title="Upcoming Measurements">
      <div className="space-y-3">
        {sortedItems.length === 0 ? (
          <div className="text-center py-6">
            <div className="text-2xl mb-2">📊</div>
            <p className="text-gray-600 text-sm">All measurements up to date! 🎉</p>
          </div>
        ) : (
          <div className="max-h-48 overflow-y-auto custom-scrollbar pr-2 space-y-3">
            {sortedItems.map((reminder) => (
              <div
                key={`${reminder.cycleId}-${reminder.metricType}`}
                className="group relative bg-white p-3 rounded-xl border border-gray-100 shadow-sm hover:shadow-md transition-all duration-300"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <h3 className="font-semibold text-sm text-gray-900 mb-1">
                      {reminder.metricType}
                    </h3>
                    <div className="flex items-center gap-2 text-xs text-gray-500">
                      <span className="font-medium">{reminder.cycleName}</span>
                      <span className="w-1 h-1 bg-gray-300 rounded-full"></span>
                      <span className="capitalize">{reminder.frequency}</span>
                    </div>
                  </div>
                  <div className="text-right">
                    <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(reminder.status, reminder.daysUntilDue)} bg-opacity-10 border`}>
                      <span className="w-2 h-2 bg-current rounded-full mr-1 animate-pulse"></span>
                      {formatDueDate(reminder.nextDueDate, reminder.daysUntilDue)}
                    </span>
                    {reminder.daysUntilDue > 0 && (
                      <p className="text-xs text-gray-500 mt-1 font-medium">
                        in {reminder.daysUntilDue} day{reminder.daysUntilDue > 1 ? 's' : ''}
                      </p>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
        
        <div className="p-3 bg-gradient-to-r from-emerald-50 to-teal-50 rounded-xl border border-emerald-100">
          <div className="flex items-center justify-between text-emerald-700 w-full">
            <div className="flex items-center gap-2 flex-1 min-w-0">
              <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2zm0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
              </svg>
              <span className="text-xs font-medium truncate">Next {sortedItems.length} measurement{sortedItems.length !== 1 ? 's' : ''}</span>
            </div>
            <span className="text-xs font-medium flex-shrink-0 ml-2">
              {sortedItems.length > 0 ? formatDueDate(sortedItems[0].nextDueDate, sortedItems[0].daysUntilDue) : 'None scheduled'}
            </span>
          </div>
        </div>
      </div>
    </DashboardCard>
  );
}
