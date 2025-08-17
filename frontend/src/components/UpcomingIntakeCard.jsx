import DashboardCard from './DashboardCard';
import { useQuery } from '@tanstack/react-query';
import { fetcher } from '../api';
import { colorDot } from './colorUtil';
import { format, isToday, isTomorrow, differenceInMinutes } from 'date-fns';

export default function UpcomingIntakeCard() {
  const { data: list = [] } = useQuery({ 
    queryKey: ['upcoming'], 
    queryFn: () => fetcher('/cycles/upcoming'),
    refetchInterval: 60000, // Refetch every minute to keep times accurate
  });

  const now = new Date();
  
  // Filter and sort upcoming doses
  const upcomingDoses = list
    .filter((i) => new Date(i.date) > now)
    .sort((a, b) => new Date(a.date) - new Date(b.date))
    .slice(0, 5); // Show next 5 doses

  const formatDateTime = (dateStr) => {
    const date = new Date(dateStr);
    
    if (isToday(date)) {
      return `Today at ${format(date, 'h:mm a')}`;
    } else if (isTomorrow(date)) {
      return `Tomorrow at ${format(date, 'h:mm a')}`;
    } else {
      return format(date, 'MMM d \'at\' h:mm a');
    }
  };

  const getTimeUntil = (dateStr) => {
    const date = new Date(dateStr);
    const minutesUntil = differenceInMinutes(date, now);
    
    if (minutesUntil < 60) {
      return `in ${minutesUntil}m`;
    } else if (minutesUntil < 1440) { // less than 24 hours
      const hours = Math.floor(minutesUntil / 60);
      return `in ${hours}h`;
    } else {
      const days = Math.floor(minutesUntil / 1440);
      return `in ${days}d`;
    }
  };

  const getPriorityColor = (dateStr) => {
    const date = new Date(dateStr);
    const minutesUntil = differenceInMinutes(date, now);
    
    if (minutesUntil < 60) return 'from-red-100 to-red-200 border-red-300';
    if (minutesUntil < 120) return 'from-orange-100 to-orange-200 border-orange-300';
    if (minutesUntil < 1440) return 'from-blue-100 to-blue-200 border-blue-300';
    return 'from-gray-100 to-gray-200 border-gray-300';
  };

  const getPriorityIcon = (dateStr) => {
    const date = new Date(dateStr);
    const minutesUntil = differenceInMinutes(date, now);
    
    if (minutesUntil < 60) return '🚨';
    if (minutesUntil < 120) return '⚠️';
    if (minutesUntil < 1440) return '⏰';
    return '📅';
  };

  return (
    <DashboardCard title="Upcoming Medication Intake">
      <div className="space-y-3">
        {upcomingDoses.length === 0 ? (
          <div className="text-center py-6">
            <div className="text-2xl mb-2">⏰</div>
            <p className="text-gray-600 text-sm">No upcoming medications</p>
          </div>
        ) : (
          <div className="max-h-48 overflow-y-auto custom-scrollbar pr-2 space-y-3">
            {upcomingDoses.map((intake, index) => (
              <div 
                key={`${intake.cycleId}-${intake.date}`} 
                className="group relative bg-white p-3 rounded-xl border border-gray-100 shadow-sm hover:shadow-md transition-all duration-300"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <p className="font-semibold text-sm text-gray-900 truncate max-w-32" title={intake.name}>
                        {intake.name}
                      </p>
                      <span className="text-lg flex-shrink-0">{getPriorityIcon(intake.date)}</span>
                    </div>
                    <p className="text-xs text-gray-600 mb-1 truncate max-w-32" title={intake.dosage}>
                      {intake.dosage}
                    </p>
                    {intake.totalDosesPerDay > 1 && (
                      <div className="inline-flex items-center px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-xs font-medium">
                        Dose {intake.doseNumber || 1} of {intake.totalDosesPerDay}
                      </div>
                    )}
                  </div>
                  <div className="text-right">
                    <div className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-gradient-to-r ${getPriorityColor(intake.date)} border`}>
                      {formatDateTime(intake.date)}
                    </div>
                    <p className="text-xs text-gray-500 mt-2 font-medium">
                      {getTimeUntil(intake.date)}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
        
        <div className="p-3 bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl border border-purple-100">
          <div className="flex items-center justify-between text-purple-700 w-full">
            <div className="flex items-center gap-2 flex-1 min-w-0">
              <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path>
              </svg>
              <span className="text-xs font-medium truncate">Upcoming Medication</span>
            </div>
            <span className="text-xs font-medium flex-shrink-0 ml-2">
              {upcomingDoses.length > 0 
                ? `${upcomingDoses.length} dose${upcomingDoses.length !== 1 ? 's' : ''} today` 
                : 'No doses scheduled'}
            </span>
          </div>
        </div>
      </div>
    </DashboardCard>
  );
}
