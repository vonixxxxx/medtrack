import { useState, useEffect } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import { Calendar as CalendarIcon, CheckCircle, XCircle, Clock, TrendingUp } from 'lucide-react';
import DashboardCard from '../DashboardCard';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { LoadingSkeleton } from '../dashboard/LoadingSkeleton';
import { getAdherenceCalendar, logAdherence } from '../../api';

export const AdherenceCalendar = ({ medicationId, patientId }) => {
  const prefersReducedMotion = useReducedMotion();
  const [calendar, setCalendar] = useState({});
  const [currentMonth, setCurrentMonth] = useState(new Date().getMonth());
  const [currentYear, setCurrentYear] = useState(new Date().getFullYear());
  const [isLoading, setIsLoading] = useState(true);
  const [statistics, setStatistics] = useState(null);

  useEffect(() => {
    loadCalendar();
  }, [medicationId, patientId, currentMonth, currentYear]);

  const loadCalendar = async () => {
    try {
      setIsLoading(true);
      const params = {
        medicationId,
        patientId,
        year: currentYear,
        month: currentMonth + 1
      };
      const data = await getAdherenceCalendar(params);
      setCalendar(data?.calendar || {});
      setStatistics(data?.statistics || null);
    } catch (error) {
      console.error('Error loading adherence calendar:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDateClick = async (date) => {
    const dateKey = date.toISOString().split('T')[0];
    const existingLog = calendar[dateKey]?.[0];
    
    if (existingLog?.status === 'taken') {
      // Mark as missed
      await logAdherence({
        medicationId,
        date: dateKey,
        status: 'missed'
      });
    } else {
      // Mark as taken
      await logAdherence({
        medicationId,
        date: dateKey,
        status: 'taken',
        takenTime: new Date().toISOString()
      });
    }
    loadCalendar();
  };

  const getDaysInMonth = (month, year) => {
    return new Date(year, month + 1, 0).getDate();
  };

  const getFirstDayOfMonth = (month, year) => {
    return new Date(year, month, 1).getDay();
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'taken':
        return 'bg-medical-500';
      case 'missed':
        return 'bg-error-500';
      case 'skipped':
        return 'bg-warning-500';
      case 'delayed':
        return 'bg-info-500';
      default:
        return 'bg-neutral-200';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'taken':
        return <CheckCircle className="w-4 h-4 text-white" />;
      case 'missed':
        return <XCircle className="w-4 h-4 text-white" />;
      case 'skipped':
        return <Clock className="w-4 h-4 text-white" />;
      default:
        return null;
    }
  };

  const daysInMonth = getDaysInMonth(currentMonth, currentYear);
  const firstDay = getFirstDayOfMonth(currentMonth, currentYear);
  const days = [];

  // Add empty cells for days before the first day of the month
  for (let i = 0; i < firstDay; i++) {
    days.push(null);
  }

  // Add cells for each day of the month
  for (let day = 1; day <= daysInMonth; day++) {
    const date = new Date(currentYear, currentMonth, day);
    const dateKey = date.toISOString().split('T')[0];
    const dayLogs = calendar[dateKey] || [];
    const status = dayLogs[0]?.status;
    
    days.push({
      day,
      date,
      dateKey,
      status,
      logs: dayLogs
    });
  }

  const navigateMonth = (direction) => {
    if (direction === 'prev') {
      if (currentMonth === 0) {
        setCurrentMonth(11);
        setCurrentYear(currentYear - 1);
      } else {
        setCurrentMonth(currentMonth - 1);
      }
    } else {
      if (currentMonth === 11) {
        setCurrentMonth(0);
        setCurrentYear(currentYear + 1);
      } else {
        setCurrentMonth(currentMonth + 1);
      }
    }
  };

  const monthNames = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
  ];

  const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

  return (
    <DashboardCard
      title="Adherence Calendar"
      icon={<CalendarIcon size={20} />}
      variant="patient"
    >
      {isLoading ? (
        <LoadingSkeleton variant="card" />
      ) : (
        <div className="space-y-6">
          {/* Statistics */}
          {statistics && (
            <div className="grid grid-cols-3 gap-4">
              <div className="text-center p-4 bg-medical-50 rounded-xl">
                <div className="text-2xl font-bold text-medical-700">
                  {statistics.adherenceRate?.toFixed(0) || 0}%
                </div>
                <div className="text-xs text-medical-600 mt-1">Adherence</div>
              </div>
              <div className="text-center p-4 bg-white rounded-xl border border-neutral-200">
                <div className="text-2xl font-bold text-neutral-900">
                  {statistics.taken || 0}
                </div>
                <div className="text-xs text-neutral-600 mt-1">Taken</div>
              </div>
              <div className="text-center p-4 bg-error-50 rounded-xl">
                <div className="text-2xl font-bold text-error-700">
                  {statistics.missed || 0}
                </div>
                <div className="text-xs text-error-600 mt-1">Missed</div>
              </div>
            </div>
          )}

          {/* Calendar Navigation */}
          <div className="flex items-center justify-between">
            <Button
              onClick={() => navigateMonth('prev')}
              variant="ghost"
              size="sm"
            >
              ← Prev
            </Button>
            <h3 className="text-lg font-semibold text-neutral-900">
              {monthNames[currentMonth]} {currentYear}
            </h3>
            <Button
              onClick={() => navigateMonth('next')}
              variant="ghost"
              size="sm"
            >
              Next →
            </Button>
          </div>

          {/* Calendar Grid */}
          <div className="grid grid-cols-7 gap-2">
            {/* Day Headers */}
            {dayNames.map(day => (
              <div
                key={day}
                className="text-center text-xs font-semibold text-neutral-600 py-2"
              >
                {day}
              </div>
            ))}

            {/* Calendar Days */}
            {days.map((dayData, index) => {
              if (!dayData) {
                return <div key={`empty-${index}`} className="aspect-square" />;
              }

              const isToday = dayData.dateKey === new Date().toISOString().split('T')[0];
              
              return (
                <motion.button
                  key={dayData.dateKey}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: index * 0.01 }}
                  onClick={() => handleDateClick(dayData.date)}
                  className={`aspect-square rounded-xl border-2 transition-all ${
                    isToday
                      ? 'border-primary-500 ring-2 ring-primary-200'
                      : 'border-neutral-200 hover:border-primary-300'
                  } ${
                    dayData.status ? getStatusColor(dayData.status) : 'bg-white'
                  } flex flex-col items-center justify-center relative`}
                >
                  <span
                    className={`text-sm font-semibold ${
                      dayData.status ? 'text-white' : 'text-neutral-900'
                    }`}
                  >
                    {dayData.day}
                  </span>
                  {dayData.status && (
                    <div className="mt-1">
                      {getStatusIcon(dayData.status)}
                    </div>
                  )}
                </motion.button>
              );
            })}
          </div>

          {/* Legend */}
          <div className="flex items-center justify-center gap-4 text-xs">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded bg-medical-500" />
              <span className="text-neutral-600">Taken</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded bg-error-500" />
              <span className="text-neutral-600">Missed</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded bg-warning-500" />
              <span className="text-neutral-600">Skipped</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded border-2 border-primary-500" />
              <span className="text-neutral-600">Today</span>
            </div>
          </div>
        </div>
      )}
    </DashboardCard>
  );
};

