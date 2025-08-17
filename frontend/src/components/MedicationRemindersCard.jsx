import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { fetcher, poster } from '../api';
import DashboardCard from './DashboardCard';
import { colorDot } from './colorUtil';

export default function MedicationRemindersCard() {
  const queryClient = useQueryClient();
  const [animatingButtons, setAnimatingButtons] = useState(new Set());
  const { data: items = [], refetch } = useQuery({ 
    queryKey: ['reminders'], 
    queryFn: () => fetcher('/cycles/today') 
  });

  const toggleTaken = async (cycleId, date, taken) => {
    const buttonId = `${cycleId}-${date}`;
    
    // Start animation
    setAnimatingButtons(prev => new Set([...prev, buttonId]));
    
    try {
      await poster(`/cycles/${cycleId}/dose`, { date, taken });
      
      // Keep animation for a bit longer to show success
      setTimeout(() => {
        setAnimatingButtons(prev => {
          const newSet = new Set(prev);
          newSet.delete(buttonId);
          return newSet;
        });
      }, 1000);
      
      // Force immediate refresh
      await refetch();
      // invalidate related queries
      queryClient.invalidateQueries({ queryKey: ['upcoming'] });
      queryClient.invalidateQueries({ queryKey: ['cycles'] });
    } catch (error) {
      // Stop animation on error
      setAnimatingButtons(prev => {
        const newSet = new Set(prev);
        newSet.delete(buttonId);
        return newSet;
      });
      console.error('Error marking dose as taken:', error);
      alert('Failed to mark dose as taken. Please try again.');
    }
  };

  // Calculate total doses and taken doses
  const totalDosesToday = items.length;
  const takenDosesToday = items.filter(item => item.status === 'taken').length;

  return (
    <DashboardCard title="Medication Reminders">
      <div className="space-y-3">
        {items.length === 0 ? (
          <div className="text-center py-6">
            <div className="text-2xl mb-2">💊</div>
            <p className="text-gray-600 text-sm">No medications today</p>
          </div>
        ) : (
          <div className="max-h-48 overflow-y-auto custom-scrollbar pr-2 space-y-3">
            {items.map((i) => (
              <li
                key={`${i.cycleId}-${i.date}`}
                className="group relative bg-white p-3 rounded-xl border border-gray-100 shadow-sm hover:shadow-md transition-all duration-300"
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start flex-1">
                    <div className="flex-1 min-w-0">
                      <p className="font-semibold text-sm text-gray-900 mb-1">{i.name}</p>
                      <p className="text-xs text-gray-600">{i.dosage}</p>
                    </div>
                  </div>
                  <button
                    onClick={() => {
                      toggleTaken(i.cycleId, i.date, true);
                    }}
                    className={`relative inline-flex items-center px-4 py-2 rounded-xl text-xs font-medium transition-all duration-300 transform hover:scale-105 ${
                      animatingButtons.has(`${i.cycleId}-${i.date}`) 
                        ? 'bg-gradient-to-r from-green-500 to-green-600 text-white shadow-lg scale-95 animate-pulse' 
                        : 'bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white shadow-md hover:shadow-lg active:scale-95'
                    }`}
                    disabled={animatingButtons.has(`${i.cycleId}-${i.date}`)}
                  >
                    {animatingButtons.has(`${i.cycleId}-${i.date}`) ? (
                      <span className="flex items-center gap-2">
                        <svg className="w-4 h-4 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                        </svg>
                        Taking...
                      </span>
                    ) : (
                      <span className="flex items-center gap-2">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                        </svg>
                        Taken
                      </span>
                    )}
                  </button>
                </div>
                
                {/* Progress indicator */}
                <div className="mt-3 pt-3 border-t border-sky-100">
                  <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
                    <span>Status</span>
                    <span className="text-sky-600 font-medium">Ready to take</span>
                  </div>
                  <div className="w-full bg-sky-200 rounded-full h-2">
                    <div className="bg-gradient-to-r from-sky-400 to-blue-500 h-2 rounded-full transition-all duration-500" style={{ width: '100%' }}></div>
                  </div>
                </div>
              </li>
            ))}
          </div>
        )}
        
        <div className="p-3 bg-gradient-to-r from-sky-50 to-blue-50 rounded-xl border border-sky-100">
          <div className="flex items-center justify-between text-sky-700">
            <div className="flex items-center gap-2">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"></path>
              </svg>
              <span className="text-xs font-medium">Today's Medications</span>
            </div>
            <span className="text-xs font-medium">
              {takenDosesToday} / {totalDosesToday} doses
            </span>
          </div>
        </div>
      </div>
    </DashboardCard>
  );
}
