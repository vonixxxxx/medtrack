import { useState } from 'react';
import { motion } from 'framer-motion';
import { Bell, Clock, Link as LinkIcon, Calendar } from 'lucide-react';
import DashboardCard from '../DashboardCard';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Badge } from '../ui/badge';

export const AdvancedReminderSettings = ({ medication, onSave }) => {
  const [settings, setSettings] = useState({
    reminderEnabled: medication?.reminderEnabled ?? true,
    reminderTimes: medication?.reminderTimes ? JSON.parse(medication.reminderTimes) : ['08:00'],
    reminderDays: medication?.reminderDays ? JSON.parse(medication.reminderDays) : [0, 1, 2, 3, 4, 5, 6],
    intervalHours: medication?.intervalHours || null,
    reminderChainId: medication?.reminderChainId || null,
    weekendMode: medication?.weekendMode || false,
    weekendDelayDays: medication?.weekendDelayDays ? JSON.parse(medication.weekendDelayDays) : []
  });

  const [newTime, setNewTime] = useState('');
  const [reminderType, setReminderType] = useState('scheduled'); // scheduled, interval, chain

  const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

  const addTime = () => {
    if (newTime && !settings.reminderTimes.includes(newTime)) {
      setSettings({
        ...settings,
        reminderTimes: [...settings.reminderTimes, newTime].sort()
      });
      setNewTime('');
    }
  };

  const removeTime = (time) => {
    setSettings({
      ...settings,
      reminderTimes: settings.reminderTimes.filter(t => t !== time)
    });
  };

  const toggleDay = (day) => {
    setSettings({
      ...settings,
      reminderDays: settings.reminderDays.includes(day)
        ? settings.reminderDays.filter(d => d !== day)
        : [...settings.reminderDays, day].sort()
    });
  };

  const toggleWeekendDay = (day) => {
    setSettings({
      ...settings,
      weekendDelayDays: settings.weekendDelayDays.includes(day)
        ? settings.weekendDelayDays.filter(d => d !== day)
        : [...settings.weekendDelayDays, day].sort()
    });
  };

  const handleSave = () => {
    const data = {
      reminderEnabled: settings.reminderEnabled,
      reminderTimes: JSON.stringify(settings.reminderTimes),
      reminderDays: JSON.stringify(settings.reminderDays),
      intervalHours: reminderType === 'interval' ? settings.intervalHours : null,
      reminderChainId: reminderType === 'chain' ? settings.reminderChainId : null,
      weekendMode: settings.weekendMode,
      weekendDelayDays: settings.weekendMode ? JSON.stringify(settings.weekendDelayDays) : null
    };
    if (onSave) onSave(data);
  };

  return (
    <DashboardCard
      title="Advanced Reminder Settings"
      icon={<Bell size={20} />}
      variant="patient"
    >
      <div className="space-y-6">
        {/* Enable/Disable */}
        <div className="flex items-center justify-between">
          <div>
            <h4 className="font-semibold text-neutral-900">Enable Reminders</h4>
            <p className="text-sm text-neutral-600">Receive notifications for this medication</p>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={settings.reminderEnabled}
              onChange={(e) => setSettings({ ...settings, reminderEnabled: e.target.checked })}
              className="sr-only peer"
            />
            <div className="w-11 h-6 bg-neutral-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-neutral-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"></div>
          </label>
        </div>

        {settings.reminderEnabled && (
          <>
            {/* Reminder Type */}
            <div>
              <label className="block text-sm font-medium text-neutral-700 mb-2">
                Reminder Type
              </label>
              <div className="grid grid-cols-3 gap-2">
                <button
                  onClick={() => setReminderType('scheduled')}
                  className={`p-3 rounded-xl border-2 transition-all ${
                    reminderType === 'scheduled'
                      ? 'border-primary-500 bg-primary-50'
                      : 'border-neutral-200 hover:border-primary-300'
                  }`}
                >
                  <Calendar className="w-5 h-5 mx-auto mb-1" />
                  <span className="text-xs font-medium">Scheduled</span>
                </button>
                <button
                  onClick={() => setReminderType('interval')}
                  className={`p-3 rounded-xl border-2 transition-all ${
                    reminderType === 'interval'
                      ? 'border-primary-500 bg-primary-50'
                      : 'border-neutral-200 hover:border-primary-300'
                  }`}
                >
                  <Clock className="w-5 h-5 mx-auto mb-1" />
                  <span className="text-xs font-medium">Interval</span>
                </button>
                <button
                  onClick={() => setReminderType('chain')}
                  className={`p-3 rounded-xl border-2 transition-all ${
                    reminderType === 'chain'
                      ? 'border-primary-500 bg-primary-50'
                      : 'border-neutral-200 hover:border-primary-300'
                  }`}
                >
                  <LinkIcon className="w-5 h-5 mx-auto mb-1" />
                  <span className="text-xs font-medium">Chain</span>
                </button>
              </div>
            </div>

            {/* Scheduled Reminders */}
            {reminderType === 'scheduled' && (
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-neutral-700 mb-2">
                    Reminder Times
                  </label>
                  <div className="flex items-center gap-2 mb-2">
                    <Input
                      type="time"
                      value={newTime}
                      onChange={(e) => setNewTime(e.target.value)}
                      className="flex-1"
                    />
                    <Button onClick={addTime} variant="secondary" size="sm">
                      Add
                    </Button>
                  </div>
                  <div className="flex items-center gap-2 flex-wrap">
                    {settings.reminderTimes.map((time) => (
                      <Badge key={time} variant="default" className="text-sm">
                        {time}
                        <button
                          onClick={() => removeTime(time)}
                          className="ml-2 hover:text-error-600"
                        >
                          ×
                        </button>
                      </Badge>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-neutral-700 mb-2">
                    Days of Week
                  </label>
                  <div className="grid grid-cols-7 gap-2">
                    {dayNames.map((day, index) => (
                      <button
                        key={index}
                        onClick={() => toggleDay(index)}
                        className={`p-2 rounded-lg text-xs font-medium transition-all ${
                          settings.reminderDays.includes(index)
                            ? 'bg-primary-600 text-white'
                            : 'bg-neutral-100 text-neutral-700 hover:bg-neutral-200'
                        }`}
                      >
                        {day}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Interval Reminders */}
            {reminderType === 'interval' && (
              <div>
                <label className="block text-sm font-medium text-neutral-700 mb-2">
                  Remind Every (hours)
                </label>
                <Input
                  type="number"
                  value={settings.intervalHours || ''}
                  onChange={(e) => setSettings({ ...settings, intervalHours: parseInt(e.target.value) || null })}
                  placeholder="e.g., 8 for every 8 hours"
                  min="1"
                />
                <p className="text-xs text-neutral-600 mt-1">
                  Reminder will trigger every X hours after the first dose
                </p>
              </div>
            )}

            {/* Chain Reminders */}
            {reminderType === 'chain' && (
              <div>
                <label className="block text-sm font-medium text-neutral-700 mb-2">
                  Chain to Another Medication
                </label>
                <Input
                  type="text"
                  value={settings.reminderChainId || ''}
                  onChange={(e) => setSettings({ ...settings, reminderChainId: e.target.value })}
                  placeholder="Medication ID or name"
                />
                <p className="text-xs text-neutral-600 mt-1">
                  This medication will remind you X hours after taking the chained medication
                </p>
              </div>
            )}

            {/* Weekend Mode */}
            <div className="pt-4 border-t border-neutral-200">
              <div className="flex items-center justify-between mb-3">
                <div>
                  <h4 className="font-semibold text-neutral-900">Weekend Mode</h4>
                  <p className="text-sm text-neutral-600">Delay reminders on selected days</p>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.weekendMode}
                    onChange={(e) => setSettings({ ...settings, weekendMode: e.target.checked })}
                    className="sr-only peer"
                  />
                  <div className="w-11 h-6 bg-neutral-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-neutral-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"></div>
                </label>
              </div>
              {settings.weekendMode && (
                <div className="grid grid-cols-7 gap-2">
                  {dayNames.map((day, index) => (
                    <button
                      key={index}
                      onClick={() => toggleWeekendDay(index)}
                      className={`p-2 rounded-lg text-xs font-medium transition-all ${
                        settings.weekendDelayDays.includes(index)
                          ? 'bg-warning-500 text-white'
                          : 'bg-neutral-100 text-neutral-700 hover:bg-neutral-200'
                      }`}
                    >
                      {day}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </>
        )}

        {/* Save Button */}
        <Button onClick={handleSave} variant="primary" size="md" className="w-full">
          Save Reminder Settings
        </Button>
      </div>
    </DashboardCard>
  );
};



