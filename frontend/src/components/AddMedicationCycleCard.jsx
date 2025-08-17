import { useState } from 'react';
import DashboardCard from './DashboardCard';
import { poster } from '../api';
import { useQueryClient } from '@tanstack/react-query';

export default function AddMedicationCycleCard() {
  const [form, setForm] = useState({ 
    name: '', 
    dosage: '', 
    startDate: '', 
    endDate: '', 
    dosesPerDay: 1, 
    sendNow: true,
    metricsToMonitor: []
  });
  const [newMetric, setNewMetric] = useState({ type: '', frequency: 'daily' });
  const queryClient = useQueryClient();

  const metricTypes = [
    'Weight', 'Height', 'Systolic Blood Pressure', 'Diastolic Blood Pressure', 
    'Heart Rate', 'Body Index', 'Hip Circumference', 'Waist Circumference', 
    'Temperature', 'Blood Sugar', 'Sleep Hours', 'Pain Level', 'BMI',
    'Body Fat Percentage', 'Muscle Mass', 'Water Intake', 'Steps', 'Distance',
    'Calories Burned', 'Oxygen Saturation', 'Respiratory Rate'
  ];

  const frequencyOptions = [
    { value: 'daily', label: 'Daily' },
    { value: 'weekly', label: 'Weekly' },
    { value: 'biweekly', label: 'Bi-weekly' },
    { value: 'monthly', label: 'Monthly' }
  ];

  const handleChange = (e) => setForm({ ...form, [e.target.name]: e.target.value });

  const addMetricToMonitor = () => {
    if (newMetric.type && !form.metricsToMonitor.some(m => m.type === newMetric.type)) {
      setForm({
        ...form,
        metricsToMonitor: [...form.metricsToMonitor, { ...newMetric, id: Date.now() }]
      });
      setNewMetric({ type: '', frequency: 'daily' });
    }
  };

  const removeMetricFromMonitor = (metricId) => {
    setForm({
      ...form,
      metricsToMonitor: form.metricsToMonitor.filter(m => m.id !== metricId)
    });
  };

  const handleSave = async () => {
    try {
      // Validation
      if (!form.name.trim()) {
        return alert('Medication name is required');
      }
      if (!form.dosage.trim()) {
        return alert('Dosage is required');
      }
      
      const payload = {
        name: form.name.trim(),
        dosage: form.dosage.trim(),
        startDate: form.startDate || null, // Use null if empty
        endDate: form.endDate || null,
        frequencyDays: 1,
        dosesPerDay: parseInt(form.dosesPerDay, 10),
        metricsToMonitor: form.metricsToMonitor
      };
      
      const response = await poster('/cycles', payload);
      
      setForm({ 
        name: '', 
        dosage: '', 
        startDate: '', 
        endDate: '', 
        dosesPerDay: 1, 
        sendNow: true,
        metricsToMonitor: []
      });
      alert('Medication cycle saved successfully!');
      queryClient.invalidateQueries();
    } catch (error) {
      console.error('Failed to save medication cycle:', error);
      alert('Failed to save medication cycle. Please try again.');
    }
  };

  return (
    <DashboardCard title="Add Medication Cycle">
      <div className="space-y-3">
        <input
          name="name"
          value={form.name}
          onChange={handleChange}
          placeholder="Medication Name *"
          className="w-full rounded-xl border px-3 py-2"
          required
        />
        <input
          name="dosage"
          value={form.dosage}
          onChange={handleChange}
          placeholder="Dosage (e.g., 500mg, 2 pills) *"
          className="w-full rounded-xl border px-3 py-2"
          required
        />
        
        <div className="space-y-2">
          <label className="text-sm font-medium text-gray-700">Start Date (optional)</label>
          <input
            type="date"
            name="startDate"
            value={form.startDate}
            onChange={handleChange}
            className="w-full rounded-xl border px-3 py-2"
            placeholder="Leave blank to start today"
          />
        </div>
        
        <div className="space-y-2">
          <label className="text-sm font-medium text-gray-700">End Date (optional)</label>
          <input
            type="date"
            name="endDate"
            value={form.endDate}
            onChange={handleChange}
            className="w-full rounded-xl border px-3 py-2"
            placeholder="Leave blank for ongoing treatment"
          />
        </div>
        <label className="text-sm">Times per day</label>
        <select
          name="dosesPerDay"
          value={form.dosesPerDay}
          onChange={handleChange}
          className="w-full rounded-xl border px-3 py-2"
        >
          {[1, 2, 3, 4].map((n) => (
            <option key={n} value={n}>{n}</option>
          ))}
        </select>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={form.sendNow}
            onChange={(e) => setForm({ ...form, sendNow: e.target.checked })}
            className="rounded"
          />
          Send day-zero notification
        </label>

        {/* Metrics to Monitor Section */}
        <div className="border-t pt-4 mt-4">
          <h4 className="font-medium text-sm text-gray-700 mb-3">Metrics to Monitor</h4>
          
          {/* Add New Metric */}
          <div className="grid grid-cols-3 gap-2 mb-3">
            <select
              value={newMetric.type}
              onChange={(e) => setNewMetric({ ...newMetric, type: e.target.value })}
              className="rounded-xl border px-3 py-2 text-sm"
            >
              <option value="">Select Metric</option>
              {metricTypes
                .filter(type => !form.metricsToMonitor.some(m => m.type === type))
                .map(type => (
                  <option key={type} value={type}>{type}</option>
                ))}
            </select>
            
            <select
              value={newMetric.frequency}
              onChange={(e) => setNewMetric({ ...newMetric, frequency: e.target.value })}
              className="rounded-xl border px-3 py-2 text-sm"
            >
              {frequencyOptions.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
            
            <button
              type="button"
              onClick={addMetricToMonitor}
              disabled={!newMetric.type}
              className="bg-blue-600 text-white px-3 py-2 rounded-xl hover:bg-blue-700 disabled:opacity-50 text-sm"
            >
              Add
            </button>
          </div>

          {/* Display Added Metrics */}
          {form.metricsToMonitor.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs text-gray-500">Selected metrics to monitor:</p>
              {form.metricsToMonitor.map(metric => (
                <div
                  key={metric.id}
                  className="flex items-center justify-between bg-blue-50 px-3 py-2 rounded-lg text-sm"
                >
                  <span className="flex items-center space-x-2">
                    <span className="font-medium">{metric.type}</span>
                    <span className="text-gray-500">•</span>
                    <span className="text-gray-600">{frequencyOptions.find(f => f.value === metric.frequency)?.label}</span>
                  </span>
                  <button
                    type="button"
                    onClick={() => removeMetricFromMonitor(metric.id)}
                    className="text-red-600 hover:text-red-800"
                  >
                    ✕
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="flex gap-3">
          <button 
            className="btn-secondary flex-1" 
            onClick={() => setForm({ name: '', dosage: '', startDate: '', endDate: '', dosesPerDay: 1, sendNow: true, metricsToMonitor: [] })}
          >
            Cancel
          </button>
          <button 
            className="btn-primary flex-1" 
            onClick={handleSave}
            disabled={!form.name.trim() || !form.dosage.trim()}
          >
            Save
          </button>
        </div>
      </div>
    </DashboardCard>
  );
}
