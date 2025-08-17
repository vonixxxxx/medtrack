import { useState } from 'react';
import DashboardCard from './DashboardCard';
import api, { poster, fetcher } from '../api';
import { useQueryClient, useQuery } from '@tanstack/react-query';

export default function AddMetricCard() {
  const [form, setForm] = useState({ 
    cycleId: '',
    date: new Date().toISOString().split('T')[0], 
    notes: '',
    metricValues: {} // Object to hold all metric values
  });
  const [useMultipleValues, setUseMultipleValues] = useState({});
  const queryClient = useQueryClient();
  const { data: cycles = [] } = useQuery({ queryKey: ['cycles'], queryFn: () => fetcher('/cycles') });

  // Get due metrics check for selected cycle
  const { data: metricsCheck, isLoading: isCheckingMetrics } = useQuery({
    queryKey: ['metricsDue', form.cycleId],
    queryFn: () => fetcher(`/cycles/${form.cycleId}/metrics-due`),
    enabled: !!form.cycleId,
  });

  // Get monitored metrics for selected cycle
  const selectedCycle = cycles.find(c => c.id.toString() === form.cycleId);
  const monitoredMetrics = metricsCheck?.dueMetrics || [];
  const allConfiguredMetrics = metricsCheck?.allConfiguredMetrics || [];
  const canLogMetrics = metricsCheck?.canLog || false;

  const handleChange = (e) => {
    if (e.target.name === 'cycleId') {
      // Reset metric values when cycle changes
      setForm({ 
        ...form, 
        [e.target.name]: e.target.value,
        metricValues: {},
      });
      setUseMultipleValues({});
    } else {
      setForm({ ...form, [e.target.name]: e.target.value });
    }
  };

  const handleMetricValueChange = (metricType, index, value) => {
    const currentValues = form.metricValues[metricType] || [''];
    const newValues = [...currentValues];
    newValues[index] = value;
    setForm({
      ...form,
      metricValues: {
        ...form.metricValues,
        [metricType]: newValues
      }
    });
  };

  const addValueField = (metricType) => {
    const currentValues = form.metricValues[metricType] || [''];
    setForm({
      ...form,
      metricValues: {
        ...form.metricValues,
        [metricType]: [...currentValues, '']
      }
    });
  };

  const removeValueField = (metricType, index) => {
    const currentValues = form.metricValues[metricType] || [''];
    if (currentValues.length > 1) {
      const newValues = currentValues.filter((_, i) => i !== index);
      setForm({
        ...form,
        metricValues: {
          ...form.metricValues,
          [metricType]: newValues
        }
      });
    }
  };

  const toggleMultipleValues = (metricType) => {
    setUseMultipleValues({
      ...useMultipleValues,
      [metricType]: !useMultipleValues[metricType]
    });
    
    if (useMultipleValues[metricType]) {
      // If disabling, keep only first value
      const currentValues = form.metricValues[metricType] || [''];
      setForm({
        ...form,
        metricValues: {
          ...form.metricValues,
          [metricType]: [currentValues[0] || '']
        }
      });
    }
  };

  const calculateMean = (values) => {
    const numericValues = values
      .map(v => parseFloat(v))
      .filter(v => !isNaN(v));
    
    if (numericValues.length === 0) return null;
    return numericValues.reduce((sum, val) => sum + val, 0) / numericValues.length;
  };

  const handleSave = async () => {
    try {
      if (!form.cycleId) return alert('Please select a medication cycle first');
      if (monitoredMetrics.length === 0) return alert('No metrics configured to monitor for this cycle');
      
      // Check if all monitored metrics have values
      const missingMetrics = monitoredMetrics.filter(metric => {
        const values = form.metricValues[metric.type] || [''];
        return !values.some(v => v.trim());
      });
      
      if (missingMetrics.length > 0) {
        return alert(`Please provide values for all monitored metrics: ${missingMetrics.map(m => m.type).join(', ')}`);
      }

      // Save all metrics
      for (const metric of monitoredMetrics) {
        const values = form.metricValues[metric.type] || [''];
        const isNumericType = [
          'Weight', 'Body Index', 'Hip Circumference', 'Heart Rate', 'Temperature', 
          'Blood Sugar', 'Sleep Hours', 'Pain Level', 'Systolic Blood Pressure', 
          'Diastolic Blood Pressure', 'Height', 'BMI', 'Waist Circumference',
          'Body Fat Percentage', 'Muscle Mass', 'Water Intake', 'Steps', 'Distance',
          'Calories Burned', 'Oxygen Saturation', 'Respiratory Rate'
        ].includes(metric.type);
        const hasMultipleValues = useMultipleValues[metric.type] && values.length > 1;
        
        let finalValue;
        let notes = form.notes;
        
        if (hasMultipleValues) {
          if (isNumericType) {
            const mean = calculateMean(values);
            if (mean === null) {
              return alert(`At least one valid numeric value is required for ${metric.type}`);
            }
            finalValue = mean;
            notes += `${notes ? ' | ' : ''}Multiple values: ${values.filter(v => v.trim()).join(', ')} (Mean: ${mean.toFixed(2)})`;
          } else {
            finalValue = values.filter(v => v.trim()).join(', ');
            notes += `${notes ? ' | ' : ''}Multiple values: ${finalValue}`;
          }
        } else {
          finalValue = values[0];
          if (isNumericType && isNaN(parseFloat(finalValue))) {
            return alert(`Numeric value required for ${metric.type}`);
          }
        }

        await poster(`/cycles/${form.cycleId}/metrics`, {
          kind: metric.type,
          valueFloat: isNumericType ? (isNaN(parseFloat(finalValue)) ? null : parseFloat(finalValue)) : null,
          valueText: isNumericType ? null : finalValue,
          date: form.date,
          notes: notes,
        });
      }
      
      // Reset form
      setForm({ 
        cycleId: form.cycleId, // Keep selected cycle
        date: new Date().toISOString().split('T')[0], 
        notes: '',
        metricValues: {}
      });
      setUseMultipleValues({});
      
      alert('All metrics saved successfully!');
      
      // Invalidate all relevant queries to refresh UI
      queryClient.invalidateQueries({ queryKey: ['cycles'] });
      queryClient.invalidateQueries({ queryKey: ['metrics'] });
      queryClient.invalidateQueries({ queryKey: ['metricsDue'] });
      queryClient.invalidateQueries({ queryKey: ['metricReminders'] });
    } catch (error) {
      console.error('Error saving metrics:', error);
      alert('Failed to save metrics. Please try again.');
    }
  };

  return (
    <DashboardCard title="Update Metrics">
      <div className="space-y-4">
        <select
          name="cycleId"
          value={form.cycleId}
          onChange={handleChange}
          className="w-full rounded-xl border px-3 py-2"
        >
          <option value="">Select Medication Cycle</option>
          {cycles.map((cycle) => (
            <option key={cycle.id} value={cycle.id}>
              {cycle.name} - {cycle.dosage}
            </option>
          ))}
        </select>

        {/* Show metrics status */}
        {form.cycleId && (
          <>
            {isCheckingMetrics ? (
              <div className="text-center py-6 text-gray-500">
                <div className="mb-2">⏳</div>
                <p className="text-sm">Checking metrics status...</p>
              </div>
            ) : !canLogMetrics ? (
              <div className="text-center py-6">
                {allConfiguredMetrics.length === 0 ? (
                  <div className="text-gray-500">
                    <div className="mb-2">📊</div>
                    <p className="text-sm">No metrics configured for monitoring</p>
                    <p className="text-xs mt-1">Configure metrics when creating the medication cycle</p>
                  </div>
                ) : (
                  <div className="text-amber-600">
                    <div className="mb-2">⏰</div>
                    <p className="text-sm font-medium">No metrics due for logging today</p>
                    <p className="text-xs mt-1">{metricsCheck?.message}</p>
                    <div className="mt-3 p-2 bg-amber-50 rounded-lg">
                      <p className="text-xs text-amber-700 font-medium mb-1">Configured metrics:</p>
                      <div className="flex flex-wrap gap-1">
                        {allConfiguredMetrics.map(metric => (
                          <span key={metric.type} className="bg-amber-200 text-amber-800 px-2 py-1 rounded-full text-xs">
                            {metric.type} ({metric.frequency})
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <>
                <div className="bg-blue-50 p-3 rounded-xl">
                  <p className="text-sm font-medium text-blue-800 mb-1">Required Metrics to Update</p>
                  <p className="text-xs text-blue-600">All metrics must be logged together:</p>
                  <div className="flex flex-wrap gap-1 mt-2">
                    {monitoredMetrics.map(metric => (
                      <span key={metric.type} className="bg-blue-200 text-blue-800 px-2 py-1 rounded-full text-xs">
                        {metric.type} ({metric.frequency})
                      </span>
                    ))}
                  </div>
                </div>

                {/* Metric Value Inputs */}
                {monitoredMetrics.map(metric => (
                  <div key={metric.type} className="border rounded-xl p-3 space-y-2">
                    <div className="flex justify-between items-center">
                      <label className="block text-sm font-medium text-gray-700">
                        {metric.type}
                      </label>
                      <label className="flex items-center gap-1 text-xs">
                        <input
                          type="checkbox"
                          checked={useMultipleValues[metric.type] || false}
                          onChange={() => toggleMultipleValues(metric.type)}
                          className="rounded"
                        />
                        Multiple values
                      </label>
                    </div>
                    
                    {useMultipleValues[metric.type] && (form.metricValues[metric.type]?.length > 1) && (
                      <p className="text-xs text-gray-500">
                        Mean: {calculateMean(form.metricValues[metric.type] || [''])?.toFixed(2) || 'N/A'}
                      </p>
                    )}
                    
                    {(form.metricValues[metric.type] || ['']).map((value, index) => (
                      <div key={index} className="flex gap-2">
                        <input
                          value={value}
                          onChange={(e) => handleMetricValueChange(metric.type, index, e.target.value)}
                          placeholder={`${metric.type} value${index > 0 ? ` ${index + 1}` : ''}`}
                          className="flex-1 rounded-lg border px-3 py-2 text-sm"
                        />
                        {useMultipleValues[metric.type] && (
                          <>
                            {(form.metricValues[metric.type]?.length > 1) && (
                              <button
                                type="button"
                                onClick={() => removeValueField(metric.type, index)}
                                className="px-2 py-1 text-red-600 hover:text-red-800 text-sm"
                              >
                                ✕
                              </button>
                            )}
                            {index === (form.metricValues[metric.type]?.length || 1) - 1 && (
                              <button
                                type="button"
                                onClick={() => addValueField(metric.type)}
                                className="px-2 py-1 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 text-sm"
                              >
                                +
                              </button>
                            )}
                          </>
                        )}
                      </div>
                    ))}
                  </div>
                ))}

                <input
                  type="date"
                  name="date"
                  value={form.date}
                  onChange={handleChange}
                  max={new Date().toISOString().split('T')[0]}
                  className="w-full rounded-xl border px-3 py-2"
                />
                <textarea
                  name="notes"
                  value={form.notes}
                  onChange={handleChange}
                  placeholder="General notes for all metrics"
                  className="w-full rounded-xl border px-3 py-2"
                  rows="2"
                />
              </>
            )}
          </>
        )}

        <div className="flex gap-3">
          <button 
            className="btn-secondary flex-1" 
            onClick={() => {
              setForm({ 
                cycleId: '',
                date: new Date().toISOString().split('T')[0], 
                notes: '',
                metricValues: {}
              });
              setUseMultipleValues({});
            }}
          >
            Cancel
          </button>
          <button 
            className="btn-primary flex-1" 
            onClick={handleSave}
            disabled={!form.cycleId || !canLogMetrics || monitoredMetrics.length === 0}
          >
            {canLogMetrics ? 'Save All Metrics' : 'Metrics Not Due'}
          </button>
        </div>
      </div>
    </DashboardCard>
  );
}
