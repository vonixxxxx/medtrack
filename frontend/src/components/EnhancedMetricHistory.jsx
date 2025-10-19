import { useState, useEffect } from "react";
import { Activity, Plus, TrendingUp, Calendar } from "lucide-react";
import { MedTrackCard } from "./MedTrackCard";
import { NeonButton } from "./NeonButton";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from "recharts";
import { motion } from "framer-motion";
import api from "../api";

export const EnhancedMetricHistory = ({ onAddMetric }) => {
  const [medications, setMedications] = useState([]);
  const [selectedMedication, setSelectedMedication] = useState(null);
  const [selectedMetric, setSelectedMetric] = useState('Blood Glucose');
  const [metricData, setMetricData] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Available metrics for selection
  const availableMetrics = [
    'Blood Glucose', 'Blood Pressure', 'Heart Rate', 'Weight', 'Temperature',
    'Pain Level', 'Sleep Quality', 'Mood', 'Energy Level', 'Blood Sugar',
    'Cholesterol', 'Blood Oxygen', 'BMI', 'Blood Pressure (Systolic)',
    'Blood Pressure (Diastolic)', 'Waist Circumference', 'Hip Circumference',
    'Body Fat Percentage', 'Muscle Mass', 'Bone Density', 'Vitamin D Level',
    'Iron Level', 'Thyroid Function', 'Kidney Function', 'Liver Function',
    'Blood Count', 'Inflammation Markers', 'Allergy Symptoms', 'Digestive Health',
    'Mental Health', 'Cognitive Function', 'Physical Activity', 'Exercise Duration',
    'Exercise Intensity', 'Steps Count', 'Calories Burned', 'Water Intake',
    'Alcohol Consumption', 'Caffeine Intake', 'Smoking Status', 'Stress Level',
    'Anxiety Level', 'Depression Score', 'Quality of Life', 'Medication Adherence',
    'Drug Interactions', 'Allergic Reactions', 'Emergency Symptoms'
  ];

  // Fetch medications on mount
  useEffect(() => {
    fetchMedications();
  }, []);

  // Fetch metric data when medication or metric changes
  useEffect(() => {
    if (selectedMedication && selectedMetric) {
      fetchMetricData();
    }
  }, [selectedMedication, selectedMetric]);

  const fetchMedications = async () => {
    try {
      const response = await api.get('meds/user');
      const medications = response.data.medications || [];
      setMedications(medications);
      
      // Auto-select first medication if available
      if (medications.length > 0) {
        setSelectedMedication(medications[0]);
      }
    } catch (err) {
      console.error('Error fetching medications:', err);
      setError('Failed to load medications');
    }
  };

  const fetchMetricData = async () => {
    if (!selectedMedication || !selectedMetric) return;
    
    setIsLoading(true);
    try {
      // Fetch real metric data from the API
      const response = await api.get('meds/user/metrics');
      const allMetrics = response.data.metrics || [];
      
      // Filter metrics for the selected medication and metric type
      const filteredMetrics = allMetrics.filter(metric => 
        metric.medicationId === selectedMedication.id && 
        metric.metric === selectedMetric
      );
      
      if (filteredMetrics.length === 0) {
        // If no real data, show empty state
        setMetricData([]);
        setError(null);
      } else {
        // Transform real data for the chart
        const chartData = transformMetricDataForChart(filteredMetrics);
        setMetricData(chartData);
        setError(null);
      }
    } catch (err) {
      console.error('Error fetching metric data:', err);
      setError('Failed to load metric data');
      setMetricData([]);
    } finally {
      setIsLoading(false);
    }
  };

  const transformMetricDataForChart = (metrics) => {
    // Group metrics by date and calculate daily averages
    const dailyData = {};
    
    metrics.forEach(metric => {
      const date = new Date(metric.timestamp).toDateString();
      if (!dailyData[date]) {
        dailyData[date] = {
          date: new Date(metric.timestamp).toLocaleDateString('en-US', { weekday: 'short' }),
          values: [],
          fullDate: metric.timestamp.split('T')[0]
        };
      }
      dailyData[date].values.push(metric.value);
    });
    
    // Calculate daily averages and create chart data
    const chartData = Object.values(dailyData).map(day => ({
      date: day.date,
      value: day.values.reduce((sum, val) => sum + val, 0) / day.values.length,
      fullDate: day.fullDate
    })).sort((a, b) => new Date(a.fullDate) - new Date(b.fullDate));
    
    return chartData;
  };

  const generateSampleMetricData = (metric) => {
    // Generate realistic sample data based on metric type
    const baseValue = getBaseValueForMetric(metric);
    const variation = getVariationForMetric(metric);
    
    const data = [];
    const today = new Date();
    
    for (let i = 6; i >= 0; i--) {
      const date = new Date(today);
      date.setDate(date.getDate() - i);
      
      const value = baseValue + (Math.random() - 0.5) * variation;
      const roundedValue = Math.round(value * 10) / 10;
      
      data.push({
        date: date.toLocaleDateString('en-US', { weekday: 'short' }),
        value: roundedValue,
        fullDate: date.toISOString().split('T')[0]
      });
    }
    
    return data;
  };

  const getBaseValueForMetric = (metric) => {
    const baseValues = {
      'Blood Glucose': 120,
      'Blood Pressure': 120,
      'Heart Rate': 75,
      'Weight': 70,
      'Temperature': 98.6,
      'Pain Level': 3,
      'Sleep Quality': 7,
      'Mood': 6,
      'Energy Level': 6,
      'BMI': 22,
      'Cholesterol': 180,
      'Blood Oxygen': 98,
      'Steps Count': 8000,
      'Calories Burned': 2000,
      'Water Intake': 8
    };
    return baseValues[metric] || 50;
  };

  const getVariationForMetric = (metric) => {
    const variations = {
      'Blood Glucose': 30,
      'Blood Pressure': 20,
      'Heart Rate': 15,
      'Weight': 2,
      'Temperature': 1,
      'Pain Level': 2,
      'Sleep Quality': 2,
      'Mood': 2,
      'Energy Level': 2,
      'BMI': 1,
      'Cholesterol': 40,
      'Blood Oxygen': 3,
      'Steps Count': 2000,
      'Calories Burned': 500,
      'Water Intake': 3
    };
    return variations[metric] || 10;
  };

  const getUnitForMetric = (metric) => {
    const units = {
      'Blood Glucose': 'mg/dL',
      'Blood Pressure': 'mmHg',
      'Heart Rate': 'bpm',
      'Weight': 'kg',
      'Temperature': '°F',
      'Pain Level': '/10',
      'Sleep Quality': '/10',
      'Mood': '/10',
      'Energy Level': '/10',
      'BMI': 'kg/m²',
      'Cholesterol': 'mg/dL',
      'Blood Oxygen': '%',
      'Steps Count': 'steps',
      'Calories Burned': 'cal',
      'Water Intake': 'glasses'
    };
    return units[metric] || '';
  };

  const calculateAverage = () => {
    if (metricData.length === 0) return 0;
    const sum = metricData.reduce((acc, item) => acc + item.value, 0);
    return Math.round((sum / metricData.length) * 10) / 10;
  };

  const getTrend = () => {
    if (metricData.length < 2) return 'stable';
    const first = metricData[0].value;
    const last = metricData[metricData.length - 1].value;
    const diff = last - first;
    const threshold = getVariationForMetric(selectedMetric) * 0.3;
    
    if (diff > threshold) return 'increasing';
    if (diff < -threshold) return 'decreasing';
    return 'stable';
  };

  const getTrendColor = () => {
    const trend = getTrend();
    switch (trend) {
      case 'increasing': return 'text-green-500';
      case 'decreasing': return 'text-red-500';
      default: return 'text-muted-foreground';
    }
  };

  const getTrendIcon = () => {
    const trend = getTrend();
    switch (trend) {
      case 'increasing': return '↗';
      case 'decreasing': return '↘';
      default: return '→';
    }
  };

  return (
    <MedTrackCard>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-foreground" />
          <h3 className="text-lg font-semibold">Health Metrics</h3>
        </div>
        <NeonButton onClick={onAddMetric} size="sm">
          Log Metrics
        </NeonButton>
      </div>

      {/* Medication and Metric Selection */}
      <div className="mb-4 space-y-3">
        <div>
          <label className="text-sm font-medium text-muted-foreground mb-2 block">
            Select Medication
          </label>
          <div className="flex flex-wrap gap-2">
            {medications.map((med) => (
              <button
                key={med.id}
                onClick={() => setSelectedMedication(med)}
                className={`px-3 py-1.5 rounded-lg text-sm transition-all ${
                  selectedMedication?.id === med.id
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-secondary text-secondary-foreground hover:bg-secondary/80'
                }`}
              >
                {med.medication_name || med.name || med.generic_name}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="text-sm font-medium text-muted-foreground mb-2 block">
            Select Metric
          </label>
          <div className="flex flex-wrap gap-2 max-h-32 overflow-y-auto">
            {availableMetrics.slice(0, 12).map((metric) => (
              <button
                key={metric}
                onClick={() => setSelectedMetric(metric)}
                className={`px-3 py-1.5 rounded-lg text-sm transition-all ${
                  selectedMetric === metric
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-secondary text-secondary-foreground hover:bg-secondary/80'
                }`}
              >
                {metric}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Chart */}
      {isLoading ? (
        <div className="flex items-center justify-center h-48">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-2"></div>
            <p className="text-sm text-muted-foreground">Loading data...</p>
          </div>
        </div>
      ) : error ? (
        <div className="text-center py-8">
          <p className="text-muted-foreground">{error}</p>
          <button 
            onClick={fetchMetricData}
            className="mt-2 text-sm text-blue-600 hover:text-blue-700"
          >
            Try again
          </button>
        </div>
      ) : metricData.length === 0 ? (
        <div className="text-center py-8">
          <Activity className="w-12 h-12 text-muted-foreground/50 mx-auto mb-3" />
          <p className="text-muted-foreground">No data available</p>
          <p className="text-sm text-muted-foreground/70">Log some metrics to see trends</p>
        </div>
      ) : (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <div className="h-48 mb-4">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={metricData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis 
                  dataKey="date" 
                  stroke="hsl(var(--muted-foreground))"
                  style={{ fontSize: "11px" }}
                />
                <YAxis 
                  stroke="hsl(var(--muted-foreground))"
                  style={{ fontSize: "11px" }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "6px",
                    color: "hsl(var(--foreground))",
                    fontSize: "12px"
                  }}
                  formatter={(value) => [`${value} ${getUnitForMetric(selectedMetric)}`, selectedMetric]}
                />
                <Line 
                  type="monotone" 
                  dataKey="value" 
                  stroke="hsl(var(--foreground))" 
                  strokeWidth={2}
                  dot={{ fill: "hsl(var(--foreground))", strokeWidth: 2, r: 3 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Summary Stats */}
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-4">
              <span className="text-muted-foreground">
                Average: <span className="font-semibold text-foreground">
                  {calculateAverage()} {getUnitForMetric(selectedMetric)}
                </span>
              </span>
              <span className={`flex items-center gap-1 ${getTrendColor()}`}>
                <span>{getTrendIcon()}</span>
                <span className="capitalize">{getTrend()}</span>
              </span>
            </div>
            <div className="text-muted-foreground">
              Last 7 days
            </div>
          </div>
        </motion.div>
      )}
    </MedTrackCard>
  );
};