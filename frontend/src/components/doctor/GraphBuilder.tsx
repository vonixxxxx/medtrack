import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';

interface Patient {
  id: number;
  name: string;
  age: number;
  sex: string;
  hba1cPercent: number;
  hba1cMmolMol: number;
  mes: number;
  conditions: string[];
  lastVisit: string;
  changePercent: number;
  ethnicity?: string;
}

interface GraphBuilderProps {
  patients: Patient[];
  filters: any;
}

export const GraphBuilder = ({ patients, filters }: GraphBuilderProps) => {
  const [graphType, setGraphType] = useState<'line' | 'bar' | 'pie'>('line');
  const [xAxis, setXAxis] = useState('age');
  const [yAxis, setYAxis] = useState('hba1cPercent');
  const [selectedPatients, setSelectedPatients] = useState<number[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);

  // React to filter changes - auto-set axis based on selected metric
  useEffect(() => {
    if (filters?.metric && filters.metric !== 'all') {
      switch (filters.metric) {
        case 'hba1c':
          setYAxis('hba1cPercent');
          break;
        case 'bmi':
          setYAxis('baseline_bmi');
          break;
        case 'weight':
          setYAxis('baseline_weight');
          break;
        default:
          break;
      }
    }
  }, [filters?.metric]);

  const xAxisOptions = [
    { value: 'age', label: 'Age' },
    { value: 'hba1cPercent', label: 'HbA1c (%)' },
    { value: 'hba1cMmolMol', label: 'HbA1c (mmol/mol)' },
    { value: 'mes', label: 'MES' },
    { value: 'changePercent', label: 'Change (%)' }
  ];

  const yAxisOptions = [
    { value: 'hba1cPercent', label: 'HbA1c (%)' },
    { value: 'hba1cMmolMol', label: 'HbA1c (mmol/mol)' },
    { value: 'mes', label: 'MES' },
    { value: 'changePercent', label: 'Change (%)' },
    { value: 'age', label: 'Age' }
  ];

  const generateGraphData = () => {
    let data = patients;
    
    // Apply patient selection filter
    if (selectedPatients.length > 0) {
      data = data.filter(patient => selectedPatients.includes(patient.id));
    }

    if (graphType === 'pie') {
      // For pie charts, group by a categorical field
      const grouped = data.reduce((acc, patient) => {
        const key = patient[xAxis as keyof Patient] as string;
        acc[key] = (acc[key] || 0) + 1;
        return acc;
      }, {} as Record<string, number>);

      return Object.entries(grouped).map(([name, value]) => ({ name, value }));
    }

    // For line and bar charts, return individual data points
    return data.map(patient => ({
      name: patient.name,
      [xAxis]: patient[xAxis as keyof Patient],
      [yAxis]: patient[yAxis as keyof Patient],
      age: patient.age,
      hba1cPercent: patient.hba1cPercent,
      hba1cMmolMol: patient.hba1cMmolMol,
      mes: patient.mes,
      changePercent: patient.changePercent
    }));
  };

  const handleGenerateGraph = async () => {
    setIsGenerating(true);
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));
    setIsGenerating(false);
  };

  const togglePatientSelection = (patientId: number) => {
    setSelectedPatients(prev => 
      prev.includes(patientId) 
        ? prev.filter(id => id !== patientId)
        : [...prev, patientId]
    );
  };

  const selectAllPatients = () => {
    setSelectedPatients(patients.map(p => p.id));
  };

  const clearSelection = () => {
    setSelectedPatients([]);
  };

  const graphData = generateGraphData();
  const COLORS = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#06b6d4'];

  const renderGraph = () => {
    if (graphData.length === 0) {
      return (
        <div className="flex flex-col items-center justify-center h-64 text-gray-400">
          <div className="text-4xl mb-2">📈</div>
          <p className="text-lg font-bold text-gray-700 mb-1">No Data Available</p>
          <p className="text-sm text-gray-600">
            {patients.length === 0 
              ? 'No patients match the current filters. Try adjusting your filters.'
              : 'Select patients or adjust graph settings to see data.'}
          </p>
        </div>
      );
    }

    switch (graphType) {
      case 'line':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={graphData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey={xAxis} stroke="#6b7280" />
              <YAxis stroke="#6b7280" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#ffffff', 
                  border: '2px solid #dbeafe',
                  borderRadius: '12px',
                  color: '#1f2937',
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                }} 
              />
              <Line 
                type="monotone" 
                dataKey={yAxis} 
                stroke="#2563eb" 
                strokeWidth={2}
                dot={{ fill: '#2563eb', strokeWidth: 2, r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        );

      case 'bar':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={graphData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey={xAxis} stroke="#6b7280" />
              <YAxis stroke="#6b7280" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#ffffff', 
                  border: '2px solid #dbeafe',
                  borderRadius: '12px',
                  color: '#1f2937',
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                }} 
              />
              <Bar dataKey={yAxis} fill="#2563eb" />
            </BarChart>
          </ResponsiveContainer>
        );

      case 'pie':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={graphData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {graphData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#ffffff', 
                  border: '2px solid #dbeafe',
                  borderRadius: '12px',
                  color: '#1f2937',
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                }} 
              />
            </PieChart>
          </ResponsiveContainer>
        );

      default:
        return null;
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ y: -2 }}
      className="bg-gradient-to-br from-white to-blue-50/30 rounded-2xl border border-blue-100 hover:border-blue-200 shadow-lg shadow-blue-600/5 hover:shadow-xl hover:shadow-blue-600/20 transition-all p-6"
    >
      <div className="flex flex-col lg:flex-row gap-6">
        {/* Controls Panel */}
        <div className="lg:w-80 space-y-4">
          <h3 className="text-xl font-bold text-gray-900 mb-4">Graph Builder</h3>
          
          {/* Graph Type Selection */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Graph Type
            </label>
            <div className="grid grid-cols-3 gap-2">
              {(['line', 'bar', 'pie'] as const).map(type => (
                <motion.button
                  key={type}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => setGraphType(type)}
                  className={`px-3 py-2 text-sm font-medium rounded-xl transition-all ${
                    graphType === type
                      ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/25'
                      : 'bg-white text-gray-700 border-2 border-gray-200 hover:border-blue-300'
                  }`}
                >
                  {type.charAt(0).toUpperCase() + type.slice(1)}
                </motion.button>
              ))}
            </div>
          </div>

          {/* X-Axis Selection */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              X-Axis
            </label>
            <select
              value={xAxis}
              onChange={(e) => setXAxis(e.target.value)}
              className="w-full px-4 py-2.5 bg-white border-2 border-gray-200 rounded-xl text-gray-900 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all hover:border-blue-300"
            >
              {xAxisOptions.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          {/* Y-Axis Selection (not for pie charts) */}
          {graphType !== 'pie' && (
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Y-Axis
              </label>
              <select
                value={yAxis}
                onChange={(e) => setYAxis(e.target.value)}
                className="w-full px-4 py-2.5 bg-white border-2 border-gray-200 rounded-xl text-gray-900 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all hover:border-blue-300"
              >
                {yAxisOptions.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Patient Selection */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="block text-sm font-semibold text-gray-700">
                Patients to Include
              </label>
              <div className="flex gap-2">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={selectAllPatients}
                  className="text-xs text-blue-600 hover:text-blue-700 font-medium"
                >
                  Select All
                </motion.button>
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={clearSelection}
                  className="text-xs text-gray-600 hover:text-gray-700 font-medium"
                >
                  Clear
                </motion.button>
              </div>
            </div>
            <div className="max-h-32 overflow-y-auto space-y-2 p-2 bg-white rounded-xl border border-gray-200">
              {patients.map(patient => (
                <label key={patient.id} className="flex items-center gap-2 text-sm text-gray-700 cursor-pointer hover:bg-blue-50 p-1 rounded-lg transition-colors">
                  <input
                    type="checkbox"
                    checked={selectedPatients.includes(patient.id)}
                    onChange={() => togglePatientSelection(patient.id)}
                    className="w-4 h-4 rounded border-gray-300 bg-white text-blue-600 focus:ring-blue-500 focus:ring-2"
                  />
                  <span className="font-medium">{patient.name || `Patient ${patient.id}`}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Generate Button */}
          <motion.button
            whileHover={{ scale: 1.02, y: -2 }}
            whileTap={{ scale: 0.98 }}
            onClick={handleGenerateGraph}
            disabled={isGenerating}
            className="w-full bg-blue-600 text-white py-2.5 px-4 rounded-xl font-semibold hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-blue-600/25 hover:shadow-xl hover:shadow-blue-600/40 disabled:hover:shadow-lg disabled:hover:shadow-blue-600/25"
          >
            {isGenerating ? 'Generating...' : 'Generate Graph'}
          </motion.button>
        </div>

        {/* Graph Display */}
        <div className="flex-1">
          <div className="bg-white rounded-xl p-4 border border-blue-100 shadow-sm">
            {renderGraph()}
          </div>
        </div>
      </div>
    </motion.div>
  );
};


