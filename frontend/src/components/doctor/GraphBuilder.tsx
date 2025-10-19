import { useState } from 'react';
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
        <div className="flex items-center justify-center h-64 text-gray-400">
          No data available for the selected criteria
        </div>
      );
    }

    switch (graphType) {
      case 'line':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={graphData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey={xAxis} stroke="#9ca3af" />
              <YAxis stroke="#9ca3af" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1f2937', 
                  border: '1px solid #374151',
                  borderRadius: '8px',
                  color: '#f9fafb'
                }} 
              />
              <Line 
                type="monotone" 
                dataKey={yAxis} 
                stroke="#3b82f6" 
                strokeWidth={2}
                dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        );

      case 'bar':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={graphData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey={xAxis} stroke="#9ca3af" />
              <YAxis stroke="#9ca3af" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1f2937', 
                  border: '1px solid #374151',
                  borderRadius: '8px',
                  color: '#f9fafb'
                }} 
              />
              <Bar dataKey={yAxis} fill="#3b82f6" />
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
                  backgroundColor: '#1f2937', 
                  border: '1px solid #374151',
                  borderRadius: '8px',
                  color: '#f9fafb'
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
      className="bg-gray-900 rounded-3xl border border-gray-800 p-6"
    >
      <div className="flex flex-col lg:flex-row gap-6">
        {/* Controls Panel */}
        <div className="lg:w-80 space-y-4">
          <h3 className="text-lg font-semibold text-white mb-4">Graph Builder</h3>
          
          {/* Graph Type Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Graph Type
            </label>
            <div className="grid grid-cols-3 gap-2">
              {(['line', 'bar', 'pie'] as const).map(type => (
                <button
                  key={type}
                  onClick={() => setGraphType(type)}
                  className={`px-3 py-2 text-sm rounded-lg transition-colors ${
                    graphType === type
                      ? 'bg-white text-black'
                      : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                  }`}
                >
                  {type.charAt(0).toUpperCase() + type.slice(1)}
                </button>
              ))}
            </div>
          </div>

          {/* X-Axis Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              X-Axis
            </label>
            <select
              value={xAxis}
              onChange={(e) => setXAxis(e.target.value)}
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-xl text-white focus:ring-2 focus:ring-white focus:border-white"
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
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Y-Axis
              </label>
              <select
                value={yAxis}
                onChange={(e) => setYAxis(e.target.value)}
                className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-xl text-white focus:ring-2 focus:ring-white focus:border-white"
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
              <label className="block text-sm font-medium text-gray-300">
                Patients to Include
              </label>
              <div className="flex gap-2">
                <button
                  onClick={selectAllPatients}
                  className="text-xs text-blue-400 hover:text-blue-300"
                >
                  Select All
                </button>
                <button
                  onClick={clearSelection}
                  className="text-xs text-gray-400 hover:text-gray-300"
                >
                  Clear
                </button>
              </div>
            </div>
            <div className="max-h-32 overflow-y-auto space-y-1">
              {patients.map(patient => (
                <label key={patient.id} className="flex items-center gap-2 text-sm text-gray-300">
                  <input
                    type="checkbox"
                    checked={selectedPatients.includes(patient.id)}
                    onChange={() => togglePatientSelection(patient.id)}
                    className="rounded border-gray-600 bg-gray-800 text-white focus:ring-white"
                  />
                  {patient.name}
                </label>
              ))}
            </div>
          </div>

          {/* Generate Button */}
          <button
            onClick={handleGenerateGraph}
            disabled={isGenerating}
            className="w-full bg-white text-black py-2 px-4 rounded-xl font-medium hover:bg-gray-200 focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-gray-900 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isGenerating ? 'Generating...' : 'Generate Graph'}
          </button>
        </div>

        {/* Graph Display */}
        <div className="flex-1">
          <div className="bg-gray-800 rounded-xl p-4">
            {renderGraph()}
          </div>
        </div>
      </div>
    </motion.div>
  );
};


