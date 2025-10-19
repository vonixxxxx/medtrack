import { useState, useMemo } from 'react';
import { motion } from 'framer-motion';

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

interface AnalyticsPanelProps {
  patients: Patient[];
  onGenerateGraph: () => void;
}

export const AnalyticsPanel = ({ patients, onGenerateGraph }: AnalyticsPanelProps) => {
  const [selectedMetric, setSelectedMetric] = useState('hba1cPercent');

  const analytics = useMemo(() => {
    if (patients.length === 0) {
      return {
        totalPatients: 0,
        averageAge: 0,
        genderDistribution: { male: 0, female: 0, other: 0 },
        averageHbA1c: 0,
        averageMES: 0,
        improvementRate: 0,
        topConditions: [],
        percentileChanges: []
      };
    }

    // Basic demographics
    const totalPatients = patients.length;
    const averageAge = patients.reduce((sum, p) => sum + p.age, 0) / totalPatients;
    
    const genderDistribution = patients.reduce((acc, p) => {
      acc[p.sex.toLowerCase()] = (acc[p.sex.toLowerCase()] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    // Medical metrics
    const averageHbA1c = patients.reduce((sum, p) => sum + p.hba1cPercent, 0) / totalPatients;
    const averageMES = patients.reduce((sum, p) => sum + p.mes, 0) / totalPatients;

    // Improvement rate (patients with negative change percent)
    const improvedPatients = patients.filter(p => p.changePercent < 0).length;
    const improvementRate = (improvedPatients / totalPatients) * 100;

    // Top conditions
    const conditionCounts = patients.reduce((acc, p) => {
      p.conditions.forEach(condition => {
        acc[condition] = (acc[condition] || 0) + 1;
      });
      return acc;
    }, {} as Record<string, number>);

    const topConditions = Object.entries(conditionCounts)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 5)
      .map(([condition, count]) => ({ condition, count }));

    // Percentile changes for selected metric
    const percentileChanges = patients
      .map(p => ({
        name: p.name,
        value: p[selectedMetric as keyof Patient] as number,
        change: p.changePercent
      }))
      .sort((a, b) => b.value - a.value);

    return {
      totalPatients,
      averageAge: Math.round(averageAge * 10) / 10,
      genderDistribution,
      averageHbA1c: Math.round(averageHbA1c * 10) / 10,
      averageMES: Math.round(averageMES * 100) / 100,
      improvementRate: Math.round(improvementRate * 10) / 10,
      topConditions,
      percentileChanges
    };
  }, [patients, selectedMetric]);

  const getMetricLabel = (metric: string) => {
    const labels: Record<string, string> = {
      'hba1cPercent': 'HbA1c (%)',
      'hba1cMmolMol': 'HbA1c (mmol/mol)',
      'mes': 'MES',
      'age': 'Age',
      'changePercent': 'Change (%)'
    };
    return labels[metric] || metric;
  };

  const getChangeColor = (change: number) => {
    if (change > 0) return 'text-red-400';
    if (change < 0) return 'text-green-400';
    return 'text-gray-400';
  };

  const getChangeIcon = (change: number) => {
    if (change > 0) return '↑';
    if (change < 0) return '↓';
    return '→';
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gray-900 rounded-3xl border border-gray-800 p-6"
    >
      <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center mb-6 gap-4">
        <div>
          <h3 className="text-lg font-semibold text-white mb-2">Analytics Summary</h3>
          <p className="text-sm text-gray-400">
            Overview of patient population and key metrics
          </p>
        </div>
        
        <div className="flex gap-3">
          <select
            value={selectedMetric}
            onChange={(e) => setSelectedMetric(e.target.value)}
            className="px-3 py-2 bg-gray-800 border border-gray-700 rounded-xl text-white text-sm focus:ring-2 focus:ring-white focus:border-white"
          >
            <option value="hba1cPercent">HbA1c (%)</option>
            <option value="hba1cMmolMol">HbA1c (mmol/mol)</option>
            <option value="mes">MES</option>
            <option value="age">Age</option>
            <option value="changePercent">Change (%)</option>
          </select>
          
          <button
            onClick={onGenerateGraph}
            className="px-4 py-2 bg-white text-black rounded-xl hover:bg-gray-200 transition-colors text-sm font-medium"
          >
            Generate Graph
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {/* Total Patients */}
        <div className="bg-gray-800 rounded-xl p-4">
          <div className="text-2xl font-bold text-white mb-1">
            {analytics.totalPatients}
          </div>
          <div className="text-sm text-gray-400">Total Patients</div>
        </div>

        {/* Average Age */}
        <div className="bg-gray-800 rounded-xl p-4">
          <div className="text-2xl font-bold text-white mb-1">
            {analytics.averageAge}
          </div>
          <div className="text-sm text-gray-400">Average Age</div>
        </div>

        {/* Average HbA1c */}
        <div className="bg-gray-800 rounded-xl p-4">
          <div className="text-2xl font-bold text-white mb-1">
            {analytics.averageHbA1c}%
          </div>
          <div className="text-sm text-gray-400">Avg HbA1c</div>
        </div>

        {/* Improvement Rate */}
        <div className="bg-gray-800 rounded-xl p-4">
          <div className="text-2xl font-bold text-white mb-1">
            {analytics.improvementRate}%
          </div>
          <div className="text-sm text-gray-400">Improvement Rate</div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Gender Distribution */}
        <div className="bg-gray-800 rounded-xl p-4">
          <h4 className="text-md font-medium text-white mb-3">Gender Distribution</h4>
          <div className="space-y-2">
            {Object.entries(analytics.genderDistribution).map(([gender, count]) => (
              <div key={gender} className="flex items-center justify-between">
                <span className="text-sm text-gray-300 capitalize">{gender}</span>
                <div className="flex items-center gap-2">
                  <div className="w-24 bg-gray-700 rounded-full h-2">
                    <div
                      className="bg-blue-500 h-2 rounded-full"
                      style={{ width: `${(count / analytics.totalPatients) * 100}%` }}
                    />
                  </div>
                  <span className="text-sm text-white w-8">{count}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Top Conditions */}
        <div className="bg-gray-800 rounded-xl p-4">
          <h4 className="text-md font-medium text-white mb-3">Top Conditions</h4>
          <div className="space-y-2">
            {analytics.topConditions.map(({ condition, count }, index) => (
              <div key={condition} className="flex items-center justify-between">
                <span className="text-sm text-gray-300 truncate">{condition}</span>
                <div className="flex items-center gap-2">
                  <div className="w-16 bg-gray-700 rounded-full h-2">
                    <div
                      className="bg-green-500 h-2 rounded-full"
                      style={{ width: `${(count / analytics.totalPatients) * 100}%` }}
                    />
                  </div>
                  <span className="text-sm text-white w-6">{count}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Percentile Changes Table */}
      {analytics.percentileChanges.length > 0 && (
        <div className="mt-6">
          <h4 className="text-md font-medium text-white mb-3">
            {getMetricLabel(selectedMetric)} - Percentile Changes
          </h4>
          <div className="bg-gray-800 rounded-xl overflow-hidden">
            <div className="max-h-48 overflow-y-auto">
              <table className="w-full">
                <thead className="bg-gray-700">
                  <tr>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-300">Patient</th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-300">
                      {getMetricLabel(selectedMetric)}
                    </th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-300">Change</th>
                  </tr>
                </thead>
                <tbody>
                  {analytics.percentileChanges.slice(0, 10).map((patient, index) => (
                    <tr key={patient.name} className="border-b border-gray-700 last:border-b-0">
                      <td className="px-4 py-2 text-sm text-gray-300">{patient.name}</td>
                      <td className="px-4 py-2 text-sm text-white">{patient.value}</td>
                      <td className="px-4 py-2 text-sm">
                        <div className={`flex items-center gap-1 ${getChangeColor(patient.change)}`}>
                          <span>{getChangeIcon(patient.change)}</span>
                          <span>{Math.abs(patient.change).toFixed(1)}%</span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </motion.div>
  );
};


