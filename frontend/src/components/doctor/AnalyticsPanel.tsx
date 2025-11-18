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
    
    // Filter out null values before calculating averages
    const patientsWithAge = patients.filter(p => p.age !== null && p.age !== undefined);
    const averageAge = patientsWithAge.length > 0 
      ? patientsWithAge.reduce((sum, p) => sum + (p.age || 0), 0) / patientsWithAge.length 
      : 0;
    
    const genderDistribution = patients.reduce((acc, p) => {
      if (p.sex) {
        acc[p.sex.toLowerCase()] = (acc[p.sex.toLowerCase()] || 0) + 1;
      }
      return acc;
    }, {} as Record<string, number>);

    // Medical metrics - filter nulls before averaging
    const patientsWithHbA1c = patients.filter(p => p.hba1cPercent !== null && p.hba1cPercent !== undefined);
    const averageHbA1c = patientsWithHbA1c.length > 0
      ? patientsWithHbA1c.reduce((sum, p) => sum + (p.hba1cPercent || 0), 0) / patientsWithHbA1c.length
      : 0;
    
    const patientsWithMES = patients.filter(p => p.mes !== null && p.mes !== undefined);
    const averageMES = patientsWithMES.length > 0
      ? patientsWithMES.reduce((sum, p) => sum + (p.mes || 0), 0) / patientsWithMES.length
      : 0;

    // Improvement rate (patients with negative change percent)
    const improvedPatients = patients.filter(p => p.changePercent && p.changePercent < 0).length;
    const improvementRate = totalPatients > 0 ? (improvedPatients / totalPatients) * 100 : 0;

    // Top conditions
    const conditionCounts = patients.reduce((acc, p) => {
      if (p.conditions && Array.isArray(p.conditions)) {
        p.conditions.forEach(condition => {
          acc[condition] = (acc[condition] || 0) + 1;
        });
      }
      return acc;
    }, {} as Record<string, number>);

    const topConditions = Object.entries(conditionCounts)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 5)
      .map(([condition, count]) => ({ condition, count }));

    // Percentile changes for selected metric
    const percentileChanges = patients
      .map(p => ({
        name: p.name || 'Unknown',
        value: (p[selectedMetric as keyof Patient] as number) || 0,
        change: p.changePercent || 0
      }))
      .filter(p => p.value > 0) // Only include patients with valid values
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
    if (change > 0) return 'text-red-600';
    if (change < 0) return 'text-green-600';
    return 'text-gray-600';
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
      whileHover={{ y: -2 }}
      className="bg-gradient-to-br from-white to-blue-50/30 rounded-2xl border border-blue-100 hover:border-blue-200 shadow-lg shadow-blue-600/5 hover:shadow-xl hover:shadow-blue-600/20 transition-all p-6"
    >
      <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center mb-6 gap-4">
        <div>
          <h3 className="text-xl font-bold text-gray-900 mb-2">Analytics Summary</h3>
          <p className="text-sm text-gray-600">
            Overview of patient population and key metrics
          </p>
        </div>
        
        <div className="flex gap-3">
          <select
            value={selectedMetric}
            onChange={(e) => setSelectedMetric(e.target.value)}
            className="px-4 py-2.5 bg-white border-2 border-gray-200 rounded-xl text-gray-900 text-sm font-medium focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all outline-none hover:border-blue-300"
          >
            <option value="hba1cPercent">HbA1c (%)</option>
            <option value="hba1cMmolMol">HbA1c (mmol/mol)</option>
            <option value="mes">MES</option>
            <option value="age">Age</option>
            <option value="changePercent">Change (%)</option>
          </select>
          
          <motion.button
            whileHover={{ scale: 1.05, y: -2 }}
            whileTap={{ scale: 0.95 }}
            onClick={onGenerateGraph}
            className="px-4 py-2.5 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition-all text-sm font-semibold shadow-lg shadow-blue-600/25 hover:shadow-xl hover:shadow-blue-600/40"
          >
            Generate Graph
          </motion.button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {/* Total Patients */}
        <motion.div 
          whileHover={{ y: -4, scale: 1.02 }}
          className="bg-gradient-to-br from-blue-50 to-white rounded-xl p-4 border border-blue-100 shadow-sm hover:shadow-md transition-all"
        >
          <div className="text-2xl font-bold text-gray-900 mb-1">
            {analytics.totalPatients}
          </div>
          <div className="text-sm text-gray-600 font-medium">Total Patients</div>
        </motion.div>

        {/* Average Age */}
        <motion.div 
          whileHover={{ y: -4, scale: 1.02 }}
          className="bg-gradient-to-br from-blue-50 to-white rounded-xl p-4 border border-blue-100 shadow-sm hover:shadow-md transition-all"
        >
          <div className="text-2xl font-bold text-gray-900 mb-1">
            {analytics.averageAge}
          </div>
          <div className="text-sm text-gray-600 font-medium">Average Age</div>
        </motion.div>

        {/* Average HbA1c */}
        <motion.div 
          whileHover={{ y: -4, scale: 1.02 }}
          className="bg-gradient-to-br from-blue-50 to-white rounded-xl p-4 border border-blue-100 shadow-sm hover:shadow-md transition-all"
        >
          <div className="text-2xl font-bold text-gray-900 mb-1">
            {analytics.averageHbA1c}%
          </div>
          <div className="text-sm text-gray-600 font-medium">Avg HbA1c</div>
        </motion.div>

        {/* Improvement Rate */}
        <motion.div 
          whileHover={{ y: -4, scale: 1.02 }}
          className="bg-gradient-to-br from-blue-50 to-white rounded-xl p-4 border border-blue-100 shadow-sm hover:shadow-md transition-all"
        >
          <div className="text-2xl font-bold text-gray-900 mb-1">
            {analytics.improvementRate}%
          </div>
          <div className="text-sm text-gray-600 font-medium">Improvement Rate</div>
        </motion.div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Gender Distribution */}
        <div className="bg-white rounded-xl p-5 border border-blue-100 shadow-sm">
          <h4 className="text-lg font-bold text-gray-900 mb-4">Gender Distribution</h4>
          <div className="space-y-3">
            {Object.entries(analytics.genderDistribution).map(([gender, count]) => (
              <div key={gender} className="flex items-center justify-between">
                <span className="text-sm text-gray-700 font-medium capitalize">{gender}</span>
                <div className="flex items-center gap-3">
                  <div className="w-24 bg-gray-200 rounded-full h-2.5">
                    <div
                      className="bg-gradient-to-r from-blue-500 to-blue-600 h-2.5 rounded-full transition-all"
                      style={{ width: `${(count / analytics.totalPatients) * 100}%` }}
                    />
                  </div>
                  <span className="text-sm text-gray-900 font-semibold w-8">{count}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Top Conditions */}
        <div className="bg-white rounded-xl p-5 border border-blue-100 shadow-sm">
          <h4 className="text-lg font-bold text-gray-900 mb-4">Top Conditions</h4>
          <div className="space-y-3">
            {analytics.topConditions.map(({ condition, count }, index) => (
              <div key={condition} className="flex items-center justify-between">
                <span className="text-sm text-gray-700 font-medium truncate">{condition}</span>
                <div className="flex items-center gap-3">
                  <div className="w-16 bg-gray-200 rounded-full h-2.5">
                    <div
                      className="bg-gradient-to-r from-blue-500 to-blue-600 h-2.5 rounded-full transition-all"
                      style={{ width: `${(count / analytics.totalPatients) * 100}%` }}
                    />
                  </div>
                  <span className="text-sm text-gray-900 font-semibold w-6">{count}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Percentile Changes Table */}
      {analytics.percentileChanges.length > 0 && (
        <div className="mt-6">
          <h4 className="text-lg font-bold text-gray-900 mb-4">
            {getMetricLabel(selectedMetric)} - Percentile Changes
          </h4>
          <div className="bg-white rounded-xl overflow-hidden border border-blue-100 shadow-sm">
            <div className="max-h-48 overflow-y-auto">
              <table className="w-full">
                <thead className="bg-blue-50 border-b border-blue-100">
                  <tr>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">Patient</th>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                      {getMetricLabel(selectedMetric)}
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">Change</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {analytics.percentileChanges.slice(0, 10).map((patient, index) => (
                    <tr key={patient.name} className="hover:bg-blue-50/50 transition-colors">
                      <td className="px-4 py-3 text-sm text-gray-900 font-medium">{patient.name}</td>
                      <td className="px-4 py-3 text-sm text-gray-700">{patient.value}</td>
                      <td className="px-4 py-3 text-sm">
                        <div className={`flex items-center gap-1 font-medium ${patient.change > 0 ? 'text-red-600' : patient.change < 0 ? 'text-green-600' : 'text-gray-600'}`}>
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


