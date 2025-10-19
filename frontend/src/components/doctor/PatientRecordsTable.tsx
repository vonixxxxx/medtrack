import { useState } from 'react';
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

interface PatientRecordsTableProps {
  patients: Patient[];
  onRefresh: () => void;
  onHbA1cAdjustment: () => void;
  onPatientSelect?: (patient: Patient) => void;
  selectedPatientId?: string;
}

export const PatientRecordsTable = ({ patients, onRefresh, onHbA1cAdjustment, onPatientSelect, selectedPatientId }: PatientRecordsTableProps) => {
  const [sortConfig, setSortConfig] = useState<{ key: keyof Patient; direction: 'asc' | 'desc' } | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedColumns, setSelectedColumns] = useState<Set<string>>(new Set([
    'name', 'age', 'sex', 'hba1cPercent', 'hba1cMmolMol', 'mes', 'conditions', 'lastVisit', 'changePercent'
  ]));

  const columns = [
    { key: 'name', label: 'Name', sortable: true },
    { key: 'age', label: 'Age', sortable: true },
    { key: 'sex', label: 'Sex', sortable: true },
    { key: 'hba1cPercent', label: 'HbA1c (%)', sortable: true },
    { key: 'hba1cMmolMol', label: 'HbA1c (mmol/mol)', sortable: true },
    { key: 'mes', label: 'MES', sortable: true },
    { key: 'conditions', label: 'Conditions', sortable: false },
    { key: 'lastVisit', label: 'Last Visit', sortable: true },
    { key: 'changePercent', label: 'Change (%)', sortable: true },
  ];

  const handleSort = (key: keyof Patient) => {
    let direction: 'asc' | 'desc' = 'asc';
    if (sortConfig && sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc';
    }
    setSortConfig({ key, direction });
  };

  const getSortedData = () => {
    if (!sortConfig) return patients;

    return [...patients].sort((a, b) => {
      const aValue = a[sortConfig.key];
      const bValue = b[sortConfig.key];

      if (aValue < bValue) {
        return sortConfig.direction === 'asc' ? -1 : 1;
      }
      if (aValue > bValue) {
        return sortConfig.direction === 'asc' ? 1 : -1;
      }
      return 0;
    });
  };

  const getFilteredData = () => {
    const sortedData = getSortedData();
    if (!searchTerm) return sortedData;

    return sortedData.filter(patient =>
      patient.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      patient.conditions.some(condition => 
        condition.toLowerCase().includes(searchTerm.toLowerCase())
      )
    );
  };

  const handleExportCSV = () => {
    const data = getFilteredData();
    const csvContent = [
      // Header
      columns.filter(col => selectedColumns.has(col.key)).map(col => col.label).join(','),
      // Data rows
      ...data.map(patient => 
        columns
          .filter(col => selectedColumns.has(col.key))
          .map(col => {
            const value = patient[col.key as keyof Patient];
            if (Array.isArray(value)) {
              return `"${value.join('; ')}"`;
            }
            return `"${value}"`;
          })
          .join(',')
      )
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'patient_records.csv';
    a.click();
    window.URL.revokeObjectURL(url);
  };

  const handleExportPDF = () => {
    // Placeholder for PDF export functionality
    console.log('PDF export functionality to be implemented');
  };

  const toggleColumn = (columnKey: string) => {
    const newSelected = new Set(selectedColumns);
    if (newSelected.has(columnKey)) {
      newSelected.delete(columnKey);
    } else {
      newSelected.add(columnKey);
    }
    setSelectedColumns(newSelected);
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

  const filteredData = getFilteredData();

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gray-900 rounded-3xl border border-gray-800 p-6"
    >
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6 gap-4">
        <div>
          <h3 className="text-xl font-semibold text-white mb-2">Patient Records</h3>
          <p className="text-sm text-gray-400">
            {filteredData.length} of {patients.length} patients
          </p>
        </div>
        
        <div className="flex flex-wrap gap-2">
          <button
            onClick={onRefresh}
            className="px-4 py-2 bg-gray-800 text-white rounded-xl hover:bg-gray-700 transition-colors"
          >
            Refresh
          </button>
          <button
            onClick={onHbA1cAdjustment}
            className="px-4 py-2 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition-colors"
          >
            HbA1c Calculator
          </button>
          <button
            onClick={handleExportCSV}
            className="px-4 py-2 bg-green-600 text-white rounded-xl hover:bg-green-700 transition-colors"
          >
            Export CSV
          </button>
          <button
            onClick={handleExportPDF}
            className="px-4 py-2 bg-red-600 text-white rounded-xl hover:bg-red-700 transition-colors"
          >
            Export PDF
          </button>
        </div>
      </div>

      {/* Search and Column Selection */}
      <div className="flex flex-col lg:flex-row gap-4 mb-6">
        <div className="flex-1">
          <input
            type="text"
            placeholder="Search patients or conditions..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-xl text-white placeholder-gray-400 focus:ring-2 focus:ring-white focus:border-white"
          />
        </div>
        
        <div className="flex flex-wrap gap-2">
          {columns.map(column => (
            <label key={column.key} className="flex items-center gap-2 text-sm text-gray-300">
              <input
                type="checkbox"
                checked={selectedColumns.has(column.key)}
                onChange={() => toggleColumn(column.key)}
                className="rounded border-gray-600 bg-gray-800 text-white focus:ring-white"
              />
              {column.label}
            </label>
          ))}
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-gray-700">
              {columns
                .filter(col => selectedColumns.has(col.key))
                .map(column => (
                  <th
                    key={column.key}
                    className={`px-4 py-3 text-left text-sm font-medium text-gray-300 ${
                      column.sortable ? 'cursor-pointer hover:text-white' : ''
                    }`}
                    onClick={column.sortable ? () => handleSort(column.key as keyof Patient) : undefined}
                  >
                    <div className="flex items-center gap-2">
                      {column.label}
                      {column.sortable && sortConfig?.key === column.key && (
                        <span className="text-xs">
                          {sortConfig.direction === 'asc' ? '↑' : '↓'}
                        </span>
                      )}
                    </div>
                  </th>
                ))}
            </tr>
          </thead>
          <tbody>
            {filteredData.map((patient, index) => (
              <tr 
                key={patient.id} 
                className={`border-b border-gray-800 hover:bg-gray-800/50 cursor-pointer ${
                  selectedPatientId === patient.id ? 'bg-blue-900/20 border-blue-500' : ''
                }`}
                onClick={() => onPatientSelect?.(patient)}
              >
                {columns
                  .filter(col => selectedColumns.has(col.key))
                  .map(column => (
                    <td key={column.key} className="px-4 py-3 text-sm text-gray-300">
                      {column.key === 'conditions' ? (
                        <div className="flex flex-wrap gap-1">
                          {patient.conditions && patient.conditions.length > 0 ? (
                            patient.conditions.map((condition, idx) => (
                              <span
                                key={idx}
                                className="px-2 py-1 bg-gray-700 text-xs rounded-full"
                              >
                                {condition}
                              </span>
                            ))
                          ) : (
                            <span className="text-gray-500 text-xs">No conditions</span>
                          )}
                        </div>
                      ) : column.key === 'changePercent' ? (
                        patient.changePercent !== null ? (
                          <div className={`flex items-center gap-1 ${getChangeColor(patient.changePercent)}`}>
                            <span>{getChangeIcon(patient.changePercent)}</span>
                            <span>{Math.abs(patient.changePercent).toFixed(1)}%</span>
                          </div>
                        ) : (
                          <span className="text-gray-500 text-xs">No data</span>
                        )
                      ) : (
                        patient[column.key as keyof Patient] !== null ? 
                          patient[column.key as keyof Patient] : 
                          <span className="text-gray-500 text-xs">No data</span>
                      )}
                    </td>
                  ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {filteredData.length === 0 && (
        <div className="text-center py-8 text-gray-400">
          No patients found matching your criteria.
        </div>
      )}
    </motion.div>
  );
};


