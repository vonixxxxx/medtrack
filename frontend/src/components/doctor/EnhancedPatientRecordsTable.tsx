import { useState } from 'react';
import { motion } from 'framer-motion';

interface Patient {
  id: string;
  userId: string;
  name: string;
  email: string;
  age: number | null;
  sex: string | null;
  ethnic_group: string | null;
  location: string | null;
  postcode: string | null;
  nhs_number: string | null;
  mrn: string | null;
  height: number | null;
  baseline_weight: number | null;
  baseline_bmi: number | null;
  baseline_weight_date: string | null;
  ascvd: boolean | null;
  htn: boolean | null;
  dyslipidaemia: boolean | null;
  osa: boolean | null;
  sleep_studies: boolean | null;
  cpap: boolean | null;
  t2dm: boolean | null;
  prediabetes: boolean | null;
  diabetes_type: string | null;
  baseline_hba1c: number | null;
  baseline_hba1c_date: string | null;
  baseline_tc: number | null;
  baseline_hdl: number | null;
  baseline_ldl: number | null;
  baseline_tg: number | null;
  baseline_lipid_date: string | null;
  lipid_lowering_treatment: string | null;
  antihypertensive_medications: string | null;
  asthma: boolean | null;
  hypertension: boolean | null;
  ischaemic_heart_disease: boolean | null;
  heart_failure: boolean | null;
  cerebrovascular_disease: boolean | null;
  pulmonary_hypertension: boolean | null;
  dvt: boolean | null;
  pe: boolean | null;
  gord: boolean | null;
  ckd: boolean | null;
  kidney_stones: boolean | null;
  masld: boolean | null;
  infertility: boolean | null;
  pcos: boolean | null;
  anxiety: boolean | null;
  depression: boolean | null;
  bipolar_disorder: boolean | null;
  emotional_eating: boolean | null;
  schizoaffective_disorder: boolean | null;
  oa_knee: boolean | null;
  oa_hip: boolean | null;
  limited_mobility: boolean | null;
  lymphoedema: boolean | null;
  thyroid_disorder: boolean | null;
  iih: boolean | null;
  epilepsy: boolean | null;
  functional_neurological_disorder: boolean | null;
  cancer: boolean | null;
  bariatric_gastric_band: boolean | null;
  bariatric_sleeve: boolean | null;
  bariatric_bypass: boolean | null;
  bariatric_balloon: boolean | null;
  diagnoses_coded_in_scr: string | null;
  total_qualifying_comorbidities: number | null;
  mes: number | null;
  notes: string | null;
  criteria_for_wegovy: string | null;
  conditions: string[];
  lastVisit: string | null;
  changePercent: number | null;
  ethnicity?: string | null;
}

interface EnhancedPatientRecordsTableProps {
  patients: Patient[];
  onRefresh: () => void;
  onHbA1cAdjustment: () => void;
  onPatientSelect?: (patient: Patient) => void;
  selectedPatientId?: string;
}

export const EnhancedPatientRecordsTable = ({ 
  patients, 
  onRefresh, 
  onHbA1cAdjustment, 
  onPatientSelect, 
  selectedPatientId 
}: EnhancedPatientRecordsTableProps) => {
  const [sortConfig, setSortConfig] = useState<{ key: keyof Patient; direction: 'asc' | 'desc' } | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [showFilters, setShowFilters] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(10);
  
  // Filter states
  const [filters, setFilters] = useState({
    sex: '',
    ethnic_group: '',
    ascvd: '',
    t2dm: '',
    htn: '',
    dyslipidaemia: '',
    bmi_range: '',
    hba1c_range: '',
    age_range: '',
    conditions: [] as string[]
  });

  const availableColumns = [
    { key: 'name', label: 'Name', sortable: true, category: 'basic' },
    { key: 'email', label: 'Email', sortable: true, category: 'basic' },
    { key: 'age', label: 'Age', sortable: true, category: 'basic' },
    { key: 'sex', label: 'Sex', sortable: true, category: 'basic' },
    { key: 'ethnic_group', label: 'Ethnic Group', sortable: true, category: 'basic' },
    { key: 'location', label: 'Location', sortable: true, category: 'basic' },
    { key: 'postcode', label: 'Postcode', sortable: true, category: 'basic' },
    { key: 'nhs_number', label: 'NHS Number', sortable: true, category: 'basic' },
    { key: 'mrn', label: 'MRN', sortable: true, category: 'basic' },
    { key: 'height', label: 'Height (cm)', sortable: true, category: 'measurements' },
    { key: 'baseline_weight', label: 'Weight (kg)', sortable: true, category: 'measurements' },
    { key: 'baseline_bmi', label: 'BMI', sortable: true, category: 'measurements' },
    { key: 'baseline_weight_date', label: 'Weight Date', sortable: true, category: 'measurements' },
    { key: 'ascvd', label: 'ASCVD', sortable: true, category: 'conditions' },
    { key: 'htn', label: 'HTN', sortable: true, category: 'conditions' },
    { key: 'dyslipidaemia', label: 'Dyslipidaemia', sortable: true, category: 'conditions' },
    { key: 'osa', label: 'OSA', sortable: true, category: 'conditions' },
    { key: 'sleep_studies', label: 'Sleep Studies', sortable: true, category: 'conditions' },
    { key: 'cpap', label: 'CPAP', sortable: true, category: 'conditions' },
    { key: 't2dm', label: 'T2DM', sortable: true, category: 'conditions' },
    { key: 'prediabetes', label: 'Prediabetes', sortable: true, category: 'conditions' },
    { key: 'diabetes_type', label: 'Diabetes Type', sortable: true, category: 'diabetes' },
    { key: 'baseline_hba1c', label: 'HbA1c (%)', sortable: true, category: 'diabetes' },
    { key: 'baseline_hba1c_date', label: 'HbA1c Date', sortable: true, category: 'diabetes' },
    { key: 'baseline_tc', label: 'Total Cholesterol', sortable: true, category: 'lipids' },
    { key: 'baseline_hdl', label: 'HDL', sortable: true, category: 'lipids' },
    { key: 'baseline_ldl', label: 'LDL', sortable: true, category: 'lipids' },
    { key: 'baseline_tg', label: 'Triglycerides', sortable: true, category: 'lipids' },
    { key: 'baseline_lipid_date', label: 'Lipid Date', sortable: true, category: 'lipids' },
    { key: 'lipid_lowering_treatment', label: 'Lipid Treatment', sortable: true, category: 'medications' },
    { key: 'antihypertensive_medications', label: 'Antihypertensive Meds', sortable: true, category: 'medications' },
    { key: 'asthma', label: 'Asthma', sortable: true, category: 'conditions' },
    { key: 'hypertension', label: 'Hypertension', sortable: true, category: 'conditions' },
    { key: 'ischaemic_heart_disease', label: 'IHD', sortable: true, category: 'conditions' },
    { key: 'heart_failure', label: 'Heart Failure', sortable: true, category: 'conditions' },
    { key: 'cerebrovascular_disease', label: 'CVD', sortable: true, category: 'conditions' },
    { key: 'pulmonary_hypertension', label: 'Pulmonary HTN', sortable: true, category: 'conditions' },
    { key: 'dvt', label: 'DVT', sortable: true, category: 'conditions' },
    { key: 'pe', label: 'PE', sortable: true, category: 'conditions' },
    { key: 'gord', label: 'GORD', sortable: true, category: 'conditions' },
    { key: 'ckd', label: 'CKD', sortable: true, category: 'conditions' },
    { key: 'kidney_stones', label: 'Kidney Stones', sortable: true, category: 'conditions' },
    { key: 'masld', label: 'MASLD', sortable: true, category: 'conditions' },
    { key: 'infertility', label: 'Infertility', sortable: true, category: 'conditions' },
    { key: 'pcos', label: 'PCOS', sortable: true, category: 'conditions' },
    { key: 'anxiety', label: 'Anxiety', sortable: true, category: 'conditions' },
    { key: 'depression', label: 'Depression', sortable: true, category: 'conditions' },
    { key: 'bipolar_disorder', label: 'Bipolar', sortable: true, category: 'conditions' },
    { key: 'emotional_eating', label: 'Emotional Eating', sortable: true, category: 'conditions' },
    { key: 'schizoaffective_disorder', label: 'Schizoaffective', sortable: true, category: 'conditions' },
    { key: 'oa_knee', label: 'OA Knee', sortable: true, category: 'conditions' },
    { key: 'oa_hip', label: 'OA Hip', sortable: true, category: 'conditions' },
    { key: 'limited_mobility', label: 'Limited Mobility', sortable: true, category: 'conditions' },
    { key: 'lymphoedema', label: 'Lymphoedema', sortable: true, category: 'conditions' },
    { key: 'thyroid_disorder', label: 'Thyroid Disorder', sortable: true, category: 'conditions' },
    { key: 'iih', label: 'IIH', sortable: true, category: 'conditions' },
    { key: 'epilepsy', label: 'Epilepsy', sortable: true, category: 'conditions' },
    { key: 'functional_neurological_disorder', label: 'FND', sortable: true, category: 'conditions' },
    { key: 'cancer', label: 'Cancer', sortable: true, category: 'conditions' },
    { key: 'bariatric_gastric_band', label: 'Gastric Band', sortable: true, category: 'bariatric' },
    { key: 'bariatric_sleeve', label: 'Sleeve', sortable: true, category: 'bariatric' },
    { key: 'bariatric_bypass', label: 'Bypass', sortable: true, category: 'bariatric' },
    { key: 'bariatric_balloon', label: 'Balloon', sortable: true, category: 'bariatric' },
    { key: 'total_qualifying_comorbidities', label: 'Comorbidities', sortable: true, category: 'clinical' },
    { key: 'mes', label: 'MES', sortable: true, category: 'clinical' },
    { key: 'notes', label: 'Notes', sortable: false, category: 'clinical' },
    { key: 'criteria_for_wegovy', label: 'Wegovy Criteria', sortable: false, category: 'clinical' },
    { key: 'conditions', label: 'Conditions', sortable: false, category: 'clinical' },
    { key: 'lastVisit', label: 'Last Visit', sortable: true, category: 'clinical' },
    { key: 'changePercent', label: 'Change (%)', sortable: true, category: 'clinical' }
  ];

  const [selectedColumns, setSelectedColumns] = useState<Set<string>>(new Set([
    'name', 'age', 'sex', 'ethnic_group', 'baseline_bmi', 'baseline_hba1c', 't2dm', 'ascvd', 'htn', 'mes', 'conditions'
  ]));

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

      if (aValue === null || aValue === undefined) return 1;
      if (bValue === null || bValue === undefined) return -1;

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
    
    return sortedData.filter(patient => {
      // Search term filter
      if (searchTerm) {
        const searchLower = searchTerm.toLowerCase();
        const searchableFields = [
          patient.name, patient.email, patient.nhs_number, patient.mrn,
          patient.postcode, patient.location, patient.ethnic_group
        ];
        
        if (!searchableFields.some(field => 
          field && field.toString().toLowerCase().includes(searchLower)
        )) {
          return false;
        }
      }

      // Individual filters
      if (filters.sex && patient.sex !== filters.sex) return false;
      if (filters.ethnic_group && patient.ethnic_group !== filters.ethnic_group) return false;
      if (filters.ascvd && patient.ascvd !== (filters.ascvd === 'true')) return false;
      if (filters.t2dm && patient.t2dm !== (filters.t2dm === 'true')) return false;
      if (filters.htn && patient.htn !== (filters.htn === 'true')) return false;
      if (filters.dyslipidaemia && patient.dyslipidaemia !== (filters.dyslipidaemia === 'true')) return false;

      // BMI range filter
      if (filters.bmi_range && patient.baseline_bmi) {
        const bmi = patient.baseline_bmi;
        switch (filters.bmi_range) {
          case 'underweight': if (bmi >= 18.5) return false; break;
          case 'normal': if (bmi < 18.5 || bmi >= 25) return false; break;
          case 'overweight': if (bmi < 25 || bmi >= 30) return false; break;
          case 'obese': if (bmi < 30) return false; break;
        }
      }

      // HbA1c range filter
      if (filters.hba1c_range && patient.baseline_hba1c) {
        const hba1c = patient.baseline_hba1c;
        switch (filters.hba1c_range) {
          case 'normal': if (hba1c >= 5.7) return false; break;
          case 'prediabetes': if (hba1c < 5.7 || hba1c >= 6.5) return false; break;
          case 'diabetes': if (hba1c < 6.5) return false; break;
        }
      }

      // Age range filter
      if (filters.age_range && patient.age) {
        const age = patient.age;
        switch (filters.age_range) {
          case '18-30': if (age < 18 || age > 30) return false; break;
          case '31-45': if (age < 31 || age > 45) return false; break;
          case '46-60': if (age < 46 || age > 60) return false; break;
          case '60+': if (age < 60) return false; break;
        }
      }

      return true;
    });
  };

  const getPaginatedData = () => {
    const filteredData = getFilteredData();
    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = startIndex + itemsPerPage;
    return filteredData.slice(startIndex, endIndex);
  };

  const formatCellValue = (value: any, key: string) => {
    if (value === null || value === undefined) return '-';
    
    if (typeof value === 'boolean') {
      return value ? '✓' : '✗';
    }
    
    if (key.includes('date') && typeof value === 'string') {
      return new Date(value).toLocaleDateString();
    }
    
    if (key === 'conditions' && Array.isArray(value)) {
      return value.length > 0 ? value.join(', ') : '-';
    }
    
    return value.toString();
  };

  const exportToCSV = () => {
    const filteredData = getFilteredData();
    const selectedColumnsArray = Array.from(selectedColumns);
    
    const headers = selectedColumnsArray.map(key => 
      availableColumns.find(col => col.key === key)?.label || key
    );
    
    const csvContent = [
      headers.join(','),
      ...filteredData.map(patient => 
        selectedColumnsArray.map(key => {
          const value = patient[key as keyof Patient];
          const formatted = formatCellValue(value, key);
          return `"${formatted}"`;
        }).join(',')
      )
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `patient_records_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  const totalPages = Math.ceil(getFilteredData().length / itemsPerPage);
  const filteredData = getFilteredData();

  return (
    <div className="bg-gray-900 rounded-3xl border border-gray-800 p-6">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold text-white mb-2">Patient Records</h2>
          <p className="text-gray-400">
            {filteredData.length} patient{filteredData.length !== 1 ? 's' : ''} found
          </p>
        </div>
        
        <div className="flex flex-wrap gap-3 mt-4 lg:mt-0">
          <button
            onClick={() => setShowFilters(!showFilters)}
            className="px-4 py-2 bg-gray-800 text-white rounded-xl hover:bg-gray-700 transition-colors"
          >
            {showFilters ? 'Hide Filters' : 'Show Filters'}
          </button>
          
          <button
            onClick={onRefresh}
            className="px-4 py-2 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition-colors"
          >
            Refresh
          </button>
          
          <button
            onClick={exportToCSV}
            className="px-4 py-2 bg-green-600 text-white rounded-xl hover:bg-green-700 transition-colors"
          >
            Export CSV
          </button>
          
          <button
            onClick={onHbA1cAdjustment}
            className="px-4 py-2 bg-purple-600 text-white rounded-xl hover:bg-purple-700 transition-colors"
          >
            HbA1c Adjustment
          </button>
        </div>
      </div>

      {/* Search and Filters */}
      <div className="mb-6 space-y-4">
        <div className="flex flex-col sm:flex-row gap-4">
          <div className="flex-1">
            <input
              type="text"
              placeholder="Search patients by name, email, NHS number, MRN, postcode..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-xl text-white placeholder-gray-400 focus:ring-2 focus:ring-white focus:border-white"
            />
          </div>
        </div>

        {showFilters && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-gray-800 rounded-xl p-6 space-y-4"
          >
            <h3 className="text-lg font-semibold text-white mb-4">Filters</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium text-white mb-2">Sex</label>
                <select
                  value={filters.sex}
                  onChange={(e) => setFilters(prev => ({ ...prev, sex: e.target.value }))}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                >
                  <option value="">All</option>
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                  <option value="Other">Other</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-white mb-2">Ethnic Group</label>
                <select
                  value={filters.ethnic_group}
                  onChange={(e) => setFilters(prev => ({ ...prev, ethnic_group: e.target.value }))}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                >
                  <option value="">All</option>
                  <option value="White British">White British</option>
                  <option value="Asian or Asian British Indian">Asian Indian</option>
                  <option value="Asian or Asian British Pakistani">Asian Pakistani</option>
                  <option value="Black or Black British Caribbean">Black Caribbean</option>
                  <option value="Other">Other</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-white mb-2">BMI Range</label>
                <select
                  value={filters.bmi_range}
                  onChange={(e) => setFilters(prev => ({ ...prev, bmi_range: e.target.value }))}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                >
                  <option value="">All</option>
                  <option value="underweight">Underweight (&lt;18.5)</option>
                  <option value="normal">Normal (18.5-24.9)</option>
                  <option value="overweight">Overweight (25-29.9)</option>
                  <option value="obese">Obese (≥30)</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-white mb-2">HbA1c Range</label>
                <select
                  value={filters.hba1c_range}
                  onChange={(e) => setFilters(prev => ({ ...prev, hba1c_range: e.target.value }))}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                >
                  <option value="">All</option>
                  <option value="normal">Normal (&lt;5.7%)</option>
                  <option value="prediabetes">Prediabetes (5.7-6.4%)</option>
                  <option value="diabetes">Diabetes (≥6.5%)</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-white mb-2">Age Range</label>
                <select
                  value={filters.age_range}
                  onChange={(e) => setFilters(prev => ({ ...prev, age_range: e.target.value }))}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                >
                  <option value="">All</option>
                  <option value="18-30">18-30</option>
                  <option value="31-45">31-45</option>
                  <option value="46-60">46-60</option>
                  <option value="60+">60+</option>
                </select>
              </div>
            </div>

            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => setFilters({
                  sex: '', ethnic_group: '', ascvd: '', t2dm: '', htn: '', dyslipidaemia: '',
                  bmi_range: '', hba1c_range: '', age_range: '', conditions: []
                })}
                className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-500 transition-colors"
              >
                Clear Filters
              </button>
            </div>
          </motion.div>
        )}
      </div>

      {/* Column Selection */}
      <div className="mb-4">
        <details className="bg-gray-800 rounded-xl p-4">
          <summary className="text-white font-medium cursor-pointer">Column Selection</summary>
          <div className="mt-4 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
            {availableColumns.map(column => (
              <label key={column.key} className="flex items-center space-x-2 text-sm">
                <input
                  type="checkbox"
                  checked={selectedColumns.has(column.key)}
                  onChange={(e) => {
                    const newSelected = new Set(selectedColumns);
                    if (e.target.checked) {
                      newSelected.add(column.key);
                    } else {
                      newSelected.delete(column.key);
                    }
                    setSelectedColumns(newSelected);
                  }}
                  className="w-4 h-4 text-white bg-gray-700 border-gray-600 rounded focus:ring-white"
                />
                <span className="text-gray-300">{column.label}</span>
              </label>
            ))}
          </div>
        </details>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-gray-700">
              {Array.from(selectedColumns).map(columnKey => {
                const column = availableColumns.find(col => col.key === columnKey);
                if (!column) return null;
                
                return (
                  <th
                    key={columnKey}
                    className={`px-4 py-3 text-left text-sm font-medium text-gray-300 ${
                      column.sortable ? 'cursor-pointer hover:text-white' : ''
                    }`}
                    onClick={() => column.sortable && handleSort(columnKey as keyof Patient)}
                  >
                    <div className="flex items-center space-x-1">
                      <span>{column.label}</span>
                      {column.sortable && sortConfig?.key === columnKey && (
                        <span className="text-white">
                          {sortConfig.direction === 'asc' ? '↑' : '↓'}
                        </span>
                      )}
                    </div>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {getPaginatedData().map((patient) => (
              <tr
                key={patient.id}
                className={`border-b border-gray-800 hover:bg-gray-800/50 cursor-pointer ${
                  selectedPatientId === patient.id ? 'bg-blue-900/20 border-blue-500' : ''
                }`}
                onClick={() => onPatientSelect?.(patient)}
              >
                {Array.from(selectedColumns).map(columnKey => (
                  <td key={columnKey} className="px-4 py-3 text-sm text-gray-300">
                    {formatCellValue(patient[columnKey as keyof Patient], columnKey)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between mt-6">
          <div className="flex items-center space-x-2">
            <span className="text-sm text-gray-400">Rows per page:</span>
            <select
              value={itemsPerPage}
              onChange={(e) => {
                setItemsPerPage(Number(e.target.value));
                setCurrentPage(1);
              }}
              className="px-3 py-1 bg-gray-800 border border-gray-700 rounded text-white text-sm"
            >
              <option value={10}>10</option>
              <option value={25}>25</option>
              <option value={50}>50</option>
              <option value={100}>100</option>
            </select>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
              disabled={currentPage === 1}
              className="px-3 py-1 bg-gray-800 text-white rounded hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Previous
            </button>
            
            <span className="text-sm text-gray-400">
              Page {currentPage} of {totalPages}
            </span>
            
            <button
              onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
              disabled={currentPage === totalPages}
              className="px-3 py-1 bg-gray-800 text-white rounded hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
