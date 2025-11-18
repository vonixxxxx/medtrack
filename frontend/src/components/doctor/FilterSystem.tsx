import { motion } from 'framer-motion';
import { Filter, X } from 'lucide-react';

interface FilterSystemProps {
  filters: {
    metric: string;
    dateRange: string;
    ethnicity: string;
    sex: string;
  };
  onFilterChange: (filters: any) => void;
}

export const FilterSystem = ({ filters, onFilterChange }: FilterSystemProps) => {

  const handleFilterChange = (key: string, value: string) => {
    onFilterChange({
      ...filters,
      [key]: value
    });
  };

  const clearFilters = () => {
    onFilterChange({
      metric: 'all',
      dateRange: 'all',
      ethnicity: 'all',
      sex: 'all'
    });
  };

  const hasActiveFilters = Object.values(filters).some(value => value !== 'all');

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ y: -2 }}
      className="bg-gradient-to-br from-white to-blue-50/30 rounded-2xl border border-blue-100 hover:border-blue-200 shadow-lg shadow-blue-600/5 hover:shadow-xl hover:shadow-blue-600/20 transition-all p-6"
    >
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl shadow-lg">
            <Filter className="text-white" size={20} />
          </div>
          <h3 className="text-xl font-bold text-gray-900">Filters</h3>
        </div>
        <div className="flex items-center gap-2">
          {hasActiveFilters && (
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={clearFilters}
              className="px-4 py-2 text-sm font-medium bg-gray-100 text-gray-700 rounded-xl hover:bg-gray-200 transition-colors flex items-center gap-2 border border-gray-200"
            >
              <X size={16} />
              Clear All
            </motion.button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Metric Filter */}
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            By Metric
          </label>
          <select
            value={filters.metric}
            onChange={(e) => handleFilterChange('metric', e.target.value)}
            className="w-full px-4 py-3 bg-white border-2 border-gray-200 rounded-xl text-gray-900 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all outline-none hover:border-blue-300"
          >
            <option value="all">All Metrics</option>
            <option value="hba1c">HbA1c</option>
            <option value="bp">Blood Pressure</option>
            <option value="bmi">BMI</option>
            <option value="weight">Weight</option>
            <option value="glucose">Glucose</option>
          </select>
        </div>

        {/* Date Range Filter */}
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            Date Range
          </label>
          <select
            value={filters.dateRange}
            onChange={(e) => handleFilterChange('dateRange', e.target.value)}
            className="w-full px-4 py-3 bg-white border-2 border-gray-200 rounded-xl text-gray-900 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all outline-none hover:border-blue-300"
          >
            <option value="all">All Time</option>
            <option value="today">Today</option>
            <option value="week">This Week</option>
            <option value="month">This Month</option>
            <option value="quarter">This Quarter</option>
            <option value="year">This Year</option>
            <option value="custom">Custom Range</option>
          </select>
        </div>

        {/* Ethnicity Filter */}
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            By Ethnicity
          </label>
          <select
            value={filters.ethnicity}
            onChange={(e) => handleFilterChange('ethnicity', e.target.value)}
            className="w-full px-4 py-3 bg-white border-2 border-gray-200 rounded-xl text-gray-900 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all outline-none hover:border-blue-300"
          >
            <option value="all">All Ethnicities</option>
            <option value="white">White</option>
            <option value="black">Black/African American</option>
            <option value="hispanic">Hispanic/Latino</option>
            <option value="asian">Asian</option>
            <option value="native">Native American</option>
            <option value="pacific">Pacific Islander</option>
            <option value="other">Other</option>
            <option value="unknown">Unknown</option>
          </select>
        </div>

        {/* Sex Filter */}
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            By Sex
          </label>
          <select
            value={filters.sex}
            onChange={(e) => handleFilterChange('sex', e.target.value)}
            className="w-full px-4 py-3 bg-white border-2 border-gray-200 rounded-xl text-gray-900 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all outline-none hover:border-blue-300"
          >
            <option value="all">All</option>
            <option value="male">Male</option>
            <option value="female">Female</option>
            <option value="other">Other</option>
          </select>
        </div>
      </div>

      {/* Active Filters Display */}
      {hasActiveFilters && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          className="mt-6 pt-6 border-t border-gray-200"
        >
          <div className="flex flex-wrap gap-2">
            <span className="text-sm font-medium text-gray-600">Active filters:</span>
            {filters.metric !== 'all' && (
              <span className="px-3 py-1.5 bg-blue-100 text-blue-700 text-xs font-medium rounded-full border border-blue-200">
                Metric: {filters.metric}
              </span>
            )}
            {filters.dateRange !== 'all' && (
              <span className="px-3 py-1.5 bg-blue-100 text-blue-700 text-xs font-medium rounded-full border border-blue-200">
                Date: {filters.dateRange}
              </span>
            )}
            {filters.ethnicity !== 'all' && (
              <span className="px-3 py-1.5 bg-blue-100 text-blue-700 text-xs font-medium rounded-full border border-blue-200">
                Ethnicity: {filters.ethnicity}
              </span>
            )}
            {filters.sex !== 'all' && (
              <span className="px-3 py-1.5 bg-blue-100 text-blue-700 text-xs font-medium rounded-full border border-blue-200">
                Sex: {filters.sex}
              </span>
            )}
          </div>
        </motion.div>
      )}
    </motion.div>
  );
};
