import { useState } from 'react';
import { motion } from 'framer-motion';

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
  const [isExpanded, setIsExpanded] = useState(false);

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
      className="bg-gray-900 rounded-3xl border border-gray-800 p-6"
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Filters</h3>
        <div className="flex items-center gap-2">
          {hasActiveFilters && (
            <button
              onClick={clearFilters}
              className="px-3 py-1 text-sm bg-gray-700 text-gray-300 rounded-lg hover:bg-gray-600 transition-colors"
            >
              Clear All
            </button>
          )}
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="px-3 py-1 text-sm bg-gray-700 text-gray-300 rounded-lg hover:bg-gray-600 transition-colors"
          >
            {isExpanded ? 'Collapse' : 'Expand'}
          </button>
        </div>
      </div>

      <div className={`grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 ${isExpanded ? 'block' : 'hidden'}`}>
        {/* Metric Filter */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            By Metric
          </label>
          <select
            value={filters.metric}
            onChange={(e) => handleFilterChange('metric', e.target.value)}
            className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-xl text-white focus:ring-2 focus:ring-white focus:border-white"
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
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Date Range
          </label>
          <select
            value={filters.dateRange}
            onChange={(e) => handleFilterChange('dateRange', e.target.value)}
            className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-xl text-white focus:ring-2 focus:ring-white focus:border-white"
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
          <label className="block text-sm font-medium text-gray-300 mb-2">
            By Ethnicity
          </label>
          <select
            value={filters.ethnicity}
            onChange={(e) => handleFilterChange('ethnicity', e.target.value)}
            className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-xl text-white focus:ring-2 focus:ring-white focus:border-white"
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
          <label className="block text-sm font-medium text-gray-300 mb-2">
            By Sex
          </label>
          <select
            value={filters.sex}
            onChange={(e) => handleFilterChange('sex', e.target.value)}
            className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-xl text-white focus:ring-2 focus:ring-white focus:border-white"
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
        <div className="mt-4 pt-4 border-t border-gray-700">
          <div className="flex flex-wrap gap-2">
            <span className="text-sm text-gray-400">Active filters:</span>
            {filters.metric !== 'all' && (
              <span className="px-2 py-1 bg-blue-600 text-white text-xs rounded-full">
                Metric: {filters.metric}
              </span>
            )}
            {filters.dateRange !== 'all' && (
              <span className="px-2 py-1 bg-green-600 text-white text-xs rounded-full">
                Date: {filters.dateRange}
              </span>
            )}
            {filters.ethnicity !== 'all' && (
              <span className="px-2 py-1 bg-purple-600 text-white text-xs rounded-full">
                Ethnicity: {filters.ethnicity}
              </span>
            )}
            {filters.sex !== 'all' && (
              <span className="px-2 py-1 bg-orange-600 text-white text-xs rounded-full">
                Sex: {filters.sex}
              </span>
            )}
          </div>
        </div>
      )}
    </motion.div>
  );
};


