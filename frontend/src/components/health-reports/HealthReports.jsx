import { useState } from 'react';
import { motion } from 'framer-motion';
import { FileText, Download, Calendar, TrendingUp } from 'lucide-react';
import DashboardCard from '../DashboardCard';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';

export const HealthReports = ({ patientId, userId }) => {
  const [reportType, setReportType] = useState('adherence');
  const [period, setPeriod] = useState('30'); // days
  const [isGenerating, setIsGenerating] = useState(false);

  const reportTypes = [
    { id: 'adherence', label: 'Adherence', icon: TrendingUp },
    { id: 'side_effects', label: 'Side Effects', icon: FileText },
    { id: 'trends', label: 'Health Trends', icon: TrendingUp },
    { id: 'comprehensive', label: 'Comprehensive', icon: FileText }
  ];

  const handleGenerateReport = async () => {
    setIsGenerating(true);
    try {
      const endDate = new Date();
      const startDate = new Date();
      startDate.setDate(startDate.getDate() - parseInt(period));

      // In a real implementation, this would call the API to generate a report
      // For now, we'll simulate it
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Download would be handled by the backend
      alert(`Report generated for ${reportType} (${period} days)`);
    } catch (error) {
      console.error('Error generating report:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleExport = (format) => {
    // In a real implementation, this would download the report
    alert(`Exporting report as ${format.toUpperCase()}`);
  };

  return (
    <DashboardCard
      title="Health Reports"
      icon={<FileText size={20} />}
      variant="patient"
    >
      <div className="space-y-6">
        {/* Report Type Selection */}
        <div>
          <label className="block text-sm font-medium text-neutral-700 mb-3">
            Report Type
          </label>
          <div className="grid grid-cols-2 gap-3">
            {reportTypes.map((type) => {
              const Icon = type.icon;
              return (
                <button
                  key={type.id}
                  onClick={() => setReportType(type.id)}
                  className={`p-4 rounded-xl border-2 transition-all text-left ${
                    reportType === type.id
                      ? 'border-primary-500 bg-primary-50'
                      : 'border-neutral-200 hover:border-primary-300'
                  }`}
                >
                  <Icon className="w-5 h-5 mb-2 text-primary-600" />
                  <div className="font-medium text-neutral-900">{type.label}</div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Time Period */}
        <div>
          <label className="block text-sm font-medium text-neutral-700 mb-2">
            Time Period
          </label>
          <div className="grid grid-cols-4 gap-2">
            {['7', '30', '90', '365'].map((days) => (
              <button
                key={days}
                onClick={() => setPeriod(days)}
                className={`p-3 rounded-xl border-2 transition-all ${
                  period === days
                    ? 'border-primary-500 bg-primary-50 text-primary-700'
                    : 'border-neutral-200 hover:border-primary-300'
                }`}
              >
                {days} days
              </button>
            ))}
          </div>
        </div>

        {/* Generate Button */}
        <Button
          onClick={handleGenerateReport}
          disabled={isGenerating}
          variant="primary"
          size="md"
          className="w-full"
        >
          {isGenerating ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
              Generating...
            </>
          ) : (
            <>
              <FileText className="w-4 h-4 mr-2" />
              Generate Report
            </>
          )}
        </Button>

        {/* Export Options */}
        <div className="pt-4 border-t border-neutral-200">
          <label className="block text-sm font-medium text-neutral-700 mb-3">
            Export Format
          </label>
          <div className="grid grid-cols-3 gap-2">
            {['PDF', 'CSV', 'JSON'].map((format) => (
              <Button
                key={format}
                onClick={() => handleExport(format.toLowerCase())}
                variant="outline"
                size="sm"
              >
                <Download className="w-4 h-4 mr-1" />
                {format}
              </Button>
            ))}
          </div>
        </div>

        {/* Report Preview Placeholder */}
        <div className="p-6 bg-neutral-50 rounded-xl border border-neutral-200">
          <div className="flex items-center gap-2 mb-3">
            <Calendar className="w-5 h-5 text-neutral-600" />
            <span className="font-medium text-neutral-900">Report Preview</span>
          </div>
          <p className="text-sm text-neutral-600">
            Select a report type and time period, then click "Generate Report" to create your health report.
          </p>
        </div>
      </div>
    </DashboardCard>
  );
};



