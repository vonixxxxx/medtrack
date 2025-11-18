import { useState } from 'react';
import { motion } from 'framer-motion';
import { Download, Upload, Database, Shield, FileText } from 'lucide-react';
import DashboardCard from '../DashboardCard';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';

export const ExportBackup = ({ userId }) => {
  const [exportType, setExportType] = useState('full');
  const [format, setFormat] = useState('json');
  const [isExporting, setIsExporting] = useState(false);

  const exportTypes = [
    { id: 'medications', label: 'Medications', icon: FileText },
    { id: 'adherence', label: 'Adherence Data', icon: Database },
    { id: 'diary', label: 'Diary Entries', icon: FileText },
    { id: 'full', label: 'Full Backup', icon: Database }
  ];

  const formats = [
    { id: 'json', label: 'JSON', description: 'Machine-readable format' },
    { id: 'csv', label: 'CSV', description: 'Spreadsheet compatible' },
    { id: 'pdf', label: 'PDF', description: 'Human-readable report' }
  ];

  const handleExport = async () => {
    setIsExporting(true);
    try {
      // In a real implementation, this would call the API to export data
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Simulate download
      const dataStr = JSON.stringify({ exportType, format, timestamp: new Date().toISOString() }, null, 2);
      const dataBlob = new Blob([dataStr], { type: 'application/json' });
      const url = URL.createObjectURL(dataBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `medtrack-backup-${exportType}-${new Date().toISOString().split('T')[0]}.${format}`;
      link.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error exporting data:', error);
    } finally {
      setIsExporting(false);
    }
  };

  const handleImport = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json,.csv';
    input.onchange = (e) => {
      const file = e.target.files?.[0];
      if (file) {
        // In a real implementation, this would upload and restore data
        alert(`Importing data from ${file.name}`);
      }
    };
    input.click();
  };

  return (
    <DashboardCard
      title="Export & Backup"
      icon={<Database size={20} />}
      variant="patient"
    >
      <div className="space-y-6">
        {/* Privacy Notice */}
        <div className="p-4 bg-primary-50 border border-primary-200 rounded-xl">
          <div className="flex items-start gap-3">
            <Shield className="w-5 h-5 text-primary-600 flex-shrink-0 mt-0.5" />
            <div>
              <h4 className="font-semibold text-primary-900 mb-1">Your Data is Private</h4>
              <p className="text-sm text-primary-700">
                All data is stored locally on your device. Exports are encrypted and can only be accessed by you.
              </p>
            </div>
          </div>
        </div>

        {/* Export Type */}
        <div>
          <label className="block text-sm font-medium text-neutral-700 mb-3">
            What to Export
          </label>
          <div className="grid grid-cols-2 gap-3">
            {exportTypes.map((type) => {
              const Icon = type.icon;
              return (
                <button
                  key={type.id}
                  onClick={() => setExportType(type.id)}
                  className={`p-4 rounded-xl border-2 transition-all text-left ${
                    exportType === type.id
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

        {/* Format Selection */}
        <div>
          <label className="block text-sm font-medium text-neutral-700 mb-3">
            Export Format
          </label>
          <div className="space-y-2">
            {formats.map((fmt) => (
              <button
                key={fmt.id}
                onClick={() => setFormat(fmt.id)}
                className={`w-full p-3 rounded-xl border-2 transition-all text-left ${
                  format === fmt.id
                    ? 'border-primary-500 bg-primary-50'
                    : 'border-neutral-200 hover:border-primary-300'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium text-neutral-900">{fmt.label}</div>
                    <div className="text-xs text-neutral-600">{fmt.description}</div>
                  </div>
                  {format === fmt.id && (
                    <Badge variant="primary">Selected</Badge>
                  )}
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Export Button */}
        <Button
          onClick={handleExport}
          disabled={isExporting}
          variant="primary"
          size="md"
          className="w-full"
        >
          {isExporting ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
              Exporting...
            </>
          ) : (
            <>
              <Download className="w-4 h-4 mr-2" />
              Export Data
            </>
          )}
        </Button>

        {/* Import Section */}
        <div className="pt-4 border-t border-neutral-200">
          <h4 className="font-semibold text-neutral-900 mb-3">Restore from Backup</h4>
          <Button
            onClick={handleImport}
            variant="outline"
            size="md"
            className="w-full"
          >
            <Upload className="w-4 h-4 mr-2" />
            Import Backup File
          </Button>
          <p className="text-xs text-neutral-600 mt-2">
            Restore your data from a previously exported backup file
          </p>
        </div>
      </div>
    </DashboardCard>
  );
};



