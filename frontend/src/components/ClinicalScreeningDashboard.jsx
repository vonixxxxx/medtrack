import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Heart, 
  Droplets, 
  Activity, 
  FileText, 
  Download, 
  Plus, 
  Calendar,
  TrendingUp,
  AlertTriangle,
  CheckCircle
} from 'lucide-react';
import IIEF5Screening from './IIEF5Screening';
import AUDITScreening from './AUDITScreening';
import HeartRiskCalculator from './HeartRiskCalculator';

const ClinicalScreeningDashboard = () => {
  const [activeAssessment, setActiveAssessment] = useState(null);
  const [screeningResults, setScreeningResults] = useState({
    iief5: null,
    audit: null,
    heartRisk: null
  });

  const handleIIEF5Submit = (results) => {
    setScreeningResults(prev => ({
      ...prev,
      iief5: results
    }));
    setActiveAssessment(null);
  };

  const handleAUDITSubmit = (results) => {
    setScreeningResults(prev => ({
      ...prev,
      audit: results
    }));
    setActiveAssessment(null);
  };

  const handleHeartRiskSubmit = (results) => {
    setScreeningResults(prev => ({
      ...prev,
      heartRisk: results
    }));
    setActiveAssessment(null);
  };

  const getSeverityIcon = (severity) => {
    if (severity?.level === 'No ED') return <CheckCircle className="w-5 h-5 text-green-600" />;
    if (severity?.level.includes('Mild')) return <AlertTriangle className="w-5 h-5 text-yellow-600" />;
    return <AlertTriangle className="w-5 h-5 text-red-600" />;
  };

  const getRiskIcon = (riskCategory) => {
    if (riskCategory?.level === 'Low Risk') return <CheckCircle className="w-5 h-5 text-green-600" />;
    if (riskCategory?.level === 'Hazardous Drinking') return <AlertTriangle className="w-5 h-5 text-yellow-600" />;
    return <AlertTriangle className="w-5 h-5 text-red-600" />;
  };

  const getRiskColor = (riskCategory) => {
    if (riskCategory?.includes('Low')) return 'text-green-600';
    if (riskCategory?.includes('Borderline') || riskCategory?.includes('Intermediate')) return 'text-yellow-600';
    return 'text-red-600';
  };

  const exportResults = () => {
    const data = {
      timestamp: new Date().toISOString(),
      iief5: screeningResults.iief5,
      audit: screeningResults.audit
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `clinical-screening-results-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  if (activeAssessment === 'iief5') {
    return (
      <div className="min-h-screen bg-gray-50 py-8">
        <div className="max-w-4xl mx-auto px-4">
          <button
            onClick={() => setActiveAssessment(null)}
            className="mb-6 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors flex items-center"
          >
            ← Back to Dashboard
          </button>
          <IIEF5Screening onSubmit={handleIIEF5Submit} />
        </div>
      </div>
    );
  }

  if (activeAssessment === 'audit') {
    return (
      <div className="min-h-screen bg-gray-50 py-8">
        <div className="max-w-4xl mx-auto px-4">
          <button
            onClick={() => setActiveAssessment(null)}
            className="mb-6 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors flex items-center"
          >
            ← Back to Dashboard
          </button>
          <AUDITScreening onSubmit={handleAUDITSubmit} />
        </div>
      </div>
    );
  }

  if (activeAssessment === 'heartRisk') {
    return (
      <div className="min-h-screen bg-gray-50 py-8">
        <div className="max-w-4xl mx-auto px-4">
          <button
            onClick={() => setActiveAssessment(null)}
            className="mb-6 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors flex items-center"
          >
            ← Back to Dashboard
          </button>
          <HeartRiskCalculator onSubmit={handleHeartRiskSubmit} />
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">Clinical Screening Dashboard</h1>
          <p className="text-gray-600">Comprehensive health assessments and screening tools</p>
        </div>

        {/* Assessment Cards */}
        <div className="grid md:grid-cols-3 gap-6 mb-8">
          <div className="bg-white border border-gray-200 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-800">IIEF-5</h3>
              <span title="IIEF-5 assesses erectile function (score 5-25; higher is better)." className="text-gray-400 cursor-help">?</span>
            </div>
            <p className="text-sm text-gray-600 mb-4">Erectile function screening (5 questions)</p>
            <button onClick={() => setActiveAssessment('iief5')} className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-md py-2 hover:from-blue-700 hover:to-indigo-700">
              {screeningResults.iief5 ? 'Retake Assessment' : 'Start Assessment'}
            </button>
            {screeningResults.iief5 && (
              <p className="text-xs text-gray-600 mt-2">Score: {screeningResults.iief5.totalScore} ({screeningResults.iief5.severity.level})</p>
            )}
          </div>

          <div className="bg-white border border-gray-200 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-800">AUDIT</h3>
              <span title="AUDIT screens for risky alcohol use (score 0-40)." className="text-gray-400 cursor-help">?</span>
            </div>
            <p className="text-sm text-gray-600 mb-4">Alcohol use screening (10 questions)</p>
            <button onClick={() => setActiveAssessment('audit')} className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-md py-2 hover:from-blue-700 hover:to-indigo-700">
              {screeningResults.audit ? 'Retake Assessment' : 'Start Assessment'}
            </button>
            {screeningResults.audit && (
              <p className="text-xs text-gray-600 mt-2">Score: {screeningResults.audit.totalScore} ({screeningResults.audit.riskCategory.level})</p>
            )}
          </div>

          <div className="bg-white border border-gray-200 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-800">Heart Risk</h3>
              <span title="10-year ASCVD/Framingham risk estimate." className="text-gray-400 cursor-help">?</span>
            </div>
            <p className="text-sm text-gray-600 mb-4">ASCVD/Framingham 10-year risk</p>
            <button onClick={() => setActiveAssessment('heartRisk')} className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-md py-2 hover:from-blue-700 hover:to-indigo-700">
              {screeningResults.heartRisk ? 'Retake Assessment' : 'Start Assessment'}
            </button>
            {screeningResults.heartRisk && (
              <p className="text-xs text-gray-600 mt-2">Risk: {screeningResults.heartRisk.result?.riskPercentage}% ({screeningResults.heartRisk.result?.riskCategory})</p>
            )}
          </div>
        </div>

        {/* Assessment Results Summary */}
        <div className="bg-white rounded-xl border border-gray-200 p-6 mb-8">
          <h3 className="text-xl font-semibold text-gray-800 mb-4">Assessment Results Summary</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {screeningResults.iief5 && (
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-600">IIEF-5</span>
                  {getSeverityIcon(screeningResults.iief5.severity)}
                </div>
                <p className="text-2xl font-bold text-gray-900">{screeningResults.iief5.totalScore}/25</p>
                <p className="text-xs text-gray-500">{screeningResults.iief5.severity.level}</p>
              </div>
            )}

            {screeningResults.audit && (
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-600">AUDIT</span>
                  {getRiskIcon(screeningResults.audit.riskCategory)}
                </div>
                <p className="text-2xl font-bold text-gray-900">{screeningResults.audit.totalScore}/40</p>
                <p className="text-xs text-gray-500">{screeningResults.audit.riskCategory.level}</p>
              </div>
            )}

            {screeningResults.heartRisk && (
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-600">Heart Risk</span>
                  {getRiskIcon(screeningResults.heartRisk.result.riskCategory)}
                </div>
                <p className="text-2xl font-bold text-gray-900">{screeningResults.heartRisk.result.riskPercentage}%</p>
                <p className="text-xs text-gray-500">{screeningResults.heartRisk.result.riskCategory}</p>
              </div>
            )}

            {screeningResults.testosterone && (
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-600">Testosterone</span>
                  <span className={`w-3 h-3 rounded-full ${screeningResults.testosterone.severity === 'low' ? 'bg-yellow-400' : 'bg-green-400'}`}></span>
                </div>
                <p className="text-2xl font-bold text-gray-900">{screeningResults.testosterone.totalScore}</p>
                <p className="text-xs text-gray-500">{screeningResults.testosterone.severity}</p>
              </div>
            )}
          </div>
        </div>

        {/* Assessment History */}
        <div className="bg-white rounded-xl border border-gray-200">
          <div className="p-6 border-b border-gray-200">
            <h3 className="text-xl font-semibold text-gray-800">Assessment History</h3>
          </div>
          <div className="p-6">
            <div className="space-y-4">
              {screeningResults.iief5 && (
                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <div>
                    <h4 className="font-medium text-gray-900">IIEF-5 Assessment</h4>
                    <p className="text-sm text-gray-500">
                      Completed on {new Date(screeningResults.iief5.completedAt).toLocaleDateString()}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-lg font-semibold text-gray-900">{screeningResults.iief5.totalScore}/25</p>
                    <p className="text-sm text-gray-500">{screeningResults.iief5.severity.level}</p>
                  </div>
                </div>
              )}

              {screeningResults.audit && (
                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <div>
                    <h4 className="font-medium text-gray-900">AUDIT Assessment</h4>
                    <p className="text-sm text-gray-500">
                      Completed on {new Date(screeningResults.audit.completedAt).toLocaleDateString()}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-lg font-semibold text-gray-900">{screeningResults.audit.totalScore}/40</p>
                    <p className="text-sm text-gray-500">{screeningResults.audit.riskCategory.level}</p>
                  </div>
                </div>
              )}

              {screeningResults.heartRisk && (
                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <div>
                    <h4 className="font-medium text-gray-900">Heart Risk Assessment</h4>
                    <p className="text-sm text-gray-500">
                      Completed on {new Date(screeningResults.heartRisk.completedAt).toLocaleDateString()}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-lg font-semibold text-gray-900">{screeningResults.heartRisk.result.riskPercentage}%</p>
                    <p className="text-sm text-gray-500">{screeningResults.heartRisk.result.riskCategory}</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Information Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          <div className="bg-white border border-gray-200 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-800">IIEF-5</h3>
              <span title="IIEF-5 assesses erectile function (score 5-25; higher is better)." className="text-gray-400 cursor-help">?</span>
            </div>
            <p className="text-sm text-gray-600 mb-4">Erectile function screening (5 questions)</p>
            <button onClick={() => setActiveAssessment('iief5')} className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-md py-2 hover:from-blue-700 hover:to-indigo-700">
              {screeningResults.iief5 ? 'Retake Assessment' : 'Start Assessment'}
            </button>
            {screeningResults.iief5 && (
              <p className="text-xs text-gray-600 mt-2">Score: {screeningResults.iief5.totalScore} ({screeningResults.iief5.severity.level})</p>
            )}
          </div>

          <div className="bg-white border border-gray-200 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-800">AUDIT</h3>
              <span title="AUDIT screens for risky alcohol use (score 0-40)." className="text-gray-400 cursor-help">?</span>
            </div>
            <p className="text-sm text-gray-600 mb-4">Alcohol use screening (10 questions)</p>
            <button onClick={() => setActiveAssessment('audit')} className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-md py-2 hover:from-blue-700 hover:to-indigo-700">
              {screeningResults.audit ? 'Retake Assessment' : 'Start Assessment'}
            </button>
            {screeningResults.audit && (
              <p className="text-xs text-gray-600 mt-2">Score: {screeningResults.audit.totalScore} ({screeningResults.audit.riskCategory.level})</p>
            )}
          </div>

          <div className="bg-white border border-gray-200 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-800">Heart Risk</h3>
              <span title="10-year ASCVD/Framingham risk estimate." className="text-gray-400 cursor-help">?</span>
            </div>
            <p className="text-sm text-gray-600 mb-4">ASCVD/Framingham 10-year risk</p>
            <button onClick={() => setActiveAssessment('heartRisk')} className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-md py-2 hover:from-blue-700 hover:to-indigo-700">
              {screeningResults.heartRisk ? 'Retake Assessment' : 'Start Assessment'}
            </button>
            {screeningResults.heartRisk && (
              <p className="text-xs text-gray-600 mt-2">Risk: {screeningResults.heartRisk.result?.riskPercentage}% ({screeningResults.heartRisk.result?.riskCategory})</p>
            )}
          </div>
        </div>
        <div className="flex justify-center mt-8">
          <button
            onClick={exportResults}
            className="px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-md hover:from-blue-700 hover:to-indigo-700 flex items-center gap-2"
          >
            <Download className="w-5 h-5" />
            Export All Results
          </button>
        </div>
      </div>
    </div>
  );
};

export default ClinicalScreeningDashboard;
