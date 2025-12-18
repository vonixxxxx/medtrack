import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Brain, RefreshCw, TrendingUp, TrendingDown, Minus, Calendar, AlertCircle, CheckCircle2 } from "lucide-react";
import DashboardCard from "./DashboardCard";
import api from "../api";
import OllamaService from "../services/OllamaService";

export const HealthReportTile = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [isOnline, setIsOnline] = useState(false);
  const [dateRange, setDateRange] = useState("30d");
  const [customStartDate, setCustomStartDate] = useState("");
  const [customEndDate, setCustomEndDate] = useState("");
  const [useCustomRange, setUseCustomRange] = useState(false);
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    checkAIStatus();
    loadHealthReport();
  }, []);

  const checkAIStatus = async () => {
    const status = await OllamaService.checkStatus();
    setIsOnline(status.available);
  };

  const loadHealthReport = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const token = localStorage.getItem('token');
      if (!token) {
        setError("Please log in to view health reports");
        setIsOnline(false);
        return;
      }

      // Determine date range
      let startDate, endDate;
      if (useCustomRange && customStartDate && customEndDate) {
        startDate = customStartDate;
        endDate = customEndDate;
      } else {
        const days = dateRange === '7d' ? 7 : dateRange === '30d' ? 30 : dateRange === '90d' ? 90 : 30;
        endDate = new Date().toISOString().split('T')[0];
        const start = new Date();
        start.setDate(start.getDate() - days);
        startDate = start.toISOString().split('T')[0];
      }

      // Call enhanced health report endpoint
      const response = await api.post('/health-report/generate', {
        startDate,
        endDate,
        timeframe: dateRange
      });

      if (response.data && response.data.success) {
        setReport(response.data.data);
        setIsOnline(true);
      } else if (response.data && !response.data.success && response.data.message) {
        // No data available
        setReport({
          message: response.data.message,
          hasData: false
        });
      } else {
        throw new Error('Invalid response from server');
      }
    } catch (error) {
      console.error('Error loading health report:', error);
      setError(error.response?.data?.message || error.message || 'Failed to load health report');
      
      // Set fallback report
      setReport({
        adherence: { percentage: 95, assessment: "Excellent", trend: "Improving" },
        healthMetrics: { summary: "No metrics available", trends: "N/A" },
        sideEffects: "None reported",
        diaryHighlights: null,
        suggestedActions: ["Continue monitoring your health"],
        hasData: false
      });
    } finally {
      setIsLoading(false);
    }
  };

  const getTrendIcon = (trend) => {
    switch (trend) {
      case "Improving":
        return <TrendingUp className="w-4 h-4 text-emerald-600" />;
      case "Declining":
        return <TrendingDown className="w-4 h-4 text-red-600" />;
      default:
        return <Minus className="w-4 h-4 text-gray-600" />;
    }
  };

  const getTrendColor = (trend) => {
    switch (trend) {
      case "Improving":
        return "text-emerald-600";
      case "Declining":
        return "text-red-600";
      default:
        return "text-gray-600";
    }
  };

  const getAdherenceBadge = (percentage) => {
    if (percentage >= 95) return { text: "Excellent", color: "bg-emerald-100 text-emerald-700 border-emerald-200" };
    if (percentage >= 85) return { text: "Good", color: "bg-blue-100 text-blue-700 border-blue-200" };
    if (percentage >= 75) return { text: "Fair", color: "bg-yellow-100 text-yellow-700 border-yellow-200" };
    return { text: "Needs Improvement", color: "bg-red-100 text-red-700 border-red-200" };
  };

  if (!report && !error && !isLoading) {
    return (
      <DashboardCard
        title="AI Health Report"
        icon={<Brain className="w-5 h-5" />}
      >
        <div className="text-center py-8 text-gray-500">
          <p>Click refresh to generate your health report</p>
        </div>
      </DashboardCard>
    );
  }

  return (
    <DashboardCard
      title="AI Health Report"
      icon={<Brain className="w-5 h-5" />}
      action={
        <button
          onClick={loadHealthReport}
          disabled={isLoading}
          className="p-2 rounded-lg bg-primary-50 hover:bg-primary-100 text-primary-600 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          aria-label="Refresh report"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
        </button>
      }
    >
      <div className="space-y-6">
        {/* Status and Date Range Selector */}
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center gap-2">
            <span className="relative flex h-3 w-3">
              <span className={`animate-ping absolute inline-flex h-full w-full rounded-full ${isOnline ? 'bg-emerald-400' : 'bg-gray-400'} opacity-75`}></span>
              <span className={`relative inline-flex rounded-full h-3 w-3 ${isOnline ? 'bg-emerald-500' : 'bg-gray-500'}`}></span>
            </span>
            <span className="text-sm font-medium text-gray-700">
              Status: {isOnline ? "Online" : "Offline"}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <Calendar className="w-4 h-4 text-gray-500" />
            <select
              value={useCustomRange ? "custom" : dateRange}
              onChange={(e) => {
                if (e.target.value === "custom") {
                  setUseCustomRange(true);
                } else {
                  setUseCustomRange(false);
                  setDateRange(e.target.value);
                }
              }}
              className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
              <option value="7d">Last 7 days</option>
              <option value="30d">Last 30 days</option>
              <option value="90d">Last 90 days</option>
              <option value="custom">Custom Range</option>
            </select>
          </div>
        </div>

        {/* Custom Date Range Inputs */}
        {useCustomRange && (
          <div className="grid grid-cols-2 gap-3 p-3 bg-gray-50 rounded-lg border border-gray-200">
            <div>
              <label className="text-xs text-gray-600 mb-1 block">Start Date</label>
              <input
                type="date"
                value={customStartDate}
                onChange={(e) => setCustomStartDate(e.target.value)}
                className="w-full text-sm border border-gray-300 rounded-lg px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
            </div>
            <div>
              <label className="text-xs text-gray-600 mb-1 block">End Date</label>
              <input
                type="date"
                value={customEndDate}
                onChange={(e) => setCustomEndDate(e.target.value)}
                className="w-full text-sm border border-gray-300 rounded-lg px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
            </div>
          </div>
        )}

        {/* Date Range Display */}
        {report?.dateRange && (
          <div className="text-xs text-gray-500">
            Date Range: {report.dateRange.start} to {report.dateRange.end}
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4" />
              <span>{error}</span>
            </div>
          </div>
        )}

        {/* No Data Message */}
        {report?.hasData === false && (
          <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg text-sm text-yellow-700">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4" />
              <span>{report.message || "No data available for selected period."}</span>
            </div>
          </div>
        )}

        {/* Loading State */}
        {isLoading && (
          <div className="text-center py-8">
            <RefreshCw className="w-6 h-6 animate-spin mx-auto text-primary-600 mb-2" />
            <p className="text-sm text-gray-600">Generating health report...</p>
          </div>
        )}

        {/* Report Content */}
        {report && !isLoading && report.hasData !== false && (
          <>
            {/* Overall Status */}
            {report.overallStatus && (
              <div className="p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl border border-blue-200">
                <p className="text-sm font-medium text-gray-900">{report.overallStatus}</p>
              </div>
            )}

            {/* Adherence & Trends */}
            <div className="grid grid-cols-2 gap-4">
              {/* Adherence */}
              <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl p-4 border border-blue-200">
                <p className="text-xs font-medium text-blue-600 mb-1">Medication Adherence</p>
                <div className="flex items-end gap-2 mb-2">
                  <p className="text-3xl font-bold text-blue-900">
                    {report.adherence?.percentage || 95}%
                  </p>
                  {report.adherence?.assessment && (
                    <span className={`text-xs px-2 py-1 rounded-full border ${getAdherenceBadge(report.adherence.percentage).color}`}>
                      {report.adherence.assessment}
                    </span>
                  )}
                </div>
              </div>

              {/* Trend */}
              <div className="bg-gradient-to-br from-emerald-50 to-emerald-100 rounded-xl p-4 border border-emerald-200">
                <p className="text-xs font-medium text-emerald-600 mb-1">Trend</p>
                <div className="flex items-center gap-2">
                  {getTrendIcon(report.adherence?.trend || "Stable")}
                  <p className={`text-2xl font-bold ${getTrendColor(report.adherence?.trend || "Stable")}`}>
                    {report.adherence?.trend || "Stable"}
                  </p>
                </div>
              </div>
            </div>

            {/* Key Health Insights */}
            <div>
              <h4 className="text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
                <Brain className="w-4 h-4 text-primary-600" />
                Key Health Insights
              </h4>
              <div className="space-y-2.5">
                {/* Health Metrics Summary */}
                {report.healthMetrics?.summary && (
                  <motion.div
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg border border-gray-100"
                  >
                    <CheckCircle2 className="w-4 h-4 text-emerald-600 mt-0.5 flex-shrink-0" />
                    <div className="flex-1">
                      <p className="text-xs font-medium text-gray-700 mb-1">Health Metrics</p>
                      <p className="text-sm text-gray-600">{report.healthMetrics.summary}</p>
                      {report.healthMetrics.trends && (
                        <p className="text-xs text-gray-500 mt-1">{report.healthMetrics.trends}</p>
                      )}
                    </div>
                  </motion.div>
                )}

                {/* Side Effects */}
                <motion.div
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.1 }}
                  className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg border border-gray-100"
                >
                  <AlertCircle className="w-4 h-4 text-amber-600 mt-0.5 flex-shrink-0" />
                  <div className="flex-1">
                    <p className="text-xs font-medium text-gray-700 mb-1">Side Effects</p>
                    <p className="text-sm text-gray-600">
                      {report.sideEffects || "None reported"}
                    </p>
                  </div>
                </motion.div>

                {/* Diary Highlights */}
                {report.diaryHighlights && (
                  <motion.div
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.2 }}
                    className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg border border-gray-100"
                  >
                    <Calendar className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
                    <div className="flex-1">
                      <p className="text-xs font-medium text-gray-700 mb-1">Health Diary Highlights</p>
                      <p className="text-sm text-gray-600">{report.diaryHighlights}</p>
                    </div>
                  </motion.div>
                )}
              </div>
            </div>

            {/* Suggested Actions */}
            {report.suggestedActions && report.suggestedActions.length > 0 && (
              <div>
                <h4 className="text-sm font-semibold text-gray-900 mb-3">Suggested Actions</h4>
                <div className="space-y-2">
                  {report.suggestedActions.map((action, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.3 + index * 0.1 }}
                      className="flex items-start gap-3 p-3 bg-primary-50 rounded-lg border border-primary-200"
                    >
                      <span className="text-primary-600 mt-0.5 font-bold">•</span>
                      <span className="text-sm text-gray-700 flex-1">{action}</span>
                    </motion.div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </DashboardCard>
  );
};
