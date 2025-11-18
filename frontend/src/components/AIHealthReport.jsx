import { Brain, RefreshCw } from "lucide-react";
import { MedTrackCard } from "./MedTrackCard";
import { NeonButton } from "./NeonButton";
import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import OllamaService from "../services/OllamaService";

export const AIHealthReport = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [isOnline, setIsOnline] = useState(false);
  const [report, setReport] = useState({
    adherence: "95%",
    trend: "Improving",
    insights: [
      "Blood sugar levels are stable and within target range",
      "Medication adherence is excellent this week",
      "Consider scheduling your next check-up"
    ]
  });

  useEffect(() => {
    checkAIStatus();
  }, []);

  const checkAIStatus = async () => {
    const status = await OllamaService.checkStatus();
    setIsOnline(status.available);
  };

  const refreshReport = async () => {
    setIsLoading(true);
    try {
      const status = await OllamaService.checkStatus();
      setIsOnline(status.available);
      
      if (status.available) {
        // Generate real AI health report
        const userData = {
          medications: ["Metformin", "Ozempic"],
          metrics: { bloodSugar: 120, weight: 70 },
          adherence: 95
        };
        const aiReport = await OllamaService.generateHealthReport(userData);
        setReport(aiReport);
      } else {
        // Use fallback data
        setReport({
          adherence: "95%",
          trend: "Improving",
          insights: [
            "AI is currently offline. Using default insights.",
            "Medication adherence looks good based on recent data."
          ]
        });
      }
    } catch (error) {
      console.error('Error refreshing report:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <MedTrackCard>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Brain className="w-5 h-5 text-foreground" />
          <h3 className="text-lg font-semibold">AI Health Report</h3>
          {isOnline && (
            <span className="text-xs bg-medical-500 text-white px-2 py-1 rounded-full">
              🟢 AI Online
            </span>
          )}
        </div>
        <button 
          onClick={refreshReport} 
          disabled={isLoading}
          className="p-2 rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-md hover:shadow-lg"
          aria-label="Refresh report"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="text-center p-3 bg-secondary/50 rounded-lg">
          <p className="text-sm text-muted-foreground">Adherence</p>
          <p className="text-2xl font-bold text-foreground">{report.adherence}</p>
        </div>
        <div className="text-center p-3 bg-secondary/50 rounded-lg">
          <p className="text-sm text-muted-foreground">Trend</p>
          <p className="text-2xl font-bold text-foreground">{report.trend}</p>
        </div>
      </div>

      <div>
        <h4 className="text-sm font-medium text-foreground mb-3">Key Insights</h4>
        <div className="space-y-2">
          {report.insights.map((insight, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className="flex items-start gap-2.5 p-3 bg-secondary/50 rounded-lg"
            >
              <span className="text-foreground mt-0.5">•</span>
              <span className="text-sm text-muted-foreground">{insight}</span>
            </motion.div>
          ))}
        </div>
      </div>
    </MedTrackCard>
  );
};