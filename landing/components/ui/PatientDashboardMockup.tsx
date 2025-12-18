"use client";

import { Pill, Clock, TrendingUp, Heart, Activity, CheckCircle2 } from "lucide-react";

export default function PatientDashboardMockup() {
  const medications = [
    { name: "Metformin", dosage: "500mg", time: "8:00 AM", taken: true },
    { name: "Lisinopril", dosage: "10mg", time: "9:00 AM", taken: true },
    { name: "Atorvastatin", dosage: "20mg", time: "8:00 PM", taken: false },
  ];

  const metrics = [
    { label: "Blood Pressure", value: "120/80", icon: Heart, color: "text-red-500" },
    { label: "Glucose", value: "95 mg/dL", icon: Activity, color: "text-green-500" },
  ];

  return (
    <div className="w-full h-full bg-gradient-to-br from-gray-50 to-white rounded-2xl p-6 overflow-hidden shadow-inner">
      {/* Header */}
      <div className="flex items-center justify-between mb-5">
        <div>
          <h3 className="text-lg font-bold text-gray-900">Today&apos;s Medications</h3>
          <p className="text-sm text-gray-500">December 13, 2025</p>
        </div>
        <div className="flex items-center gap-2 px-3 py-1.5 bg-green-100 rounded-full shadow-sm">
          <CheckCircle2 className="w-4 h-4 text-green-600" />
          <span className="text-xs font-semibold text-green-700">2/3 Taken</span>
        </div>
      </div>

      {/* Medications Grid */}
      <div className="grid grid-cols-1 gap-2.5 mb-5">
        {medications.map((med, index) => (
          <div
            key={index}
            className={`p-3.5 rounded-xl border-2 transition-all shadow-sm ${
              med.taken
                ? "bg-green-50/80 border-green-200"
                : "bg-white border-blue-200 hover:border-blue-300 hover:shadow-md"
            }`}
          >
            <div className="flex items-start justify-between">
              <div className="flex items-start gap-3 flex-1">
                <div className={`p-2 rounded-lg shadow-sm ${
                  med.taken ? "bg-green-100" : "bg-blue-100"
                }`}>
                  <Pill className={`w-4 h-4 ${
                    med.taken ? "text-green-600" : "text-blue-600"
                  }`} />
                </div>
                <div className="flex-1 min-w-0">
                  <h4 className="font-semibold text-gray-900 text-sm mb-0.5">
                    {med.name}
                  </h4>
                  <p className="text-xs text-gray-600 mb-1.5">{med.dosage}</p>
                  <div className="flex items-center gap-1.5">
                    <Clock className="w-3 h-3 text-gray-400" />
                    <span className="text-xs text-gray-500">{med.time}</span>
                  </div>
                </div>
              </div>
              {med.taken && (
                <CheckCircle2 className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-2 gap-2.5 mb-4">
        {metrics.map((metric, index) => {
          const Icon = metric.icon;
          return (
            <div
              key={index}
              className="p-3 bg-white rounded-xl border border-gray-200 shadow-sm"
            >
              <div className="flex items-center gap-2 mb-1.5">
                <Icon className={`w-4 h-4 ${metric.color}`} />
                <span className="text-xs text-gray-600 font-medium">{metric.label}</span>
              </div>
              <p className="text-base font-bold text-gray-900">{metric.value}</p>
            </div>
          );
        })}
      </div>

      {/* Adherence Chart */}
      <div className="bg-white rounded-xl border border-gray-200 p-3 shadow-sm">
        <div className="flex items-center justify-between mb-2.5">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-blue-600" />
            <span className="text-xs font-semibold text-gray-700">Adherence This Week</span>
          </div>
          <span className="text-xs font-bold text-blue-600">94%</span>
        </div>
        <div className="flex items-end gap-1.5 h-14">
          {[85, 92, 88, 96, 94, 91, 94].map((height, index) => (
            <div
              key={index}
              className="flex-1 bg-gradient-to-t from-blue-500 to-blue-400 rounded-t shadow-sm"
              style={{ height: `${height}%` }}
            />
          ))}
        </div>
        <div className="flex justify-between mt-1.5">
          {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].map((day, index) => (
            <span key={index} className="text-[10px] text-gray-400 text-center flex-1">
              {day}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

