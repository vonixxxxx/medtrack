import React, { useState } from 'react';
import { motion } from 'framer-motion';

export default function IPAQScreening({ onSubmit }) {
  const [vigorousDays, setVigorousDays] = useState('');
  const [vigorousMinutes, setVigorousMinutes] = useState('');
  const [moderateDays, setModerateDays] = useState('');
  const [moderateMinutes, setModerateMinutes] = useState('');
  const [walkingDays, setWalkingDays] = useState('');
  const [walkingMinutes, setWalkingMinutes] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const calculate = () => {
    const vd = parseInt(vigorousDays) || 0;
    const vm = parseInt(vigorousMinutes) || 0;
    const md = parseInt(moderateDays) || 0;
    const mm = parseInt(moderateMinutes) || 0;
    const wd = parseInt(walkingDays) || 0;
    const wm = parseInt(walkingMinutes) || 0;

    const vigorousMET = vd * vm * 8.0;
    const moderateMET = md * mm * 4.0;
    const walkingMET = wd * wm * 3.3;
    const totalMET = Math.round(vigorousMET + moderateMET + walkingMET);

    let category = 'low_activity';
    if (totalMET >= 3000) category = 'high_activity';
    else if (totalMET >= 600) category = 'moderate_activity';

    return { totalMET, category };
  };

  const submit = async () => {
    setSubmitting(true);
    try {
      const result = calculate();
      onSubmit?.({
        totalMET: result.totalMET,
        category: result.category,
        completedAt: new Date().toISOString()
      });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="max-w-xl mx-auto p-6 bg-white rounded-xl border border-gray-200">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">IPAQ (International Physical Activity Questionnaire)</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm text-gray-700 mb-1">Vigorous activity days (last 7 days)</label>
          <input type="number" min="0" max="7" value={vigorousDays} onChange={e=>setVigorousDays(e.target.value)} className="w-full border border-gray-300 rounded-md px-3 py-2" />
        </div>
        <div>
          <label className="block text-sm text-gray-700 mb-1">Vigorous minutes per day</label>
          <input type="number" min="0" max="300" value={vigorousMinutes} onChange={e=>setVigorousMinutes(e.target.value)} className="w-full border border-gray-300 rounded-md px-3 py-2" />
        </div>
        <div>
          <label className="block text-sm text-gray-700 mb-1">Moderate activity days</label>
          <input type="number" min="0" max="7" value={moderateDays} onChange={e=>setModerateDays(e.target.value)} className="w-full border border-gray-300 rounded-md px-3 py-2" />
        </div>
        <div>
          <label className="block text-sm text-gray-700 mb-1">Moderate minutes per day</label>
          <input type="number" min="0" max="300" value={moderateMinutes} onChange={e=>setModerateMinutes(e.target.value)} className="w-full border border-gray-300 rounded-md px-3 py-2" />
        </div>
        <div>
          <label className="block text-sm text-gray-700 mb-1">Walking days</label>
          <input type="number" min="0" max="7" value={walkingDays} onChange={e=>setWalkingDays(e.target.value)} className="w-full border border-gray-300 rounded-md px-3 py-2" />
        </div>
        <div>
          <label className="block text-sm text-gray-700 mb-1">Walking minutes per day</label>
          <input type="number" min="0" max="300" value={walkingMinutes} onChange={e=>setWalkingMinutes(e.target.value)} className="w-full border border-gray-300 rounded-md px-3 py-2" />
        </div>
      </div>
      <div className="mt-4 flex justify-end">
        <button onClick={submit} disabled={submitting} className="px-4 py-2 rounded-md bg-gradient-to-r from-blue-600 to-indigo-600 text-white hover:from-blue-700 hover:to-indigo-700 disabled:opacity-50">
          {submitting ? 'Calculating...' : 'Calculate'}
        </button>
      </div>
    </motion.div>
  );
}
