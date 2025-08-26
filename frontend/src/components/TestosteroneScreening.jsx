import React, { useState } from 'react';
import { motion } from 'framer-motion';

const SYMPTOMS = [
  { value: 'low_libido', label: 'Low libido / reduced morning erections' },
  { value: 'fatigue', label: 'Fatigue / low energy' },
  { value: 'depressed_mood', label: 'Depressed mood / irritability' },
  { value: 'reduced_muscle_mass', label: 'Reduced muscle mass/strength' },
  { value: 'increased_body_fat', label: 'Increased body fat' },
  { value: 'reduced_shaving_frequency', label: 'Reduced shaving frequency/body hair' },
  { value: 'decreased_bone_strength', label: 'Decreased bone strength' }
];

export default function TestosteroneScreening({ onSubmit }) {
  const [symptoms, setSymptoms] = useState([]);
  const [notApplicable, setNotApplicable] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const toggleSymptom = (value) => {
    setSymptoms((prev) =>
      prev.includes(value) ? prev.filter((v) => v !== value) : [...prev, value]
    );
    if (notApplicable) setNotApplicable(false);
  };

  const submit = async () => {
    setIsSubmitting(true);
    try {
      const resp = await fetch('/api/screening/testosterone', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({ symptoms, notApplicable, completedAt: new Date().toISOString() })
      });
      const data = await resp.json();
      if (resp.ok) {
        onSubmit?.({ symptoms, notApplicable, server: data });
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="max-w-xl mx-auto p-6 bg-white rounded-xl shadow-lg">
      <h2 className="text-xl font-semibold text-gray-800 mb-2">Low Testosterone Screening</h2>
      <p className="text-sm text-gray-600 mb-4">Select all symptoms that apply or choose N/A.</p>

      <div className="space-y-2 mb-4">
        {SYMPTOMS.map((s) => (
          <label key={s.value} className="flex items-center">
            <input
              type="checkbox"
              checked={symptoms.includes(s.value)}
              onChange={() => toggleSymptom(s.value)}
              className="mr-2"
            />
            <span>{s.label}</span>
          </label>
        ))}
      </div>

      <label className="flex items-center mb-4">
        <input type="checkbox" checked={notApplicable} onChange={(e) => setNotApplicable(e.target.checked)} className="mr-2" />
        <span className="text-gray-600 italic">N/A (Not Applicable)</span>
      </label>

      <button onClick={submit} disabled={isSubmitting} className="w-full bg-gray-800 text-white py-2 rounded-md disabled:opacity-50">
        {isSubmitting ? 'Submitting...' : 'Submit'}
      </button>
    </motion.div>
  );
}
