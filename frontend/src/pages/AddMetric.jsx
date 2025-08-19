import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import api from '../api';
import MobileNavigation from '../components/MobileNavigation';

export default function AddMetric() {
  const navigate = useNavigate();
  const [form, setForm] = useState({
    weight: '',
    height: '',
    bloodPressure: '',
    hipCircumference: '',
  });

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await api.post('/metrics', {
        ...form,
        weight: parseFloat(form.weight),
        height: parseFloat(form.height),
        hipCircumference: form.hipCircumference ? parseFloat(form.hipCircumference) : undefined,
      });
      navigate('/dashboard');
    } catch (err) {
      console.error(err);
    }
  };

  const metricFields = [
    { name: 'weight', placeholder: 'Weight (kg)', required: true },
    { name: 'height', placeholder: 'Height (m)', required: true },
    { name: 'bloodPressure', placeholder: 'Blood Pressure (e.g., 120/80)', required: true },
    { name: 'hipCircumference', placeholder: 'Hip Circumference (optional)', required: false }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      <MobileNavigation />
      <div className="max-w-2xl mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8">
        <motion.div 
          className="bg-white rounded-2xl sm:rounded-3xl shadow-xl p-6 sm:p-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <motion.div 
            className="mb-8"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.1, duration: 0.3 }}
          >
            <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold text-gray-900 mb-2">Add Health Metric</h1>
            <p className="text-gray-600 text-sm sm:text-base">Record your health measurements</p>
          </motion.div>

          <motion.form 
            onSubmit={handleSubmit} 
            className="space-y-6"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2, duration: 0.3 }}
          >
            {metricFields.map((field, index) => (
              <motion.div
                key={field.name}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.3 + index * 0.1, duration: 0.3 }}
              >
                <label className="block text-sm font-medium text-gray-700 mb-2 capitalize">
                  {field.name.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                  {!field.required && <span className="text-gray-400 text-xs ml-1">(optional)</span>}
                </label>
                <input
                  name={field.name}
                  placeholder={field.placeholder}
                  value={form[field.name]}
                  onChange={handleChange}
                  className="w-full p-4 border border-gray-300 rounded-xl focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-all duration-200 text-base"
                  required={field.required}
                />
              </motion.div>
            ))}
            
            <motion.button 
              type="submit"
              className="w-full bg-gradient-to-r from-emerald-600 to-teal-600 text-white py-4 px-6 rounded-xl font-medium hover:from-emerald-700 hover:to-teal-700 transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-[1.02] focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.7, duration: 0.3 }}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              Save Metric
            </motion.button>
          </motion.form>
        </motion.div>
      </div>
    </div>
  );
}
