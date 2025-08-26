import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import api from '../api';
import MobileNavigation from '../components/MobileNavigation';

export default function AddMedication() {
  const navigate = useNavigate();
  const [form, setForm] = useState({
    name: '',
    startDate: '',
    endDate: '',
    dosage: '',
    frequency: '',
  });

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await api.post('/api/medications', form);
      navigate('/dashboard');
    } catch (err) {
      console.error(err);
    }
  };

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
            <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold text-gray-900 mb-2">Add Medication</h1>
            <p className="text-gray-600 text-sm sm:text-base">Start tracking a new medication cycle</p>
          </motion.div>

          <motion.form 
            onSubmit={handleSubmit} 
            className="space-y-6"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2, duration: 0.3 }}
          >
            {['name', 'dosage', 'frequency'].map((field, index) => (
              <motion.div
                key={field}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.3 + index * 0.1, duration: 0.3 }}
              >
                <label className="block text-sm font-medium text-gray-700 mb-2 capitalize">
                  {field}
                </label>
                <input
                  name={field}
                  placeholder={`Enter ${field}`}
                  value={form[field]}
                  onChange={handleChange}
                  className="w-full p-4 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 text-base"
                  required
                />
              </motion.div>
            ))}
            
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.6, duration: 0.3 }}
            >
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Start Date
              </label>
              <input
                type="date"
                name="startDate"
                value={form.startDate}
                onChange={handleChange}
                className="w-full p-4 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 text-base"
                required
              />
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.7, duration: 0.3 }}
            >
              <label className="block text-sm font-medium text-gray-700 mb-2">
                End Date
              </label>
              <input
                type="date"
                name="endDate"
                value={form.endDate}
                onChange={handleChange}
                className="w-full p-4 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 text-base"
                required
              />
            </motion.div>
            
            <motion.button 
              type="submit"
              className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white py-4 px-6 rounded-xl font-medium hover:from-blue-700 hover:to-indigo-700 transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-[1.02] focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.8, duration: 0.3 }}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              Save Medication
            </motion.button>
          </motion.form>
        </motion.div>
      </div>
    </div>
  );
}
