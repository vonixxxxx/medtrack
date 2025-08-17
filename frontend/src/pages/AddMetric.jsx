import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../api';
import Navbar from '../components/Navbar';

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
      navigate('/');
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div>
      <Navbar />
      <div className="max-w-lg mx-auto p-4 bg-white mt-6 shadow rounded">
        <h2 className="text-xl font-semibold mb-4">Add Metric</h2>
        <form onSubmit={handleSubmit} className="space-y-3">
          <input
            name="weight"
            placeholder="Weight (kg)"
            value={form.weight}
            onChange={handleChange}
            className="w-full p-2 border rounded"
            required
          />
          <input
            name="height"
            placeholder="Height (m)"
            value={form.height}
            onChange={handleChange}
            className="w-full p-2 border rounded"
            required
          />
          <input
            name="bloodPressure"
            placeholder="Blood Pressure (e.g., 120/80)"
            value={form.bloodPressure}
            onChange={handleChange}
            className="w-full p-2 border rounded"
            required
          />
          <input
            name="hipCircumference"
            placeholder="Hip Circumference (optional)"
            value={form.hipCircumference}
            onChange={handleChange}
            className="w-full p-2 border rounded"
          />
          <button className="w-full bg-blue-600 text-white py-2 rounded">
            Save
          </button>
        </form>
      </div>
    </div>
  );
}
