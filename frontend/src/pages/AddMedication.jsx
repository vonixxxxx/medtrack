import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../api';
import Navbar from '../components/Navbar';

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
      await api.post('/medications', form);
      navigate('/');
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div>
      <Navbar />
      <div className="max-w-lg mx-auto p-4 bg-white mt-6 shadow rounded">
        <h2 className="text-xl font-semibold mb-4">Add Medication</h2>
        <form onSubmit={handleSubmit} className="space-y-3">
          {['name', 'dosage', 'frequency'].map((field) => (
            <input
              key={field}
              name={field}
              placeholder={field.charAt(0).toUpperCase() + field.slice(1)}
              value={form[field]}
              onChange={handleChange}
              className="w-full p-2 border rounded"
              required
            />
          ))}
          <input
            type="date"
            name="startDate"
            value={form.startDate}
            onChange={handleChange}
            className="w-full p-2 border rounded"
            required
          />
          <input
            type="date"
            name="endDate"
            value={form.endDate}
            onChange={handleChange}
            className="w-full p-2 border rounded"
            required
          />
          <button className="w-full bg-blue-600 text-white py-2 rounded">
            Save
          </button>
        </form>
      </div>
    </div>
  );
}
