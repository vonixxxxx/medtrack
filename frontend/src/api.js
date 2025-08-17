import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000/api',
  timeout: 30000, // 30 seconds to tolerate cold starts
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export default api;

// Simple helpers for React Query
export const fetcher = (url) => api.get(url).then((res) => res.data);
export const poster = (url, body) => api.post(url, body).then((res) => res.data);
