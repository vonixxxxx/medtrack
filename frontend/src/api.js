import axios from 'axios';

// Use relative paths for Vercel deployment (same domain)
// Fallback to env var for local development with separate backend
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '/api',
  timeout: 10000, // 10 seconds - reduced timeout
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

// OpenEMR Feature API Methods

// Encounters
export const getEncounters = (params) => api.get('/encounters', { params }).then(res => {
  const data = res.data;
  return Array.isArray(data) ? data : [];
});
export const getEncounter = (id) => api.get(`/encounters/${id}`).then(res => res.data);
export const createEncounter = (data) => api.post('/encounters', data).then(res => res.data);
export const updateEncounter = (id, data) => api.put(`/encounters/${id}`, data).then(res => res.data);
export const deleteEncounter = (id) => api.delete(`/encounters/${id}`).then(res => res.data);

// SOAP Notes
export const getSoapNotes = (params) => api.get('/soap-notes', { params }).then(res => {
  const data = res.data;
  return Array.isArray(data) ? data : [];
});
export const getSoapNote = (id) => api.get(`/soap-notes/${id}`).then(res => res.data);
export const createSoapNote = (data) => api.post('/soap-notes', data).then(res => res.data);
export const updateSoapNote = (id, data) => api.put(`/soap-notes/${id}`, data).then(res => res.data);
export const deleteSoapNote = (id) => api.delete(`/soap-notes/${id}`).then(res => res.data);

// Problems
export const getProblems = (params) => api.get('/problems', { params }).then(res => {
  const data = res.data;
  return Array.isArray(data) ? data : [];
});
export const getProblem = (id) => api.get(`/problems/${id}`).then(res => res.data);
export const createProblem = (data) => api.post('/problems', data).then(res => res.data);
export const updateProblem = (id, data) => api.put(`/problems/${id}`, data).then(res => res.data);
export const deleteProblem = (id) => api.delete(`/problems/${id}`).then(res => res.data);

// Allergies
export const getAllergies = (params) => api.get('/allergies', { params }).then(res => {
  const data = res.data;
  return Array.isArray(data) ? data : [];
});
export const getAllergy = (id) => api.get(`/allergies/${id}`).then(res => res.data);
export const createAllergy = (data) => api.post('/allergies', data).then(res => res.data);
export const updateAllergy = (id, data) => api.put(`/allergies/${id}`, data).then(res => res.data);
export const deleteAllergy = (id) => api.delete(`/allergies/${id}`).then(res => res.data);

// Immunizations
export const getImmunizations = (params) => api.get('/immunizations', { params }).then(res => {
  // Ensure we always return an array
  const data = res.data;
  return Array.isArray(data) ? data : [];
});
export const getImmunization = (id) => api.get(`/immunizations/${id}`).then(res => res.data);
export const createImmunization = (data) => api.post('/immunizations', data).then(res => res.data);
export const updateImmunization = (id, data) => api.put(`/immunizations/${id}`, data).then(res => res.data);
export const deleteImmunization = (id) => api.delete(`/immunizations/${id}`).then(res => res.data);

// Prescriptions
export const getPrescriptions = (params) => api.get('/prescriptions', { params }).then(res => {
  // Ensure we always return an array
  const data = res.data;
  return Array.isArray(data) ? data : [];
});
export const getPrescription = (id) => api.get(`/prescriptions/${id}`).then(res => res.data);
export const createPrescription = (data) => api.post('/prescriptions', data).then(res => res.data);
export const updatePrescription = (id, data) => api.put(`/prescriptions/${id}`, data).then(res => res.data);
export const deletePrescription = (id) => api.delete(`/prescriptions/${id}`).then(res => res.data);

// Billing
export const getCharges = (params) => api.get('/billing/charges', { params }).then(res => {
  const data = res.data;
  return Array.isArray(data) ? data : [];
});
export const createCharge = (data) => api.post('/billing/charges', data).then(res => res.data);
export const updateCharge = (id, data) => api.put(`/billing/charges/${id}`, data).then(res => res.data);
export const getPayments = (params) => api.get('/billing/payments', { params }).then(res => {
  const data = res.data;
  return Array.isArray(data) ? data : [];
});
export const createPayment = (data) => api.post('/billing/payments', data).then(res => res.data);

// New Feature API Methods

// Drug Interactions
export const checkDrugInteractions = (data) => api.post('/drug-interactions/check', data).then(res => res.data);
export const getMedicationInteractions = (medicationId) => api.get(`/drug-interactions/medication/${medicationId}`).then(res => res.data);
export const addDrugInteraction = (data) => api.post('/drug-interactions', data).then(res => res.data);

// Side Effects
export const getSideEffects = (params) => api.get('/side-effects', { params }).then(res => {
  const data = res.data;
  return Array.isArray(data) ? data : [];
});
export const createSideEffect = (data) => api.post('/side-effects', data).then(res => res.data);
export const updateSideEffect = (id, data) => api.put(`/side-effects/${id}`, data).then(res => res.data);
export const deleteSideEffect = (id) => api.delete(`/side-effects/${id}`).then(res => res.data);

// Adherence
export const getAdherence = (params) => api.get('/adherence', { params }).then(res => {
  const data = res.data;
  // Handle both array and object responses
  if (Array.isArray(data)) {
    return { logs: data, statistics: null };
  }
  return data || { logs: [], statistics: null };
});
export const logAdherence = (data) => api.post('/adherence', data).then(res => res.data);
export const getAdherenceCalendar = (params) => api.get('/adherence/calendar', { params }).then(res => {
  const data = res.data;
  // Ensure we always return an object with calendar and statistics
  if (Array.isArray(data)) {
    return { calendar: {}, logs: data, statistics: null };
  }
  return data || { calendar: {}, logs: [], statistics: null };
});

// Patient Profiles (Multiple Patients)
export const getPatientProfiles = (params) => api.get('/patient-profiles', { params }).then(res => {
  const data = res.data;
  return Array.isArray(data) ? data : [];
});
export const createPatientProfile = (data) => api.post('/patient-profiles', data).then(res => res.data);
export const updatePatientProfile = (id, data) => api.put(`/patient-profiles/${id}`, data).then(res => res.data);
export const deletePatientProfile = (id) => api.delete(`/patient-profiles/${id}`).then(res => res.data);

// Diary
export const getDiaryEntries = (params) => api.get('/diary', { params }).then(res => {
  const data = res.data;
  return Array.isArray(data) ? data : [];
});
export const createDiaryEntry = (data) => api.post('/diary', data).then(res => res.data);
export const updateDiaryEntry = (id, data) => api.put(`/diary/${id}`, data).then(res => res.data);
export const deleteDiaryEntry = (id) => api.delete(`/diary/${id}`).then(res => res.data);

// Pill Recognition
export const recognizePill = (formData) => api.post('/pill-recognition/recognize', formData, {
  headers: { 'Content-Type': 'multipart/form-data' }
}).then(res => res.data);
export const getRecognitionHistory = (params) => api.get('/pill-recognition/history', { params }).then(res => {
  const data = res.data;
  return Array.isArray(data) ? data : [];
});
export const verifyRecognition = (id, data) => api.patch(`/pill-recognition/${id}/verify`, data).then(res => res.data);
export const addMedicationFromPill = (data) => api.post('/pill-recognition/add-medication', data).then(res => res.data);
export const getMedicationsWithWarnings = (params) => api.get('/medications/with-warnings', { params }).then(res => res.data);

// Confir-Med API endpoints
export const getMonopharmacySideEffects = (drugName) => api.get('/mono_se', { params: { drug_name: drugName } }).then(res => res.data);
export const getPolypharmacySideEffects = (drug1, drug2) => api.get('/poly_se', { params: { drug_1: drug1, drug_2: drug2 } }).then(res => res.data);
