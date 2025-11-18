import { useState, useEffect } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import { Users, Plus, User, ChevronDown } from 'lucide-react';
import { Button } from '../ui/button';
import { getPatientProfiles, createPatientProfile } from '../../api';
import { useAuth } from '../../contexts/AuthContext';

export const PatientProfileSwitcher = ({ onProfileChange, selectedProfileId }) => {
  const prefersReducedMotion = useReducedMotion();
  const { user } = useAuth();
  const [profiles, setProfiles] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [isFormOpen, setIsFormOpen] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    relationship: 'self',
    color: '#3B82F6'
  });

  useEffect(() => {
    if (user?.id) {
      loadProfiles();
    }
  }, [user]);

  const loadProfiles = async () => {
    try {
      setIsLoading(true);
      const data = await getPatientProfiles({ userId: user.id });
      // Ensure data is always an array
      const profilesArray = Array.isArray(data) ? data : [];
      setProfiles(profilesArray);
      
      // Auto-select primary profile or first profile
      if (profilesArray.length > 0 && !selectedProfileId) {
        const primary = profilesArray.find(p => p.isPrimary) || profilesArray[0];
        if (onProfileChange) {
          onProfileChange(primary);
        }
      }
    } catch (error) {
      console.error('Error loading patient profiles:', error);
      setProfiles([]); // Set empty array on error
    } finally {
      setIsLoading(false);
    }
  };

  const handleCreateProfile = async (e) => {
    e.preventDefault();
    try {
      // For now, we'll create a profile with the current user's patient ID
      // In a real app, you'd allow selecting an existing patient or creating a new one
      const patientId = user.patientId || user.id; // Fallback to user ID
      
      await createPatientProfile({
        userId: user.id,
        patientId: patientId,
        ...formData
      });
      setIsFormOpen(false);
      setFormData({ name: '', relationship: 'self', color: '#3B82F6' });
      loadProfiles();
    } catch (error) {
      console.error('Error creating patient profile:', error);
    }
  };

  const handleProfileSelect = (profile) => {
    if (onProfileChange) {
      onProfileChange(profile);
    }
    setIsDropdownOpen(false);
  };

  const selectedProfile = Array.isArray(profiles) 
    ? (profiles.find(p => p.id === selectedProfileId) || profiles.find(p => p.isPrimary) || profiles[0])
    : null;

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 px-4 py-2 bg-white rounded-xl border border-neutral-200">
        <div className="w-8 h-8 rounded-full bg-neutral-200 animate-pulse" />
        <div className="w-24 h-4 bg-neutral-200 rounded animate-pulse" />
      </div>
    );
  }

  // Ensure profiles is always an array
  const profilesArray = Array.isArray(profiles) ? profiles : [];

  return (
    <div className="relative">
      {/* Profile Switcher Button */}
      <Button
        onClick={() => setIsDropdownOpen(!isDropdownOpen)}
        variant="ghost"
        size="md"
        className="flex items-center gap-2"
      >
        <div
          className="w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-semibold"
          style={{ backgroundColor: selectedProfile?.color || '#3B82F6' }}
        >
          {selectedProfile?.name?.charAt(0)?.toUpperCase() || <User className="w-4 h-4" />}
        </div>
        <span className="font-medium text-neutral-900">
          {selectedProfile?.name || 'Select Profile'}
        </span>
        <ChevronDown className={`w-4 h-4 transition-transform ${isDropdownOpen ? 'rotate-180' : ''}`} />
      </Button>

      {/* Dropdown */}
      {isDropdownOpen && (
        <>
          <div
            className="fixed inset-0 z-10"
            onClick={() => setIsDropdownOpen(false)}
          />
          <motion.div
            initial={{ opacity: 0, y: prefersReducedMotion ? 0 : -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="absolute top-full left-0 mt-2 w-64 bg-white rounded-2xl border border-neutral-200 shadow-large z-20"
          >
            <div className="p-2">
              {profilesArray.length === 0 ? (
                <div className="p-4 text-center text-sm text-neutral-600">
                  No profiles yet. Create one to get started.
                </div>
              ) : (
                profilesArray.map((profile) => (
                  <button
                    key={profile.id}
                    onClick={() => handleProfileSelect(profile)}
                    className={`w-full flex items-center gap-3 px-3 py-2 rounded-xl transition-all ${
                      selectedProfileId === profile.id
                        ? 'bg-primary-50 border border-primary-200'
                        : 'hover:bg-neutral-50'
                    }`}
                  >
                  <div
                    className="w-10 h-10 rounded-full flex items-center justify-center text-white text-sm font-semibold flex-shrink-0"
                    style={{ backgroundColor: profile.color || '#3B82F6' }}
                  >
                    {profile.name?.charAt(0)?.toUpperCase() || <User className="w-5 h-5" />}
                  </div>
                  <div className="flex-1 text-left">
                    <div className="font-medium text-neutral-900">{profile.name}</div>
                    {profile.relationship && (
                      <div className="text-xs text-neutral-600 capitalize">
                        {profile.relationship}
                      </div>
                    )}
                  </div>
                  {profile.isPrimary && (
                    <span className="text-xs text-primary-600 font-medium">Primary</span>
                  )}
                  </button>
                ))
              )}
              
              <div className="border-t border-neutral-200 mt-2 pt-2">
                <button
                  onClick={() => {
                    setIsFormOpen(true);
                    setIsDropdownOpen(false);
                  }}
                  className="w-full flex items-center gap-3 px-3 py-2 rounded-xl hover:bg-neutral-50 transition-all"
                >
                  <div className="w-10 h-10 rounded-full bg-neutral-100 flex items-center justify-center">
                    <Plus className="w-5 h-5 text-neutral-600" />
                  </div>
                  <span className="font-medium text-neutral-900">Add Profile</span>
                </button>
              </div>
            </div>
          </motion.div>
        </>
      )}

      {/* Add Profile Form Modal */}
      {isFormOpen && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-white rounded-2xl p-6 max-w-md w-full"
          >
            <h3 className="text-xl font-semibold text-neutral-900 mb-4">
              Add Patient Profile
            </h3>
            <form onSubmit={handleCreateProfile} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-neutral-700 mb-2">
                  Name *
                </label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  className="w-full h-11 rounded-md border border-neutral-200 px-4 text-base focus:outline-none focus:ring-2 focus:ring-primary-500"
                  placeholder="e.g., John Doe, Mom, Child"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-neutral-700 mb-2">
                  Relationship
                </label>
                <select
                  value={formData.relationship}
                  onChange={(e) => setFormData({ ...formData, relationship: e.target.value })}
                  className="w-full h-11 rounded-md border border-neutral-200 px-4 text-base focus:outline-none focus:ring-2 focus:ring-primary-500"
                >
                  <option value="self">Self</option>
                  <option value="spouse">Spouse</option>
                  <option value="child">Child</option>
                  <option value="parent">Parent</option>
                  <option value="other">Other</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-neutral-700 mb-2">
                  Color
                </label>
                <input
                  type="color"
                  value={formData.color}
                  onChange={(e) => setFormData({ ...formData, color: e.target.value })}
                  className="w-full h-11 rounded-md border border-neutral-200"
                />
              </div>
              <div className="flex items-center gap-3 pt-4">
                <Button type="submit" variant="primary" size="md" className="flex-1">
                  Create Profile
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="md"
                  onClick={() => setIsFormOpen(false)}
                >
                  Cancel
                </Button>
              </div>
            </form>
          </motion.div>
        </div>
      )}
    </div>
  );
};

