import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { User, Mail, Lock, Eye, EyeOff, Calendar, Phone, MapPin, Heart } from 'lucide-react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Card, CardContent } from './ui/card';
import api from '../api';

const RegistrationForm = ({ onComplete }) => {
  const [currentStep, setCurrentStep] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  const [formData, setFormData] = useState({
    // Step 1: Basic Info
    email: '',
    password: '',
    confirmPassword: '',
    
    // Step 2: Personal Info
    firstName: '',
    lastName: '',
    dateOfBirth: '',
    phone: '',
    
    // Step 3: Health Info
    height: '',
    weight: '',
    bloodType: '',
    allergies: '',
    medicalConditions: '',
    
    // Step 4: Emergency Contact
    emergencyContactName: '',
    emergencyContactPhone: '',
    emergencyContactRelation: '',
    
    // Step 5: Preferences
    medicationReminders: true,
    healthMetricsReminders: true,
    weeklyReports: true,
    dataSharing: false
  });

  const steps = [
    { number: 1, title: 'Account Setup', description: 'Create your secure account' },
    { number: 2, title: 'Personal Info', description: 'Tell us about yourself' },
    { number: 3, title: 'Health Profile', description: 'Your health information' },
    { number: 4, title: 'Emergency Contact', description: 'Safety first' },
    { number: 5, title: 'Preferences', description: 'Customize your experience' }
  ];

  const handleInputChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    if (error) setError('');
  };

  const validateStep = (step) => {
    switch (step) {
      case 1:
        if (!formData.email || !formData.password || !formData.confirmPassword) {
          setError('All fields are required');
          return false;
        }
        if (formData.password !== formData.confirmPassword) {
          setError('Passwords do not match');
          return false;
        }
        if (formData.password.length < 6) {
          setError('Password must be at least 6 characters');
          return false;
        }
        return true;
      
      case 2:
        if (!formData.firstName || !formData.lastName || !formData.dateOfBirth) {
          setError('First name, last name, and date of birth are required');
          return false;
        }
        return true;
      
      case 3:
        return true; // Health info is optional
      
      case 4:
        if (!formData.emergencyContactName || !formData.emergencyContactPhone) {
          setError('Emergency contact name and phone are required');
          return false;
        }
        return true;
      
      case 5:
        return true; // Preferences are optional
      
      default:
        return true;
    }
  };

  const handleNext = () => {
    if (validateStep(currentStep)) {
      setCurrentStep(prev => prev + 1);
    }
  };

  const handlePrevious = () => {
    setCurrentStep(prev => prev - 1);
  };

  const handleSubmit = async () => {
    if (!validateStep(currentStep)) return;
    
    setIsLoading(true);
    setError('');

    try {
      // Create user account
      const { data: authData } = await api.post('auth/signup', {
        email: formData.email,
        password: formData.password
      });

      // Store token
      localStorage.setItem('token', authData.token);

      // Create user profile
      const userProfile = {
        firstName: formData.firstName,
        lastName: formData.lastName,
        dateOfBirth: formData.dateOfBirth,
        phone: formData.phone,
        height: formData.height ? parseFloat(formData.height) : null,
        weight: formData.weight ? parseFloat(formData.weight) : null,
        bloodType: formData.bloodType,
        allergies: formData.allergies,
        medicalConditions: formData.medicalConditions,
        emergencyContactName: formData.emergencyContactName,
        emergencyContactPhone: formData.emergencyContactPhone,
        emergencyContactRelation: formData.emergencyContactRelation,
        medicationReminders: formData.medicationReminders,
        healthMetricsReminders: formData.healthMetricsReminders,
        weeklyReports: formData.weeklyReports,
        dataSharing: formData.dataSharing
      };

      // Store user data
      localStorage.setItem('user', JSON.stringify(userProfile));

      // Call completion handler
      onComplete({
        ...authData,
        profile: userProfile
      });

    } catch (err) {
      console.error('Registration error:', err);
      setError(err.response?.data?.error || 'Registration failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const renderStepContent = () => {
    switch (currentStep) {
      case 1:
        return (
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-white mb-2">
                Email Address *
              </label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <Input
                  type="email"
                  placeholder="Enter your email"
                  value={formData.email}
                  onChange={(e) => handleInputChange('email', e.target.value)}
                  className="pl-10 bg-gray-900 border-gray-800 text-white placeholder-gray-400 focus:border-white focus:ring-1 focus:ring-white"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-white mb-2">
                Password *
              </label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <Input
                  type={showPassword ? 'text' : 'password'}
                  placeholder="Create a password"
                  value={formData.password}
                  onChange={(e) => handleInputChange('password', e.target.value)}
                  className="pl-10 pr-10 bg-gray-900 border-gray-800 text-white placeholder-gray-400 focus:border-white focus:ring-1 focus:ring-white"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-white"
                >
                  {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-white mb-2">
                Confirm Password *
              </label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <Input
                  type={showConfirmPassword ? 'text' : 'password'}
                  placeholder="Confirm your password"
                  value={formData.confirmPassword}
                  onChange={(e) => handleInputChange('confirmPassword', e.target.value)}
                  className="pl-10 pr-10 bg-gray-900 border-gray-800 text-white placeholder-gray-400 focus:border-white focus:ring-1 focus:ring-white"
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-white"
                >
                  {showConfirmPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>
            </div>
          </div>
        );

      case 2:
        return (
          <div className="space-y-6">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-white mb-2">
                  First Name *
                </label>
                <div className="relative">
                  <User className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                  <Input
                    type="text"
                    placeholder="First name"
                    value={formData.firstName}
                    onChange={(e) => handleInputChange('firstName', e.target.value)}
                    className="pl-10 bg-gray-900 border-gray-800 text-white placeholder-gray-400 focus:border-white focus:ring-1 focus:ring-white"
                  />
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-white mb-2">
                  Last Name *
                </label>
                <div className="relative">
                  <User className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                  <Input
                    type="text"
                    placeholder="Last name"
                    value={formData.lastName}
                    onChange={(e) => handleInputChange('lastName', e.target.value)}
                    className="pl-10 bg-gray-900 border-gray-800 text-white placeholder-gray-400 focus:border-white focus:ring-1 focus:ring-white"
                  />
                </div>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-white mb-2">
                Date of Birth *
              </label>
              <div className="relative">
                <Calendar className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <Input
                  type="date"
                  value={formData.dateOfBirth}
                  onChange={(e) => handleInputChange('dateOfBirth', e.target.value)}
                  className="pl-10 bg-gray-900 border-gray-800 text-white focus:border-white focus:ring-1 focus:ring-white"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-white mb-2">
                Phone Number
              </label>
              <div className="relative">
                <Phone className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <Input
                  type="tel"
                  placeholder="Phone number (optional)"
                  value={formData.phone}
                  onChange={(e) => handleInputChange('phone', e.target.value)}
                  className="pl-10 bg-gray-900 border-gray-800 text-white placeholder-gray-400 focus:border-white focus:ring-1 focus:ring-white"
                />
              </div>
            </div>
          </div>
        );

      case 3:
        return (
          <div className="space-y-6">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-white mb-2">
                  Height (cm)
                </label>
                <Input
                  type="number"
                  placeholder="Height"
                  value={formData.height}
                  onChange={(e) => handleInputChange('height', e.target.value)}
                  className="bg-gray-900 border-gray-800 text-white placeholder-gray-400 focus:border-white focus:ring-1 focus:ring-white"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-white mb-2">
                  Weight (kg)
                </label>
                <Input
                  type="number"
                  placeholder="Weight"
                  value={formData.weight}
                  onChange={(e) => handleInputChange('weight', e.target.value)}
                  className="bg-gray-900 border-gray-800 text-white placeholder-gray-400 focus:border-white focus:ring-1 focus:ring-white"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-white mb-2">
                Blood Type
              </label>
              <select
                value={formData.bloodType}
                onChange={(e) => handleInputChange('bloodType', e.target.value)}
                className="w-full px-3 py-2 bg-gray-900 border border-gray-800 rounded-xl text-white focus:border-white focus:ring-1 focus:ring-white focus:outline-none"
              >
                <option value="">Select blood type</option>
                <option value="A+">A+</option>
                <option value="A-">A-</option>
                <option value="B+">B+</option>
                <option value="B-">B-</option>
                <option value="AB+">AB+</option>
                <option value="AB-">AB-</option>
                <option value="O+">O+</option>
                <option value="O-">O-</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-white mb-2">
                Allergies
              </label>
              <Input
                type="text"
                placeholder="List any allergies (optional)"
                value={formData.allergies}
                onChange={(e) => handleInputChange('allergies', e.target.value)}
                className="bg-gray-900 border-gray-800 text-white placeholder-gray-400 focus:border-white focus:ring-1 focus:ring-white"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-white mb-2">
                Medical Conditions
              </label>
              <Input
                type="text"
                placeholder="List any medical conditions (optional)"
                value={formData.medicalConditions}
                onChange={(e) => handleInputChange('medicalConditions', e.target.value)}
                className="bg-gray-900 border-gray-800 text-white placeholder-gray-400 focus:border-white focus:ring-1 focus:ring-white"
              />
            </div>
          </div>
        );

      case 4:
        return (
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-white mb-2">
                Emergency Contact Name *
              </label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <Input
                  type="text"
                  placeholder="Emergency contact name"
                  value={formData.emergencyContactName}
                  onChange={(e) => handleInputChange('emergencyContactName', e.target.value)}
                  className="pl-10 bg-gray-900 border-gray-800 text-white placeholder-gray-400 focus:border-white focus:ring-1 focus:ring-white"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-white mb-2">
                Emergency Contact Phone *
              </label>
              <div className="relative">
                <Phone className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <Input
                  type="tel"
                  placeholder="Emergency contact phone"
                  value={formData.emergencyContactPhone}
                  onChange={(e) => handleInputChange('emergencyContactPhone', e.target.value)}
                  className="pl-10 bg-gray-900 border-gray-800 text-white placeholder-gray-400 focus:border-white focus:ring-1 focus:ring-white"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-white mb-2">
                Relationship
              </label>
              <select
                value={formData.emergencyContactRelation}
                onChange={(e) => handleInputChange('emergencyContactRelation', e.target.value)}
                className="w-full px-3 py-2 bg-gray-900 border border-gray-800 rounded-xl text-white focus:border-white focus:ring-1 focus:ring-white focus:outline-none"
              >
                <option value="">Select relationship</option>
                <option value="spouse">Spouse</option>
                <option value="parent">Parent</option>
                <option value="child">Child</option>
                <option value="sibling">Sibling</option>
                <option value="friend">Friend</option>
                <option value="other">Other</option>
              </select>
            </div>
          </div>
        );

      case 5:
        return (
          <div className="space-y-6">
            <div className="space-y-4">
              <h3 className="text-lg font-medium text-white">Notification Preferences</h3>
              
              <div className="flex items-center justify-between p-4 bg-gray-900 rounded-xl border border-gray-800">
                <div>
                  <h4 className="text-white font-medium">Medication Reminders</h4>
                  <p className="text-sm text-gray-400">Get reminded to take your medications</p>
                </div>
                <input
                  type="checkbox"
                  checked={formData.medicationReminders}
                  onChange={(e) => handleInputChange('medicationReminders', e.target.checked)}
                  className="w-5 h-5 text-white bg-gray-800 border-gray-600 rounded focus:ring-white focus:ring-2"
                />
              </div>

              <div className="flex items-center justify-between p-4 bg-gray-900 rounded-xl border border-gray-800">
                <div>
                  <h4 className="text-white font-medium">Health Metrics Reminders</h4>
                  <p className="text-sm text-gray-400">Get reminded to log your health metrics</p>
                </div>
                <input
                  type="checkbox"
                  checked={formData.healthMetricsReminders}
                  onChange={(e) => handleInputChange('healthMetricsReminders', e.target.checked)}
                  className="w-5 h-5 text-white bg-gray-800 border-gray-600 rounded focus:ring-white focus:ring-2"
                />
              </div>

              <div className="flex items-center justify-between p-4 bg-gray-900 rounded-xl border border-gray-800">
                <div>
                  <h4 className="text-white font-medium">Weekly Health Reports</h4>
                  <p className="text-sm text-gray-400">Receive weekly summaries of your health data</p>
                </div>
                <input
                  type="checkbox"
                  checked={formData.weeklyReports}
                  onChange={(e) => handleInputChange('weeklyReports', e.target.checked)}
                  className="w-5 h-5 text-white bg-gray-800 border-gray-600 rounded focus:ring-white focus:ring-2"
                />
              </div>

              <div className="flex items-center justify-between p-4 bg-gray-900 rounded-xl border border-gray-800">
                <div>
                  <h4 className="text-white font-medium">Data Sharing</h4>
                  <p className="text-sm text-gray-400">Allow anonymous data sharing for research</p>
                </div>
                <input
                  type="checkbox"
                  checked={formData.dataSharing}
                  onChange={(e) => handleInputChange('dataSharing', e.target.checked)}
                  className="w-5 h-5 text-white bg-gray-800 border-gray-600 rounded focus:ring-white focus:ring-2"
                />
              </div>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-black flex items-center justify-center p-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-2xl"
      >
        {/* Header */}
        <div className="text-center mb-8">
          <div className="w-16 h-16 rounded-full bg-white flex items-center justify-center mx-auto mb-4">
            <Heart className="w-8 h-8 text-black" />
          </div>
          <h1 className="text-3xl font-bold text-white mb-2">MedTrack</h1>
          <p className="text-gray-400">Your Health, Tracked Simply</p>
        </div>

        {/* Progress Steps */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            {steps.map((step, index) => (
              <div key={step.number} className="flex items-center">
                <div className={`
                  w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium
                  ${currentStep >= step.number 
                    ? 'bg-white text-black' 
                    : 'bg-gray-800 text-gray-400'
                  }
                `}>
                  {step.number}
                </div>
                {index < steps.length - 1 && (
                  <div className={`
                    flex-1 h-0.5 mx-2
                    ${currentStep > step.number ? 'bg-white' : 'bg-gray-800'}
                  `} />
                )}
              </div>
            ))}
          </div>
          <div className="text-center">
            <h2 className="text-xl font-semibold text-white">
              {steps[currentStep - 1].title}
            </h2>
            <p className="text-gray-400 text-sm">
              {steps[currentStep - 1].description}
            </p>
          </div>
        </div>

        {/* Form Card */}
        <Card className="bg-gray-900 border-gray-800">
          <CardContent className="p-8">
            {error && (
              <div className="mb-6 p-4 bg-red-900/20 border border-red-800 rounded-xl text-red-400 text-sm">
                {error}
              </div>
            )}

            {renderStepContent()}

            {/* Navigation Buttons */}
            <div className="flex justify-between mt-8">
              <Button
                onClick={handlePrevious}
                disabled={currentStep === 1}
                className="px-6 py-3 bg-gray-800 hover:bg-gray-700 text-white rounded-xl font-medium transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Previous
              </Button>

              {currentStep < 5 ? (
                <Button
                  onClick={handleNext}
                  className="px-6 py-3 bg-white hover:bg-gray-200 text-black rounded-xl font-medium transition-all duration-200"
                >
                  Next
                </Button>
              ) : (
                <Button
                  onClick={handleSubmit}
                  disabled={isLoading}
                  className="px-6 py-3 bg-white hover:bg-gray-200 text-black rounded-xl font-medium transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isLoading ? (
                    <div className="flex items-center">
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-black mr-2"></div>
                      Creating Account...
                    </div>
                  ) : (
                    'Create Account'
                  )}
                </Button>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Footer */}
        <div className="text-center mt-6">
          <p className="text-gray-400 text-sm">
            Already have an account?{' '}
            <a 
              href="/login" 
              className="text-white hover:text-gray-300 font-medium transition-colors"
            >
              Sign in
            </a>
          </p>
        </div>
      </motion.div>
    </div>
  );
};

export default RegistrationForm;




