import { useState, useRef } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import { Camera, Upload, CheckCircle, XCircle, Image as ImageIcon, AlertTriangle } from 'lucide-react';
import DashboardCard from '../DashboardCard';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { recognizePill, getRecognitionHistory, verifyRecognition, addMedicationFromPill } from '../../api';
import { useAuth } from '../../contexts/AuthContext';

export const PillRecognition = ({ patientId }) => {
  const prefersReducedMotion = useReducedMotion();
  const { user } = useAuth();
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [preview, setPreview] = useState(null);
  const [history, setHistory] = useState([]);
  const fileInputRef = useRef(null);

  const handleFileSelect = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      if (file.size > 10 * 1024 * 1024) {
        setError('File size must be less than 10MB');
        return;
      }
      
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleRecognize = async () => {
    const file = fileInputRef.current?.files?.[0];
    if (!file) {
      setError('Please select an image first');
      return;
    }

    setIsProcessing(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('image', file);
      formData.append('userId', user.id);
      if (patientId) formData.append('patientId', patientId);

      const recognition = await recognizePill(formData);
      setResult(recognition);
      
      // Show interaction warnings if any
      if (recognition.interactions && recognition.interactions.length > 0) {
        // Interactions are included in the response
        console.log('Interactions detected:', recognition.interactions);
      }
      
      loadHistory();
    } catch (err) {
      console.error('Error recognizing pill:', err);
      setError('Failed to recognize pill. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const loadHistory = async () => {
    try {
      const params = { userId: user.id };
      if (patientId) params.patientId = patientId;
      const data = await getRecognitionHistory(params);
      // Ensure data is always an array
      const historyArray = Array.isArray(data) ? data : [];
      setHistory(historyArray);
    } catch (error) {
      console.error('Error loading recognition history:', error);
      setHistory([]); // Set empty array on error
    }
  };

  const handleVerify = async (id, verified) => {
    try {
      await verifyRecognition(id, { verified });
      loadHistory();
      if (result?.id === id) {
        setResult({ ...result, verified });
      }
    } catch (error) {
      console.error('Error verifying recognition:', error);
    }
  };

  return (
    <DashboardCard
      title="Pill Recognition"
      icon={<Camera size={20} />}
      variant="patient"
    >
      <div className="space-y-6">
        {/* Image Upload */}
        <div>
          <label className="block text-sm font-medium text-neutral-700 mb-2">
            Upload Pill Image
          </label>
          <div className="border-2 border-dashed border-neutral-300 rounded-2xl p-6 text-center hover:border-primary-400 transition-colors">
            {preview ? (
              <div className="space-y-4">
                <img
                  src={preview}
                  alt="Pill preview"
                  className="max-h-48 mx-auto rounded-xl"
                />
                <Button
                  onClick={() => {
                    setPreview(null);
                    if (fileInputRef.current) fileInputRef.current.value = '';
                  }}
                  variant="ghost"
                  size="sm"
                >
                  Change Image
                </Button>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="flex justify-center">
                  <div className="w-16 h-16 rounded-full bg-primary-100 flex items-center justify-center">
                    <ImageIcon className="w-8 h-8 text-primary-600" />
                  </div>
                </div>
                <div>
                  <label htmlFor="pill-image" className="cursor-pointer">
                    <Button variant="primary" size="md" as="span">
                      <Upload className="w-4 h-4 mr-2" />
                      Choose Image
                    </Button>
                  </label>
                  <input
                    ref={fileInputRef}
                    id="pill-image"
                    type="file"
                    accept="image/*"
                    onChange={handleFileSelect}
                    className="hidden"
                  />
                </div>
                <p className="text-xs text-neutral-600">
                  JPG, PNG, or WebP up to 10MB
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Recognize Button */}
        {preview && (
          <Button
            onClick={handleRecognize}
            disabled={isProcessing}
            variant="primary"
            size="md"
            className="w-full"
          >
            {isProcessing ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                Recognizing...
              </>
            ) : (
              <>
                <Camera className="w-4 h-4 mr-2" />
                Recognize Pill
              </>
            )}
          </Button>
        )}

        {/* Error Message */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: prefersReducedMotion ? 0 : -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="p-3 bg-error-50 border border-error-200 rounded-xl"
          >
            <p className="text-sm text-error-700">{error}</p>
          </motion.div>
        )}

        {/* Recognition Result */}
        {result && (
          <motion.div
            initial={{ opacity: 0, y: prefersReducedMotion ? 0 : 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-4"
          >
            {/* Interaction Warnings */}
            {result.interactions && result.interactions.length > 0 && (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="p-4 bg-error-50 border-2 border-error-300 rounded-xl"
              >
                <div className="flex items-center gap-2 mb-2">
                  <AlertTriangle className="w-5 h-5 text-error-600" />
                  <h4 className="font-semibold text-error-900">
                    Drug Interaction Warning
                  </h4>
                </div>
                <p className="text-sm text-error-700 mb-3">
                  This medication may interact with your current medications:
                </p>
                <div className="space-y-2">
                  {result.interactions.map((interaction, idx) => (
                    <div key={idx} className="p-3 bg-white rounded-lg border border-error-200">
                      <div className="flex items-center gap-2 mb-1">
                        <Badge variant={interaction.severity === 'severe' ? 'error' : 'warning'}>
                          {interaction.severity} - {interaction.type}
                        </Badge>
                      </div>
                      <p className="text-sm font-medium text-neutral-900 mb-1">
                        {interaction.medication1} + {interaction.medication2}
                      </p>
                      <p className="text-xs text-neutral-700 mb-2">{interaction.description}</p>
                      {interaction.management && (
                        <div className="mt-2 p-2 bg-neutral-50 rounded text-xs text-neutral-700">
                          <span className="font-medium">Management: </span>
                          {interaction.management}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </motion.div>
            )}

            {/* Recognition Details */}
            <div className="p-6 bg-white rounded-2xl border-2 border-primary-200">
              <div className="flex items-center justify-between mb-4">
                <h4 className="font-semibold text-neutral-900">Recognition Result</h4>
                {result.verified ? (
                  <Badge variant="success">Verified</Badge>
                ) : (
                  <Badge variant="warning">Unverified</Badge>
                )}
              </div>
            
            {result.recognized ? (
              <div className="space-y-3">
                {result.medicationName && (
                  <div>
                    <p className="text-xs text-neutral-600 mb-1">Medication</p>
                    <p className="font-semibold text-neutral-900">{result.medicationName}</p>
                  </div>
                )}
                <div className="grid grid-cols-2 gap-4">
                  {result.imprint && (
                    <div>
                      <p className="text-xs text-neutral-600 mb-1">Imprint</p>
                      <p className="text-sm text-neutral-900">{result.imprint}</p>
                    </div>
                  )}
                  {result.shape && (
                    <div>
                      <p className="text-xs text-neutral-600 mb-1">Shape</p>
                      <p className="text-sm text-neutral-900 capitalize">{result.shape}</p>
                    </div>
                  )}
                  {result.color && (
                    <div>
                      <p className="text-xs text-neutral-600 mb-1">Color</p>
                      <p className="text-sm text-neutral-900 capitalize">{result.color}</p>
                    </div>
                  )}
                  {result.size && (
                    <div>
                      <p className="text-xs text-neutral-600 mb-1">Size</p>
                      <p className="text-sm text-neutral-900 capitalize">{result.size}</p>
                    </div>
                  )}
                </div>
                {result.confidence && (
                  <div>
                    <p className="text-xs text-neutral-600 mb-1">Confidence</p>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-2 bg-neutral-200 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-primary-600 transition-all"
                          style={{ width: `${result.confidence * 100}%` }}
                        />
                      </div>
                      <span className="text-xs text-neutral-600">
                        {(result.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                )}
                {!result.verified && (
                  <div className="flex items-center gap-2 pt-2 flex-wrap">
                    <Button
                      onClick={async () => {
                        try {
                          // Add medication from recognized pill
                          const response = await addMedicationFromPill({
                            recognitionId: result.id,
                            medicationData: {
                              dosage: 'As directed',
                              frequency: 'daily'
                            }
                          });
                          
                          if (response.hasInteractions) {
                            alert('Medication added, but interactions detected. Please review warnings.');
                          } else {
                            alert('Medication added successfully!');
                          }
                          
                          handleVerify(result.id, true);
                        } catch (error) {
                          console.error('Error adding medication:', error);
                          alert('Failed to add medication. Please try again.');
                        }
                      }}
                      variant="primary"
                      size="sm"
                    >
                      <CheckCircle className="w-4 h-4 mr-1" />
                      Add to Medications
                    </Button>
                    <Button
                      onClick={() => handleVerify(result.id, true)}
                      variant="success"
                      size="sm"
                    >
                      <CheckCircle className="w-4 h-4 mr-1" />
                      Verify Only
                    </Button>
                    <Button
                      onClick={() => handleVerify(result.id, false)}
                      variant="ghost"
                      size="sm"
                    >
                      <XCircle className="w-4 h-4 mr-1" />
                      Incorrect
                    </Button>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-4">
                <XCircle className="w-12 h-12 text-neutral-400 mx-auto mb-2" />
                <p className="text-sm text-neutral-600">
                  Could not recognize pill. Please try a clearer image.
                </p>
              </div>
            )}
            </div>
          </motion.div>
        )}

        {/* Recognition History */}
        {history.length > 0 && (
          <div>
            <h4 className="text-sm font-semibold text-neutral-900 mb-3">Recent Recognitions</h4>
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {history.slice(0, 5).map((item) => (
                <div
                  key={item.id}
                  className="p-3 bg-neutral-50 rounded-xl border border-neutral-200"
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-neutral-900">
                        {item.medicationName || 'Unknown'}
                      </p>
                      <p className="text-xs text-neutral-600">
                        {new Date(item.createdAt).toLocaleDateString()}
                      </p>
                    </div>
                    {item.verified ? (
                      <CheckCircle className="w-5 h-5 text-medical-600" />
                    ) : (
                      <Badge variant="warning">Unverified</Badge>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </DashboardCard>
  );
};

