import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, Info, X } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '../ui/alert';

/**
 * Patient Alert Component
 * Based on Confir-Med PatientAlert.svelte
 * Displays warnings and reports for patients
 */

export const PatientAlert = ({ alerts = [], type = 'warning' }) => {
  const [visible, setVisible] = useState(true);

  if (!visible || !alerts || alerts.length === 0) {
    return null;
  }

  const isWarning = type === 'warning';

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.2 }}
        >
          <Alert
            variant={isWarning ? 'destructive' : 'default'}
            className="relative"
          >
            <button
              onClick={() => setVisible(false)}
              className="absolute right-2 top-2"
              aria-label="Close"
            >
              <X
                className={`h-4 w-4 ${
                  isWarning
                    ? 'text-error-500 hover:text-error-700'
                    : 'text-neutral-500 hover:text-neutral-700'
                } transition-colors duration-150`}
              />
            </button>

            <AlertTitle>
              <div className="flex items-center gap-1">
                {isWarning ? (
                  <>
                    <AlertTriangle className="h-5 w-5" />
                    <h2 className="font-semibold">Warnings</h2>
                  </>
                ) : (
                  <>
                    <Info className="h-5 w-5" />
                    <h2 className="font-semibold">Reports</h2>
                  </>
                )}
              </div>
            </AlertTitle>
            <AlertDescription>
              <div className="text-sm space-y-1">
                {alerts.map((alert) => (
                  <div key={alert.id || alert.message}>
                    <p>
                      {alert.patient && `${alert.patient}: `}
                      {alert.message}
                    </p>
                  </div>
                ))}
              </div>
            </AlertDescription>
          </Alert>
        </motion.div>
      )}
    </AnimatePresence>
  );
};



