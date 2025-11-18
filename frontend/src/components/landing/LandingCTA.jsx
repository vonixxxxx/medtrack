import { motion } from "framer-motion";
import { ArrowRight, User, Stethoscope } from "lucide-react";
import { useState } from "react";
import LandingAuthModal from "./LandingAuthModal";

export default function LandingCTA() {
  const [authModalOpen, setAuthModalOpen] = useState(false);
  const [authType, setAuthType] = useState("signup");
  const [userType, setUserType] = useState("patient");

  const openSignup = (type) => {
    setUserType(type);
    setAuthType("signup");
    setAuthModalOpen(true);
  };

  return (
    <>
      <section id="cta-section" className="py-24 lg:py-32 bg-gradient-to-br from-blue-600 via-blue-600 to-blue-700 relative overflow-hidden">
        {/* Background Pattern */}
        <div className="absolute inset-0 opacity-10">
          <div className="absolute inset-0" style={{
            backgroundImage: `radial-gradient(circle at 2px 2px, white 1px, transparent 0)`,
            backgroundSize: "40px 40px",
          }} />
        </div>

        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            className="text-center"
          >
            <h2 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-white mb-6">
              Join the Future of Healthcare
            </h2>
            <p className="text-xl text-blue-100 mb-12 max-w-2xl mx-auto">
              Start your journey with MedTrack today. Choose the experience that&apos;s right for you.
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <motion.button
                whileHover={{ scale: 1.05, y: -2 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => openSignup("patient")}
                className="group w-full sm:w-auto px-8 py-4 bg-white text-blue-600 rounded-full font-semibold text-lg shadow-xl hover:shadow-2xl transition-all flex items-center justify-center gap-2"
              >
                <User size={20} />
                Sign Up as Patient
                <ArrowRight className="group-hover:translate-x-1 transition-transform" size={20} />
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.05, y: -2 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => openSignup("clinician")}
                className="group w-full sm:w-auto px-8 py-4 bg-blue-800 text-white rounded-full font-semibold text-lg border-2 border-blue-400 hover:bg-blue-700 transition-all flex items-center justify-center gap-2"
              >
                <Stethoscope size={20} />
                Sign Up as Clinician
                <ArrowRight className="group-hover:translate-x-1 transition-transform" size={20} />
              </motion.button>
            </div>

            <motion.p
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              transition={{ delay: 0.3 }}
              className="mt-8 text-blue-100 text-sm"
            >
              Already have an account?{" "}
              <button
                onClick={() => {
                  setAuthType("login");
                  setAuthModalOpen(true);
                }}
                className="underline hover:text-white transition-colors font-medium"
              >
                Sign in here
              </button>
            </motion.p>
          </motion.div>
        </div>
      </section>

      <LandingAuthModal
        isOpen={authModalOpen}
        onClose={() => setAuthModalOpen(false)}
        initialType={authType}
        initialUserType={userType}
      />
    </>
  );
}


