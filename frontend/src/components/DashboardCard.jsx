import { motion } from 'framer-motion';

export default function DashboardCard({ title, children, className = "" }) {
  return (
    <motion.div 
      className={`bg-white rounded-2xl sm:rounded-3xl shadow-lg hover:shadow-xl p-4 sm:p-6 transition-all duration-200 ${className}`}
      whileHover={{ y: -2, scale: 1.01 }}
      transition={{ type: "spring", stiffness: 300, damping: 30 }}
    >
      <h3 className="text-base sm:text-lg lg:text-xl font-semibold mb-3 sm:mb-4 text-gray-800 bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
        {title}
      </h3>
      <div className="text-sm sm:text-base">
        {children}
      </div>
    </motion.div>
  );
}
