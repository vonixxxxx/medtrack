import { motion } from 'framer-motion';
import MobileNavigation from '../components/MobileNavigation';
import UpcomingMeasurementsCard from '../components/UpcomingMeasurementsCard';
import MedicationRemindersCard from '../components/MedicationRemindersCard';
import MetricRemindersCard from '../components/MetricRemindersCard';
import MetricHistoryTable from '../components/MetricHistoryTable';
import CycleDetailCard from '../components/CycleDetailCard';
import AddMedicationCycleCard from '../components/AddMedicationCycleCard';
import AddMetricCard from '../components/AddMetricCard';
import UpcomingIntakeCard from '../components/UpcomingIntakeCard';

export default function Dashboard() {
  const cardVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 }
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      <MobileNavigation />
      
      <motion.div 
        className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {/* Header */}
        <motion.div 
          className="mb-8"
          variants={cardVariants}
        >
          <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold text-gray-900 mb-2">
            Dashboard
          </h1>
          <p className="text-gray-600 text-sm sm:text-base">
            Monitor your medications and health metrics
          </p>
        </motion.div>

        {/* Responsive Grid Layout */}
        <motion.div 
          className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4 sm:gap-6"
          variants={containerVariants}
        >
          <motion.div variants={cardVariants} className="w-full">
            <UpcomingMeasurementsCard />
          </motion.div>
          
          <motion.div variants={cardVariants} className="w-full">
            <MedicationRemindersCard />
          </motion.div>
          
          <motion.div variants={cardVariants} className="w-full">
            <MetricRemindersCard />
          </motion.div>
          
          <motion.div variants={cardVariants} className="w-full lg:col-span-2 xl:col-span-3">
            <MetricHistoryTable />
          </motion.div>
          
          <motion.div variants={cardVariants} className="w-full">
            <UpcomingIntakeCard />
          </motion.div>
          
          <motion.div variants={cardVariants} className="w-full">
            <CycleDetailCard />
          </motion.div>
          
          <motion.div variants={cardVariants} className="w-full">
            <AddMedicationCycleCard />
          </motion.div>
          
          <motion.div variants={cardVariants} className="w-full lg:col-span-2 xl:col-span-1">
            <AddMetricCard />
          </motion.div>
        </motion.div>
      </motion.div>
    </div>
  );
}
