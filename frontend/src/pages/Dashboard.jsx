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
        className="max-w-6xl mx-auto p-4"
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

        {/* Original Tile Layout with Responsive Improvements */}
        <motion.div 
          className="columns-1 md:columns-2 gap-6 space-y-6"
          variants={containerVariants}
        >
          <motion.div variants={cardVariants} className="mb-6 break-inside-avoid">
            <UpcomingMeasurementsCard />
          </motion.div>
          
          <motion.div variants={cardVariants} className="mb-6 break-inside-avoid">
            <MedicationRemindersCard />
          </motion.div>
          
          <motion.div variants={cardVariants} className="mb-6 break-inside-avoid">
            <MetricRemindersCard />
          </motion.div>
          
          <motion.div variants={cardVariants} className="mb-6 break-inside-avoid">
            <MetricHistoryTable />
          </motion.div>
          
          <motion.div variants={cardVariants} className="mb-6 break-inside-avoid">
            <UpcomingIntakeCard />
          </motion.div>
          
          <motion.div variants={cardVariants} className="mb-6 break-inside-avoid">
            <CycleDetailCard />
          </motion.div>
          
          <motion.div variants={cardVariants} className="mb-6 break-inside-avoid">
            <AddMedicationCycleCard />
          </motion.div>
          
          <motion.div variants={cardVariants} className="mb-6 break-inside-avoid">
            <AddMetricCard />
          </motion.div>
        </motion.div>
      </motion.div>
    </div>
  );
}
