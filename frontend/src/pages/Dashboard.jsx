import Navigation from '../components/Navigation';
import UpcomingMeasurementsCard from '../components/UpcomingMeasurementsCard';
import MedicationRemindersCard from '../components/MedicationRemindersCard';
import MetricRemindersCard from '../components/MetricRemindersCard';
import MetricHistoryTable from '../components/MetricHistoryTable';
import CycleDetailCard from '../components/CycleDetailCard';
import AddMedicationCycleCard from '../components/AddMedicationCycleCard';
import AddMetricCard from '../components/AddMetricCard';
import UpcomingIntakeCard from '../components/UpcomingIntakeCard';

export default function Dashboard() {
  return (
    <div className="min-h-screen bg-gray-100">
      <Navigation />
      <div className="max-w-6xl mx-auto p-4 columns-1 md:columns-2 gap-6 space-y-6">
        <div className="mb-6 break-inside-avoid">
          <UpcomingMeasurementsCard />
        </div>
        <div className="mb-6 break-inside-avoid">
          <MedicationRemindersCard />
        </div>
        <div className="mb-6 break-inside-avoid">
          <MetricRemindersCard />
        </div>
        <div className="mb-6 break-inside-avoid">
          <MetricHistoryTable />
        </div>
        <div className="mb-6 break-inside-avoid">
          <UpcomingIntakeCard />
        </div>
        <div className="mb-6 break-inside-avoid">
          <CycleDetailCard />
        </div>
        <div className="mb-6 break-inside-avoid">
          <AddMedicationCycleCard />
        </div>
        <div className="mb-6 break-inside-avoid">
          <AddMetricCard />
        </div>
      </div>
    </div>
  );
}
