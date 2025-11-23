# Clinician Dashboard - Complete Feature & Architecture Report

## Table of Contents
1. [Overview](#overview)
2. [Authentication & Authorization](#authentication--authorization)
3. [Core Dashboard Components](#core-dashboard-components)
4. [Patient Records Management](#patient-records-management)
5. [Filter System](#filter-system)
6. [Analytics Panel](#analytics-panel)
7. [Medical History Parser](#medical-history-parser)
8. [Graph Builder](#graph-builder)
9. [HbA1c Adjustment Calculator](#hba1c-adjustment-calculator)
10. [Metrics Analytics](#metrics-analytics)
11. [AI Data Validation Panel](#ai-data-validation-panel)
12. [Backend API Architecture](#backend-api-architecture)
13. [Database Schema](#database-schema)
14. [Data Flow Diagrams](#data-flow-diagrams)

---

## Overview

The Clinician Dashboard is a comprehensive web-based interface designed for healthcare professionals to manage patient records, analyze medical data, and utilize AI-powered tools for clinical decision-making. The dashboard is built with React (TypeScript/JavaScript) frontend and Node.js/Express backend, utilizing Prisma ORM for database management.

**Key Technologies:**
- Frontend: React, TypeScript, Tailwind CSS, Framer Motion, Recharts
- Backend: Node.js, Express.js, Prisma ORM
- Database: PostgreSQL
- AI Services: BioGPT, Ollama (for medical text parsing)
- Authentication: JWT tokens with role-based access control

---

## Authentication & Authorization

### Architecture

**Frontend (`DoctorDashboard.tsx`)**
- Component: `DoctorDashboard`
- Location: `frontend/src/pages/DoctorDashboard.tsx`
- Authentication Flow:
  1. Checks for JWT token in `localStorage`
  2. Validates token via `GET /api/auth/me`
  3. Redirects to login if authentication fails
  4. Loads user data and hospital code

**Backend Middleware**
- Location: `backend/src/middleware/roleMiddleware.js`
- Middleware: `requireClinician`
- Functionality:
  - Validates JWT token from Authorization header
  - Verifies user role is "clinician"
  - Attaches user data to `req.user`
  - Returns 401/403 on failure

**Route Protection**
```javascript
// All doctor routes protected
router.use(requireClinician);
```

**Hospital Code Isolation**
- Clinicians are assigned a `hospitalCode` during registration
- All patient queries filtered by `hospitalCode`
- Patients must have matching `hospitalCode` to be accessible
- Stored in JWT token for fast access

### How It Works

1. **Login Process:**
   - User authenticates with email/password
   - Backend validates credentials and returns JWT token
   - Token includes: `{ id, email, role, hospitalCode }`
   - Token stored in `localStorage` on frontend

2. **Dashboard Access:**
   - Dashboard checks token existence on mount
   - Makes authenticated request to `/api/auth/me`
   - If valid, loads dashboard; if not, redirects to login

3. **API Requests:**
   - All API calls include token in `Authorization: Bearer <token>` header
   - Backend validates token and role on each request
   - Patient data automatically filtered by clinician's `hospitalCode`

---

## Core Dashboard Components

### Main Dashboard Layout

**Component:** `DoctorDashboard.tsx`
**State Management:**
```typescript
- isProfileOpen: boolean
- isSettingsOpen: boolean
- isHbA1cModalOpen: boolean
- isAIValidationOpen: boolean
- showMetricsAnalytics: boolean
- patients: Patient[]
- filteredPatients: Patient[]
- selectedPatient: Patient | null
- filters: FilterState
- isLoading: boolean
- user: User | null
```

**Component Structure:**
```
DoctorDashboard
├── Header (shared component)
├── Welcome Section
│   ├── Title & Hospital Code display
│   └── Selected Patient indicator
├── FilterSystem
├── AnalyticsPanel
├── MedicalHistoryParser
├── AI Feature Buttons
│   ├── AI Data Validation
│   └── Metrics Analytics toggle
├── EnhancedPatientRecordsTable
├── GraphBuilder
└── Modals
    ├── HbA1cAdjustmentModal
    └── AIValidationPanel
```

**Data Loading:**
- `loadUserData()`: Fetches authenticated user info
- `loadPatients()`: Fetches all patients with matching `hospitalCode`
- Both called on component mount via `useEffect`

**Patient Selection Flow:**
1. User clicks patient row in table
2. `handlePatientSelect()` updates `selectedPatient` state
3. Selected patient highlighted in table
4. Patient info displayed in header
5. Enables patient-specific features (Metrics Analytics, etc.)

---

## Patient Records Management

### Enhanced Patient Records Table

**Component:** `EnhancedPatientRecordsTable.tsx`
**Location:** `frontend/src/components/doctor/EnhancedPatientRecordsTable.tsx`

#### Features

1. **Comprehensive Data Display**
   - 60+ columns of patient data organized by categories:
     - Basic Demographics: name, email, age, sex, ethnicity, location, postcode, NHS number, MRN
     - Measurements: height, weight, BMI, weight date
     - Conditions: 30+ boolean condition flags (ASCVD, HTN, T2DM, OSA, etc.)
     - Diabetes: type, baseline HbA1c, HbA1c dates
     - Lipids: total cholesterol, HDL, LDL, triglycerides
     - Medications: lipid lowering, antihypertensive
     - Bariatric: gastric band, sleeve, bypass, balloon
     - Clinical: comorbidities, MES score, notes

2. **Column Management**
   - Dynamic column selection via checkbox interface
   - Organized by category (basic, measurements, conditions, diabetes, lipids, medications, clinical)
   - Persistent selection across sessions (default includes most common columns)
   - Collapsible column selector panel

3. **Sorting**
   - Sortable columns indicated by arrow indicators
   - Click header to sort ascending/descending
   - Handles null/undefined values (sorted to end)
   - Maintains sort state across filters/pagination

4. **Search Functionality**
   - Real-time text search across multiple fields:
     - Name, email, NHS number, MRN, postcode, location, ethnicity
   - Case-insensitive matching
   - Updates filtered results instantly

5. **Advanced Filtering**
   - **Sex Filter:** Male, Female, Other, All
   - **Ethnic Group Filter:** White British, Asian Indian, Asian Pakistani, Black Caribbean, Other
   - **BMI Range:** Underweight (<18.5), Normal (18.5-24.9), Overweight (25-29.9), Obese (≥30)
   - **HbA1c Range:** Normal (<5.7%), Prediabetes (5.7-6.4%), Diabetes (≥6.5%)
   - **Age Range:** 18-30, 31-45, 46-60, 60+
   - **Condition Filters:** ASCVD, T2DM, HTN, Dyslipidaemia (boolean)
   - Filters can be combined (AND logic)
   - Clear all filters button

6. **Pagination**
   - Configurable items per page: 10, 25, 50, 100
   - Previous/Next navigation
   - Current page indicator
   - Total pages calculation

7. **Data Export**
   - Export filtered results to CSV
   - Includes only selected columns
   - Properly formatted with headers
   - File name includes date: `patient_records_YYYY-MM-DD.csv`

8. **Visual Features**
   - Row highlighting on hover
   - Selected patient highlighted with blue border
   - Click row to select patient
   - Responsive table with horizontal scroll
   - Loading states during data fetch

#### Architecture

**State Management:**
```typescript
- sortConfig: { key, direction } | null
- searchTerm: string
- showFilters: boolean
- currentPage: number
- itemsPerPage: number
- filters: FilterState
- selectedColumns: Set<string>
```

**Data Flow:**
1. Receives `patients` array as prop
2. Applies filters → `getFilteredData()`
3. Applies sorting → `getSortedData()`
4. Applies pagination → `getPaginatedData()`
5. Renders table with formatted cell values

**Cell Formatting:**
- Boolean values: ✓ / ✗
- Dates: `toLocaleDateString()`
- Arrays (conditions): Joined with commas
- Null/undefined: Displayed as "-"

**API Integration:**
- `onRefresh()` callback triggers parent to reload patients
- `onPatientSelect()` callback updates selected patient in parent
- `onHbA1cAdjustment()` opens HbA1c modal

---

## Filter System

### Component

**Component:** `FilterSystem.tsx`
**Location:** `frontend/src/components/doctor/FilterSystem.tsx`

#### Features

1. **Filter Types**
   - **Metric Filter:** All Metrics, HbA1c, Blood Pressure, BMI, Weight, Glucose
   - **Date Range Filter:** All Time, Today, This Week, This Month, This Quarter, This Year, Custom Range
   - **Ethnicity Filter:** All, White, Black/African American, Hispanic/Latino, Asian, Native American, Pacific Islander, Other, Unknown
   - **Sex Filter:** All, Male, Female, Other

2. **UI Features**
   - Expandable/collapsible panel
   - Active filters displayed as colored badges
   - "Clear All" button when filters active
   - Responsive grid layout (1-4 columns based on screen size)

#### Architecture

**Props:**
```typescript
interface FilterSystemProps {
  filters: {
    metric: string;
    dateRange: string;
    ethnicity: string;
    sex: string;
  };
  onFilterChange: (filters: FilterState) => void;
}
```

**State:**
- `isExpanded: boolean` - Controls panel visibility

**Filter Application:**
- Filters passed to parent `DoctorDashboard`
- Parent applies filters to patient data
- Filtered patients passed to other components

**Note:** Filter system provides UI; actual filtering logic in `DoctorDashboard.handleFilterChange()`

---

## Analytics Panel

### Component

**Component:** `AnalyticsPanel.tsx`
**Location:** `frontend/src/components/doctor/AnalyticsPanel.tsx`

#### Features

1. **Summary Statistics (Key Metrics)**
   - **Total Patients:** Count of filtered patients
   - **Average Age:** Mean age of patient population
   - **Average HbA1c:** Mean HbA1c percentage
   - **Improvement Rate:** Percentage of patients with negative change percent

2. **Demographics Analysis**
   - **Gender Distribution:** Breakdown by sex with visual bars
   - Percentage and count for each gender
   - Horizontal bar visualization

3. **Conditions Analysis**
   - **Top 5 Conditions:** Most common conditions across patients
   - Condition frequency with visual bars
   - Percentage of patients with each condition

4. **Percentile Analysis**
   - Sortable table showing patients by selected metric
   - Displays metric value and change percentage
   - Color-coded change indicators:
     - Red (↑): Increase
     - Green (↓): Decrease
     - Gray (→): No change

5. **Metric Selection**
   - Dropdown to select metric for percentile analysis:
     - HbA1c (%)
     - HbA1c (mmol/mol)
     - MES
     - Age
     - Change (%)

#### Architecture

**Props:**
```typescript
interface AnalyticsPanelProps {
  patients: Patient[];
  onGenerateGraph: () => void;
}
```

**Computed Analytics (useMemo):**
```typescript
analytics = {
  totalPatients: number;
  averageAge: number;
  genderDistribution: { male, female, other };
  averageHbA1c: number;
  averageMES: number;
  improvementRate: number; // % with negative change
  topConditions: Array<{ condition: string, count: number }>;
  percentileChanges: Array<{ name: string, value: number, change: number }>;
}
```

**Calculation Logic:**
- Recalculates when `patients` or `selectedMetric` changes
- Filters out null/undefined values
- Sorts and ranks by selected metric
- Calculates percentages and averages

**Data Visualization:**
- Uses inline CSS for bar charts
- Color-coded metrics for quick interpretation
- Responsive grid layout for statistics cards

---

## Medical History Parser

### Component

**Component:** `MedicalHistoryParser.tsx`
**Location:** `frontend/src/components/doctor/MedicalHistoryParser.tsx`

#### Features

1. **Text Input**
   - Large textarea for pasting medical notes
   - Accepts unstructured clinical documentation
   - Handles discharge notes, progress notes, lab results

2. **AI Processing**
   - Uses BioGPT AI for medical text extraction
   - Extracts structured data from unstructured text
   - Normalizes condition names
   - Maps extracted data to database fields

3. **Data Extraction**
   - Patient demographics (name, age, sex)
   - Medical conditions (normalized)
   - Lab values (HbA1c, lipids, etc.)
   - Medications
   - Clinical measurements (BMI, weight, etc.)

4. **Audit Trail**
   - Creates audit logs for all extracted data
   - Compares with existing patient data
   - Flags discrepancies for clinician review
   - Shows count of updates requiring review

5. **Condition Management**
   - Displays extracted conditions as tags
   - Normalized condition names (e.g., "T2DM" → "Type 2 Diabetes Mellitus")
   - Allows adding conditions to patient record
   - Requires patient selection before adding

#### Architecture

**Props:**
```typescript
interface MedicalHistoryParserProps {
  selectedPatientId?: string;
  onConditionsAdded?: () => void;
}
```

**State:**
```typescript
- medicalNotes: string
- isProcessing: boolean
- extractedConditions: string[]
- error: string
- success: string
```

**API Integration:**
- `POST /api/doctor/parse-history`
  - Request: `{ patientId, medicalNotes }`
  - Response: `{ parsedData, conditions, auditLogs, ... }`

**Backend Processing:**
1. Receives medical notes
2. Calls AI parsing service (BioGPT/Ollama)
3. Extracts structured data
4. Compares with current patient data
5. Creates audit log entries for discrepancies
6. Extracts and normalizes conditions
7. Returns parsed data and conditions

**Data Flow:**
```
User Input (Text) 
  → API Call 
  → AI Processing (BioGPT/Ollama)
  → Data Extraction & Normalization
  → Comparison with Existing Data
  → Audit Log Creation
  → Response with Extracted Data
  → Display Conditions & Results
```

**Condition Normalization:**
- Maps abbreviations to full names
- Removes duplicates
- Categorizes by medical specialty
- Stores both raw and normalized forms

---

## Graph Builder

### Component

**Component:** `GraphBuilder.tsx`
**Location:** `frontend/src/components/doctor/GraphBuilder.tsx`

#### Features

1. **Graph Types**
   - **Line Chart:** Time series or scatter plots
   - **Bar Chart:** Comparative data visualization
   - **Pie Chart:** Distribution/categorical data

2. **Axis Configuration**
   - **X-Axis Options:** Age, HbA1c (%), HbA1c (mmol/mol), MES, Change (%)
   - **Y-Axis Options:** HbA1c (%), HbA1c (mmol/mol), MES, Change (%), Age
   - Different options for pie charts (categorical grouping)

3. **Patient Selection**
   - Checkbox list of all patients
   - Select All / Clear All buttons
   - Filter graph to selected patients only
   - Shows only selected patients' data

4. **Interactive Charts**
   - Built with Recharts library
   - Responsive container
   - Tooltips on hover
   - Dark theme styling
   - Grid lines for readability

5. **Data Processing**
   - Automatically groups data by date for line/bar charts
   - Aggregates categorical data for pie charts
   - Handles multiple metrics simultaneously

#### Architecture

**Props:**
```typescript
interface GraphBuilderProps {
  patients: Patient[];
  filters: FilterState;
}
```

**State:**
```typescript
- graphType: 'line' | 'bar' | 'pie'
- xAxis: string
- yAxis: string
- selectedPatients: number[]
- isGenerating: boolean
```

**Data Generation:**
- `generateGraphData()`: Processes patient data based on selections
- For line/bar: Creates data points with x/y values
- For pie: Groups and counts by categorical field

**Chart Rendering:**
- Uses Recharts components:
  - `LineChart`, `BarChart`, `PieChart`
  - `ResponsiveContainer` for responsive sizing
  - Custom styling for dark theme

**Visual Features:**
- Color-coded data series
- Interactive tooltips
- Legend for multiple series
- Smooth animations

---

## HbA1c Adjustment Calculator

### Component

**Component:** `HbA1cAdjustmentModal.tsx`
**Location:** `frontend/src/components/doctor/HbA1cAdjustmentModal.tsx`

#### Features

1. **Input Fields**
   - **Measured HbA1c (%):** Current HbA1c measurement
   - **Weight (kg):** Patient weight in kilograms
   - **Medication Doses:** 17 diabetes medications with dose inputs

2. **Supported Medications**
   - Insulin
   - Metformin
   - Sulfonylureas: Glimepiride, Glipizide, Glyburide
   - Thiazolidinediones: Pioglitazone
   - DPP-4 Inhibitors: Sitagliptin, Saxagliptin, Linagliptin
   - GLP-1 Agonists: Liraglutide, Exenatide (BID & QW), Dulaglutide, Semaglutide
   - SGLT2 Inhibitors: Dapagliflozin, Canagliflozin, Empagliflozin

3. **Calculation Results**
   - **MES (Medication Effect Score):** Quantifies medication impact
   - **Adjusted HbA1c (%):** HbA1c adjusted for medication effect
   - **Adjusted HbA1c (mmol/mol):** International standard units
   - Detailed breakdown of calculation

4. **Clinical Interpretation**
   - Explanation of MES significance
   - Guidance on using adjusted values
   - Treatment decision support information

#### Architecture

**Component Type:** Modal (overlay)
**State:**
```typescript
- measuredHbA1c: string
- weight: string
- medications: Medication[]
- result: CalculationResult | null
- isCalculating: boolean
- error: string
```

**API Integration:**
- `POST /api/doctor/hba1c-adjust`
- Request: `{ measuredHbA1cPercent, weightKg, medications: { drug: dose } }`
- Response: `{ MES, adjustedHbA1cPercent, adjustedHbA1cMmolMol }`

**Backend Service:**
- Location: `backend/src/services/hba1cService.js`
- Function: `calculateAdjustedHbA1c()`

**Calculation Formula:**
```javascript
// MES Calculation for each medication:
MES += (actualDose / maxDose) * factor

// Adjusted HbA1c:
adjustedHbA1cPercent = measuredHbA1cPercent + MES
adjustedHbA1cMmolMol = (adjustedHbA1cPercent - 2.15) * 10.929
```

**Medication Factors:**
- Each medication has:
  - Max dose (mg/day or units)
  - Factor (contribution to MES)
  - Example: Metformin [2550mg, 1.5], Semaglutide [1mg, 1.4]

**UI/UX:**
- Two-column layout (inputs | results)
- Real-time calculation on button click
- Validation for required fields
- Error handling and display
- Reset button to clear all inputs

---

## Metrics Analytics

### Component

**Component:** `MetricsAnalytics.tsx`
**Location:** `frontend/src/components/doctor/MetricsAnalytics.tsx`

#### Features

1. **Data Sources Integration**
   - **Metric Trends:** Patient-logged metrics
   - **Lab Results:** Laboratory test results
   - **Vital Signs:** Clinical vital measurements
   - Unified view of all metric data

2. **Filtering & Selection**
   - **Metric Filter:** All metrics or specific metric
   - **Date Range:** Start and end date selection
   - **View Type:** Line chart or bar chart
   - Real-time filtering

3. **Data Visualization**
   - **Interactive Charts:** Line or bar charts
   - Multiple metrics on same chart
   - Color-coded data series
   - Hover tooltips with values
   - Date-formatted X-axis

4. **Metrics Table**
   - Tabular view of all data points
   - Shows: Date, Metric Name, Value, Unit, Source, Change %
   - Color-coded source badges
   - Trend direction indicators
   - Limited to 10 most recent entries

5. **Trend Analysis**
   - Calculates change from first to last value
   - Percentage change calculation
   - Trend direction (increasing/decreasing/stable)
   - Color-coded trend indicators

6. **Summary Statistics**
   - Total data points count
   - Number of unique metrics tracked
   - Date range summary

#### Architecture

**Props:**
```typescript
interface MetricsAnalyticsProps {
  patientId: string;
  patientName: string;
}
```

**State:**
```typescript
- metrics: MetricData[]
- labResults: LabResult[]
- vitalSigns: VitalSign[]
- loading: boolean
- selectedMetric: string
- dateRange: { start: string, end: string }
- viewType: 'line' | 'bar'
```

**API Integration:**
- `GET /api/metrics/patient/:patientId`
- `GET /api/lab-results/patient/:patientId`
- `GET /api/vital-signs/patient/:patientId`

**Data Processing:**
1. Fetch all three data sources in parallel
2. Combine into unified array with source tags
3. Filter by selected metric and date range
4. Group by date for charting
5. Calculate trends per metric

**Chart Data Structure:**
```typescript
chartData = [
  { date: "2024-01-01", "HbA1c": 7.5, "Weight": 80, ... },
  { date: "2024-01-15", "HbA1c": 7.2, "Weight": 79, ... }
]
```

**Trend Calculation:**
```typescript
change = lastValue - firstValue
changePercent = (change / firstValue) * 100
direction = change > 0 ? 'increasing' : change < 0 ? 'decreasing' : 'stable'
```

**Visual Features:**
- Dark theme styling
- Responsive charts (Recharts)
- Color-coded sources:
  - Lab Results: Blue
  - Vital Signs: Green
  - Metric Trends: Purple
- Trend indicators with colors (red=increase, green=decrease)

---

## AI Data Validation Panel

### Component

**Component:** `AIValidationPanel.tsx`
**Location:** `frontend/src/components/doctor/AIValidationPanel.tsx`

#### Features

1. **Medical Notes Processing**
   - Textarea for pasting unstructured medical notes
   - AI-powered intelligent parsing
   - Patient matching with confidence scores
   - Multi-patient selection when matches found

2. **Patient Matching**
   - Fuzzy matching algorithm
   - Confidence scores (0-1)
   - Multiple match candidates
   - Displays: Name, Email, NHS Number, MRN, Confidence %

3. **Data Extraction Preview**
   - Shows AI-extracted patient information
   - Displays clinical data summary
   - Condition and medication counts
   - Lab results summary

4. **Audit Log Management**
   - Lists all AI-suggested changes
   - Shows field name, old value, new value
   - AI confidence score with color coding:
     - High (≥0.8): Green
     - Medium (0.6-0.8): Yellow
     - Low (<0.6): Red
   - AI suggestion explanation text

5. **Change Approval Workflow**
   - Approve individual changes
   - Reject individual changes
   - Approve all changes (batch)
   - Visual indicators for approved status

6. **Status Tracking**
   - Pending suggestions count
   - Approved/rejected status
   - Created timestamp for each log

#### Architecture

**Props:**
```typescript
interface AIValidationPanelProps {
  patientId?: string;
  onClose: () => void;
  onPatientSelected?: (patientId: string) => void;
}
```

**State:**
```typescript
- auditLogs: AuditLog[]
- patientMatches: PatientMatch[]
- extractedData: any
- loading: boolean
- selectedPatientId: string | null
- showPatientSelection: boolean
- medicalNotes: string
- processing: boolean
```

**API Integration:**

1. **Intelligent Parse:**
   - `POST /api/doctor/intelligent-parse`
   - Request: `{ medicalNotes, hospitalCode }`
   - Response: 
     ```typescript
     {
       action: 'select_patient' | 'single_match' | 'create_patient',
       patientMatches?: PatientMatch[],
       extractedData: ExtractedData,
       patient?: Patient
     }
     ```

2. **Audit Logs:**
   - `GET /api/doctor/patients/:patientId/audit-logs`
   - Returns array of audit log entries

3. **Approve/Reject:**
   - `POST /api/doctor/audit-logs/:logId/approve`
   - `POST /api/doctor/audit-logs/:logId/reject`

**Audit Log Structure:**
```typescript
interface AuditLog {
  id: string;
  patientId: string;
  field_name: string;
  old_value: string | null;
  new_value: string;
  ai_confidence: number; // 0-1
  ai_suggestion: string;
  clinician_approved: boolean;
  createdAt: string;
}
```

**Backend Processing Flow:**
```
Medical Notes Input
  → Intelligent Parser Service
  → Patient Matching (fuzzy search)
  → Data Extraction (BioGPT/Ollama)
  → Field Mapping to Database Schema
  → Comparison with Existing Data
  → Create Audit Logs for Discrepancies
  → Return Matches + Extracted Data
```

**Patient Matching Algorithm:**
- Searches by name, email, NHS number, MRN
- Calculates similarity scores
- Returns top matches with confidence %
- Allows clinician to select correct patient

**Approval Workflow:**
1. AI extracts and suggests changes
2. Audit logs created with `clinician_approved: false`
3. Clinician reviews in validation panel
4. Approve/reject individual changes
5. On approve: Updates patient record, marks log as approved
6. On reject: Removes log entry

---

## Backend API Architecture

### Route Structure

**Base Path:** `/api/doctor`
**Middleware:** All routes protected by `requireClinician`

### Endpoints

#### 1. Patient Management

**GET `/api/doctor/patients`**
- Fetches all patients with matching `hospitalCode`
- Includes user info, conditions, latest metrics
- Transforms data for frontend consumption
- **Response:** Array of patient objects

**GET `/api/doctor/patients/:id`**
- Fetches specific patient details
- Includes survey data, metrics, medications
- Validates patient belongs to clinician's hospital
- **Response:** Patient object with relations

**POST `/api/doctor/patients/:id/conditions`**
- Adds conditions to patient record
- Accepts array of condition names
- Normalizes condition names
- **Request:** `{ conditions: string[] }`
- **Response:** `{ message, conditions }`

#### 2. Medical History Parsing

**POST `/api/doctor/parse-history`**
- Parses unstructured medical notes
- Uses AI to extract structured data
- Creates audit logs for discrepancies
- Extracts and saves conditions
- **Request:** `{ patientId, medicalNotes }`
- **Response:** `{ parsedData, conditions, auditLogs, ... }`

**POST `/api/doctor/intelligent-parse`**
- Advanced parsing with patient matching
- Returns patient matches or creates new patient
- **Request:** `{ medicalNotes, hospitalCode }`
- **Response:** `{ action, patientMatches?, extractedData, patient? }`

#### 3. Patient Data Updates

**PUT `/api/doctor/patients/:patientId`**
- Updates patient record fields
- Validates and sanitizes input
- **Request:** `{ field1: value1, field2: value2, ... }`
- **Response:** `{ success, patient }`

#### 4. Audit Log Management

**GET `/api/doctor/patients/:patientId/audit-logs`**
- Fetches all audit logs for patient
- Ordered by creation date (newest first)
- **Response:** Array of audit log objects

**POST `/api/doctor/audit-logs/:logId/approve`**
- Approves AI suggestion
- Updates patient record with new value
- Marks audit log as approved
- **Response:** Updated audit log

**POST `/api/doctor/audit-logs/:logId/reject`**
- Rejects AI suggestion
- Removes audit log entry
- **Response:** Success message

**POST `/api/doctor/patients/:patientId/approve-ai-suggestions`**
- Batch approval of multiple changes
- Updates patient with all approved fields
- Marks all related logs as approved
- **Request:** `{ fieldUpdates: { field: value } }`
- **Response:** `{ success, patient }`

#### 5. HbA1c Calculator

**POST `/api/doctor/hba1c-adjust`**
- Calculates adjusted HbA1c using MES
- **Request:** `{ measuredHbA1cPercent, weightKg, medications: { drug: dose } }`
- **Response:** `{ MES, adjustedHbA1cPercent, adjustedHbA1cMmolMol }`

#### 6. Analytics & Reporting

**GET `/api/doctor/analytics`**
- Returns population-level analytics
- Patient count, metrics summary
- **Response:** `{ totalPatients, totalMetrics, hospitalCode, lastUpdated }`

**GET `/api/doctor/export/patients`**
- Exports all patient data
- Includes survey data and latest metrics
- **Response:** Array of patient objects with relations

### Controller Architecture

**Location:** `backend/src/controllers/doctorController.js`

**Key Functions:**
- `getPatients()`: Patient list with filtering
- `getPatient()`: Single patient details
- `addPatientConditions()`: Condition management
- `parseMedicalHistory()`: AI parsing with audit logs
- `calculateHbA1cAdjustment()`: MES calculation
- `getAnalytics()`: Population analytics
- `exportPatients()`: Data export
- `getPatientAuditLogs()`: Audit log retrieval
- `approveAISuggestions()`: Batch approval
- `updatePatientData()`: Record updates

### Service Layer

**HbA1c Service:** `backend/src/services/hba1cService.js`
- `calculateAdjustedHbA1c()`: Main calculation function
- `getMedicationInfo()`: Drug information lookup
- `getAllMedications()`: List all supported medications

**AI Parsing Services:**
- `backend/utils/intelligentMedicalParser.js`: Intelligent parsing with patient matching
- `backend/utils/ollamaParser.js`: Ollama integration for AI parsing
- Uses BioGPT or Ollama for medical text extraction

### Database Access

**ORM:** Prisma
**Database:** PostgreSQL
**Connection:** Managed via Prisma Client

**Key Queries:**
```javascript
// Patients by hospital code
prisma.patient.findMany({
  where: { user: { hospitalCode, role: 'patient' } },
  include: { user, conditions, metrics }
})

// Create audit log
prisma.aiAuditLog.create({ data: {...} })

// Update patient
prisma.patient.update({ where: { id }, data: {...} })
```

---

## Database Schema

### Core Models

#### User Model
```prisma
model User {
  id            String    @id @default(cuid())
  email         String    @unique
  password      String
  name          String?
  role          String    @default("patient") // 'patient' | 'clinician'
  hospitalCode  String    // Links patients to clinicians
  createdAt     DateTime  @default(now())
  updatedAt     DateTime  @updatedAt
  
  // Relations
  patientProfile   Patient?
  clinicianProfile Clinician?
  medications      Medication[]
  metrics          Metric[]
  surveyData       UserSurveyData?
}
```

#### Patient Model
```prisma
model Patient {
  id             String   @id @default(cuid())
  userId         String   @unique
  patient_audit_id String @unique
  
  // Demographics
  sex            String?
  ethnic_group   String?
  ethnicity      String?
  location       String?
  postcode       String?
  nhs_number     String?
  mrn            String?
  dob            DateTime?
  
  // Measurements
  height         Float?
  baseline_weight Float?
  baseline_bmi   Float?
  baseline_weight_date DateTime?
  
  // Conditions (30+ boolean fields)
  ascvd          Boolean  @default(false)
  htn            Boolean  @default(false)
  t2dm           Boolean  @default(false)
  // ... many more
  
  // Diabetes
  diabetes_type  String?
  baseline_hba1c Float?
  baseline_hba1c_date DateTime?
  hba1c_percent  Float?
  hba1c_mmol     Float?
  
  // Lipids
  baseline_tc    Float?
  baseline_hdl   Float?
  baseline_ldl   Float?
  baseline_tg    Float?
  baseline_lipid_date DateTime?
  
  // Medications
  lipid_lowering_treatment String?
  antihypertensive_medications String?
  
  // Clinical
  mes            Float?
  total_qualifying_comorbidities Int?
  notes          String?
  criteria_for_wegovy String?
  
  // Relations
  user           User     @relation(fields: [userId], references: [id])
  conditions     Condition[]
  auditLogs      AiAuditLog[]
}
```

#### Condition Model
```prisma
model Condition {
  id          String   @id @default(cuid())
  patientId   String
  name        String
  normalized  String
  createdAt   DateTime @default(now())
  
  patient     Patient  @relation(fields: [patientId], references: [id])
}
```

#### AiAuditLog Model
```prisma
model AiAuditLog {
  id                  String   @id @default(cuid())
  patientId           String
  field_name          String
  old_value           String?
  new_value           String
  ai_confidence       Float
  ai_suggestion       String
  clinician_approved  Boolean  @default(false)
  createdAt           DateTime @default(now())
  
  patient             Patient  @relation(fields: [patientId], references: [id])
}
```

#### Clinician Model
```prisma
model Clinician {
  id          String   @id @default(cuid())
  userId      String   @unique
  hospitalCode String
  
  user        User     @relation(fields: [userId], references: [id])
}
```

#### Metric Model
```prisma
model Metric {
  id          String   @id @default(cuid())
  userId      String
  metric_name String
  value       Float
  unit        String
  timestamp   DateTime @default(now())
  date        DateTime
  
  user        User     @relation(fields: [userId], references: [id])
}
```

### Relationships

```
User
├── Patient (1:1)
│   ├── Condition (1:many)
│   └── AiAuditLog (1:many)
├── Clinician (1:1)
├── Medication (1:many)
├── Metric (1:many)
└── UserSurveyData (1:1)
```

### Indexing

- `User.hospitalCode`: Indexed for fast patient queries
- `User.email`: Unique index
- `Patient.userId`: Unique index
- `Condition.patientId`: Indexed for patient queries
- `AiAuditLog.patientId`: Indexed for audit log queries

---

## Data Flow Diagrams

### Patient Data Loading Flow

```
User Login
  ↓
JWT Token Created (includes hospitalCode)
  ↓
Dashboard Mount
  ↓
API: GET /api/auth/me (validate token)
  ↓
API: GET /api/doctor/patients
  ↓
Backend: Query by hospitalCode
  ↓
Transform Patient Data
  ↓
Return to Frontend
  ↓
State Update: setPatients()
  ↓
Render EnhancedPatientRecordsTable
```

### Medical History Parsing Flow

```
User Pastes Medical Notes
  ↓
Click "Validate & Sort with AI"
  ↓
API: POST /api/doctor/parse-history
  Request: { patientId, medicalNotes }
  ↓
Backend: AI Parsing Service
  ↓
Extract Structured Data
  ├── Demographics
  ├── Conditions
  ├── Lab Values
  └── Medications
  ↓
Compare with Existing Patient Data
  ↓
Create Audit Logs for Discrepancies
  ↓
Extract & Normalize Conditions
  ↓
Save Conditions to Database
  ↓
Response: { parsedData, conditions, auditLogs }
  ↓
Frontend: Display Extracted Conditions
  ↓
User Reviews & Adds to Patient
```

### AI Validation Workflow

```
User Pastes Medical Notes in Validation Panel
  ↓
API: POST /api/doctor/intelligent-parse
  ↓
Backend: Patient Matching Service
  ├── Fuzzy Search by Name/Email/NHS/MRN
  ├── Calculate Confidence Scores
  └── Return Top Matches
  ↓
If Multiple Matches:
  → Display Match Selection UI
  → User Selects Patient
  ↓
AI Data Extraction (BioGPT/Ollama)
  ↓
Map to Database Fields
  ↓
Compare with Existing Data
  ↓
Create Audit Logs
  ↓
API: GET /api/doctor/patients/:id/audit-logs
  ↓
Display Audit Logs in UI
  ↓
Clinician Reviews Each Change
  ↓
Approve/Reject Individual or All
  ↓
On Approve:
  → API: POST /api/doctor/audit-logs/:id/approve
  → Update Patient Record
  → Mark Log as Approved
```

### HbA1c Adjustment Calculation Flow

```
User Opens HbA1c Modal
  ↓
Enters Measured HbA1c & Weight
  ↓
Enters Medication Doses
  ↓
Click "Calculate Adjustment"
  ↓
API: POST /api/doctor/hba1c-adjust
  Request: { measuredHbA1cPercent, weightKg, medications }
  ↓
Backend: hba1cService.calculateAdjustedHbA1c()
  ↓
For Each Medication:
  ├── Get maxDose & factor
  ├── Calculate: (dose / maxDose) * factor
  └── Add to MES
  ↓
Calculate Adjusted Values:
  ├── adjustedHbA1cPercent = measured + MES
  └── adjustedHbA1cMmolMol = convert(adjustedPercent)
  ↓
Response: { MES, adjustedHbA1cPercent, adjustedHbA1cMmolMol }
  ↓
Display Results in Modal
```

### Analytics Calculation Flow

```
AnalyticsPanel Receives Patients Prop
  ↓
useMemo Hook Triggers
  ↓
Calculate Demographics:
  ├── Total Patients Count
  ├── Average Age
  └── Gender Distribution
  ↓
Calculate Medical Metrics:
  ├── Average HbA1c
  ├── Average MES
  └── Improvement Rate (% with negative change)
  ↓
Analyze Conditions:
  ├── Count Each Condition
  ├── Sort by Frequency
  └── Top 5 Conditions
  ↓
Percentile Analysis:
  ├── Sort by Selected Metric
  ├── Calculate Change %
  └── Identify Trend Direction
  ↓
Return Analytics Object
  ↓
Render Statistics Cards & Tables
```

---

## Security Considerations

### Authentication
- JWT tokens with 7-day expiration
- Tokens include user ID, email, role, hospitalCode
- Token validation on every API request
- Automatic logout on token expiration

### Authorization
- Role-based access control (clinician role required)
- Hospital code isolation (patients filtered by hospitalCode)
- Patients can only be accessed by clinicians with matching hospitalCode
- No cross-hospital data access

### Data Privacy
- Password hashing with bcrypt (SALT_ROUNDS)
- Secure token storage in localStorage
- HTTPS recommended for production
- Audit logs track all AI-suggested changes

### Input Validation
- Request body validation on backend
- SQL injection prevention via Prisma ORM
- XSS prevention in frontend rendering
- Type checking in TypeScript components

---

## Performance Optimizations

### Frontend
- `useMemo` for expensive calculations (AnalyticsPanel)
- Pagination for large patient tables
- Lazy loading of modal components
- Debounced search inputs
- React component memoization

### Backend
- Database indexes on frequently queried fields
- Prisma query optimization (select specific fields)
- Parallel API calls where possible (Promise.all)
- Efficient filtering at database level

### Data Loading
- Batch patient data loading
- Include only necessary relations
- Limit metric queries (latest only)
- CSV export uses streaming for large datasets

---

## Error Handling

### Frontend Error Handling
- Try-catch blocks around API calls
- User-friendly error messages
- Loading states during async operations
- Graceful fallbacks for missing data
- Error boundaries for component errors

### Backend Error Handling
- Try-catch in all controller functions
- Standardized error responses
- HTTP status codes (400, 401, 403, 404, 500)
- Error logging to console/files
- Validation error messages

### Common Error Scenarios
1. **Authentication Failure:** Redirect to login
2. **Authorization Failure:** 403 Forbidden
3. **Patient Not Found:** 404 with message
4. **Invalid Input:** 400 with validation details
5. **Server Error:** 500 with generic message

---

## Future Enhancements

### Planned Features
1. **Advanced Filtering:** Save filter presets
2. **Export Options:** PDF reports, Excel export
3. **Real-time Updates:** WebSocket for live data
4. **Bulk Operations:** Multi-patient actions
5. **Custom Dashboards:** User-configurable layouts
6. **Advanced Analytics:** Statistical analysis, predictive models
7. **Integration:** EHR system connections
8. **Mobile Support:** Responsive design improvements

### Technical Improvements
1. **Caching:** Redis for frequently accessed data
2. **Search:** Elasticsearch for advanced search
3. **Notifications:** Real-time alerts for new data
4. **Audit Trail:** Complete activity logging
5. **Backup:** Automated database backups
6. **Monitoring:** Application performance monitoring

---

## Conclusion

The Clinician Dashboard is a comprehensive, feature-rich application designed to support healthcare professionals in managing patient data, analyzing medical metrics, and leveraging AI for clinical decision support. The architecture is modular, scalable, and follows best practices for security, performance, and user experience.

**Key Strengths:**
- Comprehensive patient data management
- AI-powered medical text parsing
- Advanced analytics and visualization
- Secure role-based access control
- Intuitive user interface
- Robust error handling

**Architecture Highlights:**
- Clean separation of frontend/backend
- Service layer for business logic
- Database abstraction via Prisma
- RESTful API design
- Component-based React architecture

This report documents every feature, its architecture, and how it works within the overall system. For specific implementation details, refer to the source code files mentioned throughout this document.

---

*Report Generated: $(date)*
*Version: 1.0*






