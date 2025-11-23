# MedTrack - Final Project Structure

## 📁 Directory Structure

```
medtrack/
├── frontend/                    # Vite + React + TypeScript frontend
│   ├── src/                     # Frontend source code
│   │   ├── api.js              # API client (updated to use /api)
│   │   └── ...                  # Other frontend files
│   ├── package.json
│   └── vite.config.js
│
├── api/                         # Vercel serverless functions
│   ├── lib/                     # Shared utilities
│   │   ├── prisma.ts           # Prisma client singleton
│   │   └── auth.ts             # Auth utilities
│   ├── auth/                    # Authentication endpoints
│   │   ├── login.ts
│   │   ├── signup.ts
│   │   └── me.ts
│   ├── doctor/                  # Clinician endpoints
│   │   └── patients.ts
│   ├── medications/             # Medication endpoints
│   │   └── validateMedication.ts
│   ├── meds/                    # User medication endpoints
│   │   ├── user.ts             # GET/POST user medications
│   │   ├── schedule.ts
│   │   └── cycles.ts
│   ├── metrics/                 # Health metrics
│   │   └── user.ts
│   ├── health.ts                # Health check
│   ├── test-public.ts           # Public test endpoint
│   ├── hello.ts                 # Hello world
│   ├── health-metrics.ts
│   ├── medication-schedules.ts
│   ├── prisma/                  # Prisma schema
│   │   └── schema.prisma
│   ├── utils/                   # Backend utilities (copied)
│   ├── package.json
│   └── tsconfig.json
│
├── package.json                 # Root package.json
├── vercel.json                   # Vercel configuration
└── README.md                     # Comprehensive documentation
```

## ✅ Completed Conversions

### Core Infrastructure
- ✅ Root package.json with dev scripts
- ✅ vercel.json configuration
- ✅ Prisma client singleton pattern
- ✅ Auth utilities
- ✅ Frontend API client updated to use relative paths

### Converted Routes
- ✅ `/api/health` - Health check
- ✅ `/api/test-public` - Public test
- ✅ `/api/hello` - Hello world
- ✅ `/api/auth/login` - Login
- ✅ `/api/auth/signup` - Signup
- ✅ `/api/auth/me` - Get current user
- ✅ `/api/doctor/patients` - Get all patients
- ✅ `/api/medications/validateMedication` - Validate medication
- ✅ `/api/meds/user` - GET/POST user medications
- ✅ `/api/meds/schedule` - Medication schedule
- ✅ `/api/meds/cycles` - Medication cycles
- ✅ `/api/metrics/user` - User metrics
- ✅ `/api/health-metrics` - Health metrics
- ✅ `/api/medication-schedules` - Medication schedules

## 🔄 Remaining Routes to Convert

### From simple-server.js
- `/api/doctor/parse-history` - Parse medical history (complex)
- `/api/doctor/intelligent-parse` - AI-powered parsing
- `/api/doctor/patients/:patientId` - Update patient
- `/api/doctor/audit-logs/:logId/approve` - Approve audit log
- `/api/doctor/audit-logs/:logId/reject` - Reject audit log
- `/api/doctor/patients/:patientId/audit-logs` - Get audit logs
- `/api/auth/survey-status` - Survey completion status
- `/api/auth/survey-data` - Save survey data
- `/api/auth/complete-survey` - Mark survey complete
- `/api/metrics/patient/:patientId` - Patient metrics
- `/api/lab-results/patient/:patientId` - Lab results
- `/api/vital-signs/patient/:patientId` - Vital signs
- `/api/ai/status` - AI status
- `/api/ai/models` - AI models
- `/api/ai/assistant` - AI assistant
- `/api/ai/health-report` - Health report

### From backend/src/routes/
- `ai.js` → `/api/ai/*` routes
- `medication-tracking.js` → `/api/medications/*` routes
- `health-metrics.js` → `/api/health-metrics/*` routes
- `medication-schedules.js` → `/api/medication-schedules/*` routes
- `encounters.js` → `/api/encounters/*`
- `soap-notes.js` → `/api/soap-notes/*`
- `problems.js` → `/api/problems/*`
- `allergies.js` → `/api/allergies/*`
- `immunizations.js` → `/api/immunizations/*`
- `prescriptions.js` → `/api/prescriptions/*`
- `billing.js` → `/api/billing/*`
- `drug-interactions.js` → `/api/drug-interactions/*`
- `side-effects.js` → `/api/side-effects/*`
- `adherence.js` → `/api/adherence/*`
- `patient-profiles.js` → `/api/patient-profiles/*`
- `diary.js` → `/api/diary/*`
- `pill-recognition.js` → `/api/pill-recognition/*`
- `monopharmacy.js` → `/api/mono_se/*`
- `polypharmacy.js` → `/api/poly_se/*`

## 🛠️ Utilities to Migrate

Copy and adapt from `backend/utils/`:
- `intelligentMedicalParser.js`
- `ollamaParser.js`
- `biogptClient.js`
- `medicationMatchingService.js`
- Other utilities as needed

## 📝 Next Steps

1. Continue converting remaining routes from `simple-server.js`
2. Convert route files from `backend/src/routes/`
3. Migrate utilities to `/api/utils/` or `/api/lib/`
4. Test all endpoints locally with `vercel dev`
5. Deploy to Vercel and test in production
6. Update frontend to handle any API changes

## 🚀 Deployment

```bash
# Install dependencies
npm run install:all

# Generate Prisma client
cd api && npm run prisma:generate

# Deploy to Vercel
vercel --prod
```

See README.md for detailed instructions.
