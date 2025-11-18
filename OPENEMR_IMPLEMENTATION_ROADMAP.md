# OpenEMR Feature Implementation Roadmap for MedTrack

**Generated:** November 2025  
**Target Stack:** Flask API + Supabase + SvelteKit Frontend  
**Compliance:** HIPAA/GDPR Compliant

---

## Executive Summary

This document provides a comprehensive roadmap for implementing OpenEMR features into the MedTrack application. The implementation follows a modular, phased approach ensuring HIPAA/GDPR compliance at every stage.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Phase 1: Core Foundation](#phase-1-core-foundation)
3. [Phase 2: Patient Management](#phase-2-patient-management)
4. [Phase 3: Clinical Documentation](#phase-3-clinical-documentation)
5. [Phase 4: Scheduling & Appointments](#phase-4-scheduling--appointments)
6. [Phase 5: Billing & Financial](#phase-5-billing--financial)
7. [Phase 6: Advanced Features](#phase-6-advanced-features)
8. [Security & Compliance](#security--compliance)
9. [API Design](#api-design)
10. [Database Schema](#database-schema)

---

## Architecture Overview

### Technology Stack

**Backend:**
- **Framework:** Flask (Python)
- **Database:** Supabase (PostgreSQL)
- **ORM:** SQLAlchemy
- **Authentication:** Supabase Auth + JWT
- **API:** RESTful API + FHIR R4 (future)

**Frontend:**
- **Framework:** SvelteKit
- **UI Library:** Tailwind CSS + shadcn-svelte
- **State Management:** Svelte stores
- **Charts:** Recharts / Chart.js

**Infrastructure:**
- **Hosting:** Vercel (Frontend) + Railway/Render (Backend)
- **Storage:** Supabase Storage (documents)
- **CDN:** Cloudflare
- **Monitoring:** Sentry

### Architecture Principles

1. **Modular Design:** Each feature is a self-contained module
2. **API-First:** All features exposed via REST API
3. **Security by Default:** HIPAA/GDPR compliance built-in
4. **Scalable:** Designed for horizontal scaling
5. **Audit Trail:** All actions logged for compliance

---

## Phase 1: Core Foundation

### 1.1 Database Schema Setup

**Supabase Tables:**

```sql
-- Users and Authentication (extends Supabase auth.users)
CREATE TABLE profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id),
  email TEXT UNIQUE NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('patient', 'clinician', 'admin')),
  first_name TEXT,
  last_name TEXT,
  phone TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Facilities/Organizations
CREATE TABLE facilities (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  address TEXT,
  city TEXT,
  state TEXT,
  zip_code TEXT,
  phone TEXT,
  email TEXT,
  npi TEXT, -- National Provider Identifier
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Providers/Clinicians
CREATE TABLE providers (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES profiles(id),
  facility_id UUID REFERENCES facilities(id),
  npi TEXT,
  specialty TEXT,
  license_number TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Audit Log (HIPAA Compliance)
CREATE TABLE audit_log (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES profiles(id),
  action TEXT NOT NULL,
  resource_type TEXT NOT NULL,
  resource_id UUID,
  ip_address INET,
  user_agent TEXT,
  details JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Row Level Security Policies
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE facilities ENABLE ROW LEVEL SECURITY;
ALTER TABLE providers ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_log ENABLE ROW LEVEL SECURITY;
```

**Implementation Steps:**

1. Create Supabase project
2. Run migration scripts
3. Set up Row Level Security (RLS) policies
4. Create database functions for audit logging
5. Set up database triggers for automatic audit logging

**Flask API Endpoints:**

```python
# app/routes/auth.py
@bp.route('/auth/register', methods=['POST'])
@bp.route('/auth/login', methods=['POST'])
@bp.route('/auth/logout', methods=['POST'])
@bp.route('/auth/refresh', methods=['POST'])
@bp.route('/auth/me', methods=['GET'])
```

**SvelteKit Components:**

- `src/routes/auth/login/+page.svelte`
- `src/routes/auth/register/+page.svelte`
- `src/lib/stores/auth.ts`

**Timeline:** 2 weeks

---

## Phase 2: Patient Management

### 2.1 Patient Demographics

**Supabase Tables:**

```sql
-- Patients
CREATE TABLE patients (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES profiles(id),
  facility_id UUID REFERENCES facilities(id),
  mrn TEXT, -- Medical Record Number
  first_name TEXT NOT NULL,
  last_name TEXT NOT NULL,
  middle_name TEXT,
  date_of_birth DATE NOT NULL,
  sex TEXT CHECK (sex IN ('M', 'F', 'O', 'U')),
  gender_identity TEXT,
  race TEXT,
  ethnicity TEXT,
  preferred_language TEXT,
  marital_status TEXT,
  ssn TEXT, -- Encrypted
  address_line1 TEXT,
  address_line2 TEXT,
  city TEXT,
  state TEXT,
  zip_code TEXT,
  country TEXT DEFAULT 'US',
  phone_home TEXT,
  phone_mobile TEXT,
  phone_work TEXT,
  email TEXT,
  emergency_contact_name TEXT,
  emergency_contact_phone TEXT,
  emergency_contact_relation TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  created_by UUID REFERENCES profiles(id),
  updated_by UUID REFERENCES profiles(id)
);

-- Patient Insurance
CREATE TABLE patient_insurance (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
  insurance_type TEXT CHECK (insurance_type IN ('primary', 'secondary', 'tertiary')),
  insurance_company_id UUID REFERENCES insurance_companies(id),
  policy_number TEXT,
  group_number TEXT,
  subscriber_name TEXT,
  subscriber_ssn TEXT, -- Encrypted
  subscriber_dob DATE,
  copay_amount DECIMAL(10,2),
  effective_date DATE,
  expiration_date DATE,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insurance Companies
CREATE TABLE insurance_companies (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  payer_id TEXT, -- CMS Payer ID
  address TEXT,
  phone TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Patient Employers
CREATE TABLE patient_employers (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
  employer_name TEXT,
  address TEXT,
  phone TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Patient History (Audit Trail)
CREATE TABLE patient_history (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
  field_name TEXT NOT NULL,
  old_value TEXT,
  new_value TEXT,
  changed_by UUID REFERENCES profiles(id),
  changed_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Flask API Endpoints:**

```python
# app/routes/patients.py
@bp.route('/api/patients', methods=['GET', 'POST'])
@bp.route('/api/patients/<uuid:patient_id>', methods=['GET', 'PUT', 'DELETE'])
@bp.route('/api/patients/<uuid:patient_id>/insurance', methods=['GET', 'POST'])
@bp.route('/api/patients/<uuid:patient_id>/employer', methods=['GET', 'POST'])
@bp.route('/api/patients/search', methods=['GET'])
@bp.route('/api/patients/<uuid:patient_id>/history', methods=['GET'])
```

**SvelteKit Components:**

- `src/routes/patients/+page.svelte` - Patient list
- `src/routes/patients/[id]/+page.svelte` - Patient detail
- `src/routes/patients/[id]/edit/+page.svelte` - Edit patient
- `src/lib/components/patient/PatientForm.svelte`
- `src/lib/components/patient/PatientSearch.svelte`
- `src/lib/components/patient/InsuranceForm.svelte`

**Features to Implement:**

1. Patient registration form
2. Patient search (by name, MRN, DOB, SSN)
3. Patient demographics editing
4. Insurance management
5. Employer information
6. Patient merge (duplicate management)
7. Patient labels/printing
8. Patient photo upload

**Timeline:** 3 weeks

### 2.2 Patient Portal Integration

**Features:**

- Patient self-registration
- Patient login
- View own medical records
- Update demographics
- View appointments
- Request appointments
- Secure messaging

**Timeline:** 2 weeks

---

## Phase 3: Clinical Documentation

### 3.1 Encounters

**Supabase Tables:**

```sql
-- Encounters
CREATE TABLE encounters (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
  provider_id UUID REFERENCES providers(id),
  facility_id UUID REFERENCES facilities(id),
  encounter_date DATE NOT NULL,
  encounter_time TIME,
  encounter_type TEXT, -- office, inpatient, outpatient, emergency
  reason TEXT,
  billing_facility_id UUID REFERENCES facilities(id),
  class_code TEXT,
  class_display TEXT,
  status TEXT DEFAULT 'finished', -- planned, arrived, triaged, in-progress, finished, cancelled
  priority TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  created_by UUID REFERENCES profiles(id),
  updated_by UUID REFERENCES profiles(id)
);

-- SOAP Notes
CREATE TABLE soap_notes (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  encounter_id UUID REFERENCES encounters(id) ON DELETE CASCADE,
  subjective TEXT,
  objective TEXT,
  assessment TEXT,
  plan TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  created_by UUID REFERENCES profiles(id),
  updated_by UUID REFERENCES profiles(id)
);
```

**Flask API Endpoints:**

```python
# app/routes/encounters.py
@bp.route('/api/encounters', methods=['GET', 'POST'])
@bp.route('/api/encounters/<uuid:encounter_id>', methods=['GET', 'PUT', 'DELETE'])
@bp.route('/api/encounters/<uuid:encounter_id>/soap', methods=['GET', 'POST', 'PUT'])
@bp.route('/api/patients/<uuid:patient_id>/encounters', methods=['GET'])
```

**SvelteKit Components:**

- `src/routes/encounters/+page.svelte`
- `src/routes/encounters/[id]/+page.svelte`
- `src/lib/components/encounter/EncounterForm.svelte`
- `src/lib/components/encounter/SoapNoteEditor.svelte`

**Timeline:** 2 weeks

### 3.2 Clinical Forms

**Supabase Tables:**

```sql
-- Vitals
CREATE TABLE vitals (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  encounter_id UUID REFERENCES encounters(id) ON DELETE CASCADE,
  patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
  date DATE NOT NULL,
  time TIME,
  height DECIMAL(5,2), -- cm
  weight DECIMAL(5,2), -- kg
  bmi DECIMAL(4,1),
  temperature DECIMAL(4,1), -- Celsius
  pulse INTEGER,
  respiration INTEGER,
  systolic_bp INTEGER,
  diastolic_bp INTEGER,
  oxygen_saturation DECIMAL(4,1),
  head_circumference DECIMAL(5,2),
  waist_circumference DECIMAL(5,2),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  created_by UUID REFERENCES profiles(id)
);

-- Review of Systems
CREATE TABLE review_of_systems (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  encounter_id UUID REFERENCES encounters(id) ON DELETE CASCADE,
  patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
  constitutional TEXT,
  eyes TEXT,
  ent TEXT,
  cardiovascular TEXT,
  respiratory TEXT,
  gastrointestinal TEXT,
  genitourinary TEXT,
  musculoskeletal TEXT,
  integumentary TEXT,
  neurological TEXT,
  psychiatric TEXT,
  endocrine TEXT,
  hematologic TEXT,
  allergic TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  created_by UUID REFERENCES profiles(id)
);

-- Custom Forms (Layout Based Forms)
CREATE TABLE form_data (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  encounter_id UUID REFERENCES encounters(id) ON DELETE CASCADE,
  patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
  form_name TEXT NOT NULL,
  field_id TEXT NOT NULL,
  field_value TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (id, field_id)
);
```

**Flask API Endpoints:**

```python
# app/routes/forms.py
@bp.route('/api/vitals', methods=['GET', 'POST'])
@bp.route('/api/vitals/<uuid:vital_id>', methods=['GET', 'PUT'])
@bp.route('/api/ros', methods=['GET', 'POST'])
@bp.route('/api/forms/<form_name>', methods=['GET', 'POST'])
```

**Timeline:** 3 weeks

### 3.3 Problem List

**Supabase Tables:**

```sql
-- Problems/Conditions
CREATE TABLE problems (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
  encounter_id UUID REFERENCES encounters(id),
  title TEXT NOT NULL,
  code TEXT, -- ICD-10 code
  code_type TEXT DEFAULT 'ICD10',
  begin_date DATE,
  end_date DATE,
  status TEXT DEFAULT 'active', -- active, resolved, inactive
  severity TEXT,
  notes TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  created_by UUID REFERENCES profiles(id)
);
```

**Timeline:** 1 week

### 3.4 Medications

**Supabase Tables:**

```sql
-- Medications
CREATE TABLE medications (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
  encounter_id UUID REFERENCES encounters(id),
  medication_name TEXT NOT NULL,
  ndc_code TEXT, -- National Drug Code
  rxnorm_code TEXT,
  dosage TEXT,
  unit TEXT,
  route TEXT, -- oral, topical, injection, etc.
  frequency TEXT,
  quantity DECIMAL(10,2),
  refills INTEGER DEFAULT 0,
  start_date DATE,
  end_date DATE,
  status TEXT DEFAULT 'active', -- active, discontinued, completed
  instructions TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  created_by UUID REFERENCES profiles(id)
);

-- Medication History
CREATE TABLE medication_history (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  medication_id UUID REFERENCES medications(id) ON DELETE CASCADE,
  action TEXT, -- started, stopped, changed, refilled
  notes TEXT,
  changed_by UUID REFERENCES profiles(id),
  changed_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Timeline:** 2 weeks

### 3.5 Allergies

**Supabase Tables:**

```sql
-- Allergies
CREATE TABLE allergies (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
  allergen TEXT NOT NULL,
  allergen_type TEXT, -- drug, food, environmental, other
  reaction TEXT,
  severity TEXT, -- mild, moderate, severe
  onset_date DATE,
  status TEXT DEFAULT 'active', -- active, resolved, inactive
  notes TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  created_by UUID REFERENCES profiles(id)
);
```

**Timeline:** 1 week

---

## Phase 4: Scheduling & Appointments

### 4.1 Appointment Scheduling

**Supabase Tables:**

```sql
-- Appointments
CREATE TABLE appointments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
  provider_id UUID REFERENCES providers(id),
  facility_id UUID REFERENCES facilities(id),
  appointment_date DATE NOT NULL,
  appointment_time TIME NOT NULL,
  duration INTEGER DEFAULT 30, -- minutes
  appointment_type TEXT, -- routine, follow-up, urgent, etc.
  status TEXT DEFAULT 'scheduled', -- scheduled, confirmed, checked-in, in-progress, completed, cancelled, no-show
  reason TEXT,
  notes TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  created_by UUID REFERENCES profiles(id)
);

-- Appointment Categories
CREATE TABLE appointment_categories (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  duration INTEGER DEFAULT 30,
  color TEXT,
  facility_id UUID REFERENCES facilities(id)
);

-- Recurring Appointments
CREATE TABLE recurring_appointments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
  provider_id UUID REFERENCES providers(id),
  facility_id UUID REFERENCES facilities(id),
  start_date DATE NOT NULL,
  end_date DATE,
  frequency TEXT, -- daily, weekly, monthly
  day_of_week INTEGER, -- 0-6 (Sunday-Saturday)
  day_of_month INTEGER, -- 1-31
  appointment_type TEXT,
  duration INTEGER DEFAULT 30,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Flask API Endpoints:**

```python
# app/routes/appointments.py
@bp.route('/api/appointments', methods=['GET', 'POST'])
@bp.route('/api/appointments/<uuid:appointment_id>', methods=['GET', 'PUT', 'DELETE'])
@bp.route('/api/appointments/calendar', methods=['GET'])
@bp.route('/api/appointments/available-slots', methods=['GET'])
@bp.route('/api/appointments/<uuid:appointment_id>/check-in', methods=['POST'])
@bp.route('/api/appointments/<uuid:appointment_id>/cancel', methods=['POST'])
```

**SvelteKit Components:**

- `src/routes/appointments/+page.svelte` - Calendar view
- `src/lib/components/appointments/Calendar.svelte`
- `src/lib/components/appointments/AppointmentForm.svelte`
- `src/lib/components/appointments/AppointmentList.svelte`

**Features:**

- Calendar view (day/week/month)
- Appointment creation/editing
- Recurring appointments
- Appointment reminders (email/SMS)
- Check-in functionality
- Waitlist management

**Timeline:** 3 weeks

---

## Phase 5: Billing & Financial

### 5.1 Billing System

**Supabase Tables:**

```sql
-- Charges
CREATE TABLE charges (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  encounter_id UUID REFERENCES encounters(id) ON DELETE CASCADE,
  patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
  code TEXT NOT NULL, -- CPT/HCPCS code
  code_type TEXT DEFAULT 'CPT',
  description TEXT,
  units DECIMAL(10,2) DEFAULT 1,
  fee DECIMAL(10,2) NOT NULL,
  date_of_service DATE NOT NULL,
  provider_id UUID REFERENCES providers(id),
  status TEXT DEFAULT 'pending', -- pending, billed, paid, denied, cancelled
  created_at TIMESTAMPTZ DEFAULT NOW(),
  created_by UUID REFERENCES profiles(id)
);

-- Payments
CREATE TABLE payments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
  encounter_id UUID REFERENCES encounters(id),
  amount DECIMAL(10,2) NOT NULL,
  payment_method TEXT, -- cash, check, credit_card, insurance
  payment_date DATE NOT NULL,
  check_number TEXT,
  credit_card_last4 TEXT,
  notes TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  created_by UUID REFERENCES profiles(id)
);

-- Payment Allocations
CREATE TABLE payment_allocations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  payment_id UUID REFERENCES payments(id) ON DELETE CASCADE,
  charge_id UUID REFERENCES charges(id) ON DELETE CASCADE,
  amount DECIMAL(10,2) NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insurance Claims
CREATE TABLE claims (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
  insurance_id UUID REFERENCES patient_insurance(id),
  claim_number TEXT UNIQUE,
  total_charges DECIMAL(10,2),
  total_paid DECIMAL(10,2),
  status TEXT DEFAULT 'pending', -- pending, submitted, paid, denied, rejected
  submitted_date DATE,
  paid_date DATE,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Claim Lines
CREATE TABLE claim_lines (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  claim_id UUID REFERENCES claims(id) ON DELETE CASCADE,
  charge_id UUID REFERENCES charges(id),
  line_number INTEGER,
  code TEXT,
  units DECIMAL(10,2),
  fee DECIMAL(10,2),
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Flask API Endpoints:**

```python
# app/routes/billing.py
@bp.route('/api/charges', methods=['GET', 'POST'])
@bp.route('/api/charges/<uuid:charge_id>', methods=['GET', 'PUT', 'DELETE'])
@bp.route('/api/payments', methods=['GET', 'POST'])
@bp.route('/api/claims', methods=['GET', 'POST'])
@bp.route('/api/claims/<uuid:claim_id>', methods=['GET', 'PUT'])
@bp.route('/api/claims/<uuid:claim_id>/submit', methods=['POST'])
@bp.route('/api/patients/<uuid:patient_id>/statement', methods=['GET'])
```

**Timeline:** 4 weeks

---

## Phase 6: Advanced Features

### 6.1 Document Management

**Supabase Tables:**

```sql
-- Documents
CREATE TABLE documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
  encounter_id UUID REFERENCES encounters(id),
  document_type TEXT, -- lab_result, imaging, letter, etc.
  category TEXT,
  title TEXT NOT NULL,
  file_path TEXT NOT NULL, -- Supabase Storage path
  file_name TEXT NOT NULL,
  file_size BIGINT,
  mime_type TEXT,
  date DATE,
  status TEXT DEFAULT 'active',
  created_at TIMESTAMPTZ DEFAULT NOW(),
  created_by UUID REFERENCES profiles(id)
);
```

**Implementation:**

- Use Supabase Storage for file storage
- Implement file upload/download endpoints
- Add document viewer component
- Support PDF, images, text files

**Timeline:** 2 weeks

### 6.2 Laboratory & Orders

**Supabase Tables:**

```sql
-- Lab Orders
CREATE TABLE lab_orders (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
  encounter_id UUID REFERENCES encounters(id),
  provider_id UUID REFERENCES providers(id),
  order_date DATE NOT NULL,
  order_status TEXT DEFAULT 'ordered', -- ordered, collected, in-progress, completed, cancelled
  notes TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  created_by UUID REFERENCES profiles(id)
);

-- Lab Results
CREATE TABLE lab_results (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  lab_order_id UUID REFERENCES lab_orders(id) ON DELETE CASCADE,
  test_name TEXT NOT NULL,
  test_code TEXT, -- LOINC code
  result_value TEXT,
  result_unit TEXT,
  reference_range TEXT,
  abnormal_flag TEXT, -- normal, high, low, critical
  result_date DATE,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Timeline:** 2 weeks

### 6.3 Prescriptions (eRx)

**Supabase Tables:**

```sql
-- Prescriptions
CREATE TABLE prescriptions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
  encounter_id UUID REFERENCES encounters(id),
  provider_id UUID REFERENCES providers(id),
  medication_name TEXT NOT NULL,
  ndc_code TEXT,
  rxnorm_code TEXT,
  dosage TEXT,
  quantity DECIMAL(10,2),
  refills INTEGER DEFAULT 0,
  instructions TEXT,
  pharmacy_id UUID REFERENCES pharmacies(id),
  status TEXT DEFAULT 'active', -- active, filled, cancelled, expired
  date_prescribed DATE NOT NULL,
  date_filled DATE,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  created_by UUID REFERENCES profiles(id)
);

-- Pharmacies
CREATE TABLE pharmacies (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  npi TEXT,
  address TEXT,
  phone TEXT,
  fax TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Timeline:** 2 weeks

### 6.4 Messaging System

**Supabase Tables:**

```sql
-- Messages
CREATE TABLE messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  from_user_id UUID REFERENCES profiles(id),
  to_user_id UUID REFERENCES profiles(id),
  patient_id UUID REFERENCES patients(id),
  subject TEXT,
  message TEXT NOT NULL,
  priority TEXT DEFAULT 'normal', -- low, normal, high, urgent
  status TEXT DEFAULT 'unread', -- unread, read, archived
  created_at TIMESTAMPTZ DEFAULT NOW(),
  read_at TIMESTAMPTZ
);

-- Message Attachments
CREATE TABLE message_attachments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  message_id UUID REFERENCES messages(id) ON DELETE CASCADE,
  file_path TEXT NOT NULL,
  file_name TEXT NOT NULL,
  file_size BIGINT,
  mime_type TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Timeline:** 2 weeks

---

## Security & Compliance

### HIPAA Compliance Features

1. **Access Controls:**
   - Role-based access control (RBAC)
   - Row-level security (RLS) in Supabase
   - API endpoint authorization

2. **Audit Logging:**
   - All data access logged
   - All modifications logged
   - User activity tracking
   - Failed login attempts

3. **Data Encryption:**
   - Encryption at rest (Supabase)
   - Encryption in transit (TLS/SSL)
   - Encrypted sensitive fields (SSN, etc.)

4. **Authentication:**
   - Multi-factor authentication (MFA)
   - Session management
   - Password policies
   - Account lockout

5. **Data Integrity:**
   - Database constraints
   - Transaction management
   - Backup and recovery

### GDPR Compliance Features

1. **Data Export:**
   - Patient data export (JSON/XML)
   - Complete medical record export

2. **Data Deletion:**
   - Right to be forgotten
   - Anonymization procedures
   - Data retention policies

3. **Consent Management:**
   - Consent tracking
   - Consent withdrawal

4. **Privacy Controls:**
   - Data minimization
   - Purpose limitation
   - Access restrictions

### Implementation

**Flask Middleware:**

```python
# app/middleware/audit.py
def audit_log_middleware():
    # Log all API requests
    pass

# app/middleware/auth.py
def require_auth():
    # Verify JWT token
    # Check user permissions
    pass

# app/middleware/encryption.py
def encrypt_sensitive_fields():
    # Encrypt SSN, etc.
    pass
```

**Supabase RLS Policies:**

```sql
-- Example: Patients can only see their own data
CREATE POLICY "Patients can view own data"
ON patients FOR SELECT
USING (auth.uid() = user_id);

-- Example: Clinicians can view patients in their facility
CREATE POLICY "Clinicians can view facility patients"
ON patients FOR SELECT
USING (
  EXISTS (
    SELECT 1 FROM providers
    WHERE providers.user_id = auth.uid()
    AND providers.facility_id = patients.facility_id
  )
);
```

---

## API Design

### RESTful API Structure

**Base URL:** `https://api.medtrack.com/v1`

**Authentication:**
- Header: `Authorization: Bearer <token>`
- Token obtained via Supabase Auth

**Response Format:**
```json
{
  "success": true,
  "data": {},
  "meta": {
    "page": 1,
    "per_page": 20,
    "total": 100
  },
  "errors": []
}
```

**Error Format:**
```json
{
  "success": false,
  "errors": [
    {
      "code": "VALIDATION_ERROR",
      "message": "Invalid input",
      "field": "email"
    }
  ]
}
```

### API Endpoints Summary

**Authentication:**
- `POST /auth/register`
- `POST /auth/login`
- `POST /auth/logout`
- `POST /auth/refresh`
- `GET /auth/me`

**Patients:**
- `GET /patients`
- `POST /patients`
- `GET /patients/:id`
- `PUT /patients/:id`
- `DELETE /patients/:id`
- `GET /patients/search`

**Encounters:**
- `GET /encounters`
- `POST /encounters`
- `GET /encounters/:id`
- `PUT /encounters/:id`

**Appointments:**
- `GET /appointments`
- `POST /appointments`
- `GET /appointments/:id`
- `PUT /appointments/:id`
- `DELETE /appointments/:id`

**Billing:**
- `GET /charges`
- `POST /charges`
- `GET /payments`
- `POST /payments`
- `GET /claims`
- `POST /claims`

---

## Database Schema

### Complete Schema Overview

**Core Tables:**
- `profiles` - User profiles
- `facilities` - Healthcare facilities
- `providers` - Healthcare providers
- `patients` - Patient demographics
- `audit_log` - Audit trail

**Clinical Tables:**
- `encounters` - Patient encounters
- `soap_notes` - SOAP documentation
- `vitals` - Vital signs
- `problems` - Problem list
- `medications` - Medications
- `allergies` - Allergies
- `immunizations` - Immunizations
- `procedures` - Procedures

**Scheduling Tables:**
- `appointments` - Appointments
- `appointment_categories` - Appointment types
- `recurring_appointments` - Recurring appointments

**Billing Tables:**
- `charges` - Service charges
- `payments` - Payments
- `payment_allocations` - Payment distribution
- `claims` - Insurance claims
- `claim_lines` - Claim line items

**Other Tables:**
- `documents` - Document management
- `lab_orders` - Laboratory orders
- `lab_results` - Laboratory results
- `prescriptions` - Prescriptions
- `messages` - Internal messaging
- `insurance_companies` - Insurance companies
- `patient_insurance` - Patient insurance
- `pharmacies` - Pharmacies

---

## Implementation Timeline

### Phase 1: Foundation (Weeks 1-2)
- Database setup
- Authentication system
- Basic API structure
- Security middleware

### Phase 2: Patient Management (Weeks 3-5)
- Patient demographics
- Patient search
- Insurance management
- Patient portal

### Phase 3: Clinical Documentation (Weeks 6-10)
- Encounters
- SOAP notes
- Clinical forms
- Problem list
- Medications
- Allergies

### Phase 4: Scheduling (Weeks 11-13)
- Appointment scheduling
- Calendar view
- Appointment reminders

### Phase 5: Billing (Weeks 14-17)
- Charge capture
- Payment processing
- Insurance claims
- Financial reports

### Phase 6: Advanced Features (Weeks 18-23)
- Document management
- Laboratory orders
- Prescriptions
- Messaging system
- Reporting

**Total Timeline:** ~6 months

---

## Next Steps

1. **Review and Approve:** Review this roadmap with stakeholders
2. **Set Up Infrastructure:** Create Supabase project, set up Flask API
3. **Begin Phase 1:** Start with core foundation
4. **Iterate:** Implement features incrementally with testing
5. **Compliance Review:** Regular HIPAA/GDPR compliance audits

---

**Document Version:** 1.0  
**Last Updated:** November 2025  
**Status:** Ready for Implementation



