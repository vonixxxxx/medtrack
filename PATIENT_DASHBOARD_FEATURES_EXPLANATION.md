# Patient Dashboard Medical Records Features - Explanation

## Overview

The Patient Dashboard includes four medical record tiles that allow patients to view their health information. These features are **read-only** for patients (they can view but not edit), while clinicians have full CRUD (Create, Read, Update, Delete) capabilities.

---

## 1. Problem List

### What It Is
The **Problem List** displays all active and historical medical conditions, diagnoses, and health problems for the patient. This is a standard component of Electronic Health Records (EHR) systems.

### Purpose
- Track chronic conditions (e.g., Type 2 Diabetes, Hypertension)
- Monitor active vs. resolved problems
- Provide a comprehensive view of patient health history
- Support clinical decision-making

### Data Displayed
- **Problem Title**: Name of the condition (e.g., "Type 2 Diabetes")
- **ICD-10 Code**: Standardized medical coding for billing and classification
- **Status**: Active, Resolved, or Inactive
- **Severity**: Mild, Moderate, or Severe
- **Dates**: When the problem began and when it was resolved (if applicable)
- **Notes**: Additional clinical notes about the problem

### Source
This feature was copied from **OpenEMR**, an open-source Electronic Health Records system. OpenEMR is widely used in healthcare and follows industry standards for medical record keeping.

**OpenEMR Reference**: The problem list is a core component of OpenEMR's patient chart, allowing clinicians to maintain an active list of patient diagnoses and conditions.

---

## 2. Allergies

### What It Is
The **Allergies** section displays all known allergies and adverse reactions the patient has to medications, foods, environmental factors, or other substances.

### Purpose
- **Critical Safety Feature**: Prevents prescribing medications that could cause allergic reactions
- Track allergy severity and reactions
- Monitor allergy status (active vs. resolved)
- Support medication safety checks

### Data Displayed
- **Allergen**: The substance causing the allergy (e.g., "Penicillin", "Peanuts")
- **Allergen Type**: Drug, Food, Environmental, or Other
- **Reaction**: What happens when exposed (e.g., "Hives", "Difficulty breathing")
- **Severity**: Mild, Moderate, or Severe
- **Onset Date**: When the allergy was first identified
- **Status**: Active, Resolved, or Inactive
- **Notes**: Additional details about the allergy

### Source
This feature was copied from **OpenEMR**. Allergy tracking is a critical safety feature in all EHR systems and is required for clinical decision support.

**OpenEMR Reference**: OpenEMR's allergy module allows clinicians to document patient allergies with detailed information including allergen type, reaction, and severity.

---

## 3. Immunizations

### What It Is
The **Immunizations** section displays the patient's complete vaccination history, including all vaccines received, dates administered, and relevant details.

### Purpose
- Track vaccination history for public health compliance
- Ensure patients are up-to-date on required immunizations
- Support preventive care and public health reporting
- Provide vaccination records for travel, school, or employment

### Data Displayed
- **Vaccine Name**: Name of the vaccine (e.g., "COVID-19 Vaccine", "Influenza")
- **CVX Code**: Standardized vaccine coding system
- **Administration Date**: When the vaccine was given
- **Route**: How it was administered (IM, SC, Oral, etc.)
- **Site**: Where it was administered (e.g., "Left arm")
- **Dose**: Amount given (e.g., "0.5 mL")
- **Manufacturer**: Vaccine manufacturer (e.g., "Pfizer", "Moderna")
- **Lot Number**: Batch number for tracking
- **Provider**: Who administered the vaccine
- **Notes**: Additional information

### Source
This feature was copied from **OpenEMR**. Immunization tracking is essential for preventive care and public health reporting.

**OpenEMR Reference**: OpenEMR's immunization module follows CDC standards for vaccine tracking and includes support for CVX codes and vaccine information systems (VIS).

---

## 4. Prescriptions

### What It Is
The **Prescriptions** section displays all current and historical prescriptions that have been prescribed to the patient by healthcare providers.

### Purpose
- View current medications and dosages
- Track prescription history
- Monitor medication status (active, filled, cancelled, expired)
- Support medication reconciliation
- Provide medication information for other healthcare providers

### Data Displayed
- **Medication Name**: Name of the prescribed medication
- **Dosage**: Amount and unit (e.g., "500mg")
- **Frequency**: How often to take (e.g., "Once daily", "Twice daily")
- **Route**: How to take it (Oral, Topical, Injection, etc.)
- **Prescribed Date**: When the prescription was written
- **Status**: Active, Filled, Cancelled, or Expired
- **Refills**: Number of refills remaining
- **Instructions**: Special instructions for taking the medication

### Source
This feature was copied from **OpenEMR**. Prescription management is a core feature of all EHR systems.

**OpenEMR Reference**: OpenEMR's prescription module allows clinicians to prescribe medications, track refills, and manage prescription status. It integrates with medication databases and supports e-prescribing.

---

## Why These Features Are Read-Only for Patients

These medical records are **read-only** for patients because:

1. **Clinical Accuracy**: Medical diagnoses, allergies, and prescriptions must be documented by licensed healthcare providers to ensure accuracy and legal compliance.

2. **Patient Safety**: Incorrect information could lead to dangerous medical decisions. Only qualified clinicians should modify medical records.

3. **Regulatory Compliance**: Healthcare regulations (HIPAA, FDA, etc.) require that medical records be maintained by licensed professionals.

4. **Standard Practice**: This follows industry-standard EHR design where patients can view their records but cannot edit clinical data.

---

## Where These Features Were Copied From

All four features were copied from **OpenEMR** (Open Electronic Medical Records), which is:

- **Open-source EHR system** used by thousands of healthcare providers worldwide
- **HIPAA-compliant** and follows healthcare industry standards
- **Comprehensive** with modules for all aspects of patient care
- **Well-documented** with extensive feature sets

### OpenEMR Modules Referenced:
1. **Problem List Module**: `openemr/interface/patient_file/summary/problems.php`
2. **Allergy Module**: `openemr/interface/patient_file/summary/allergies.php`
3. **Immunization Module**: `openemr/interface/patient_file/summary/immunizations.php`
4. **Prescription Module**: `openemr/interface/patient_file/summary/prescriptions.php`

These modules were analyzed and their functionality was reimplemented in MedTrack using:
- **Modern React components** with the MedTrack design system
- **RESTful API endpoints** following OpenEMR's data models
- **Prisma database schema** based on OpenEMR's database structure

---

## Technical Implementation

### Backend
- **Controllers**: `backend/src/controllers/problemController.js`, `allergyController.js`, `immunizationController.js`, `prescriptionController.js`
- **Routes**: `backend/src/routes/problems.js`, `allergies.js`, `immunizations.js`, `prescriptions.js`
- **Database Models**: Prisma models for `Problem`, `Allergy`, `Immunization`, `Prescription`

### Frontend
- **Components**: 
  - `frontend/src/components/problems/ProblemList.jsx`
  - `frontend/src/components/allergies/AllergyList.jsx`
  - `frontend/src/components/immunizations/ImmunizationList.jsx`
  - `frontend/src/components/prescriptions/PrescriptionList.jsx`
- **API Methods**: All methods in `frontend/src/api.js` for fetching these records

---

## Summary

These four tiles provide patients with a comprehensive view of their medical records:
- **Problem List**: Medical conditions and diagnoses
- **Allergies**: Known allergies and reactions
- **Immunizations**: Vaccination history
- **Prescriptions**: Current and past medications

All features are read-only for patients (view-only) and were copied from OpenEMR, an industry-standard open-source EHR system, ensuring they follow healthcare best practices and regulatory requirements.



