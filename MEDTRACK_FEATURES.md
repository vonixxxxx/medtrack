# MedTrack Registration & Dashboard Features

## Overview

MedTrack is a comprehensive health tracking application that provides detailed registration forms, health metrics calculation, and a comprehensive dashboard for healthcare professionals and patients. The system implements evidence-based questionnaires and calculations for various health assessments.

## Core Features

### 1. Registration Form

#### Core Demographics
- **Date of Birth**: Automatically calculates current age to one decimal place
- **Biological Sex at Birth**: Male/Female/Other options
- **Ethnicity**: Office for National Statistics (ONS) categories

#### Female-Specific Health Assessment
- **Menses**: Yes/No with detailed follow-up questions
  - Age at menarche (8-20 years)
  - Menstrual regularity (Regular/Irregular)
  - Date of last menstrual period
  - Cycle length (20-45 days)
  - Period duration (1-14 days)
  - Contraception use and type
- **Pregnancy History**: Previous pregnancies (Yes/No)
- **Menopause Status**: 
  - Perimenopausal (Yes/No)
  - Postmenopausal (Yes/No)
  - Age at menopause (30-70 years)
  - Menopause type (Natural/Early/Premature Ovarian Insufficiency/Surgical/Induced)
  - HRT use and type

#### Male-Specific Health Assessment
- **IIEF-5 Questionnaire**: Erectile dysfunction screening (score 5-25)
- **Low Testosterone Symptoms**: Comprehensive symptom checklist
  - Low libido/reduced morning erections
  - Fatigue/low energy
  - Depressed mood/irritability
  - Reduced muscle mass/strength
  - Increased body fat
  - Reduced shaving frequency/body hair
  - Decreased bone strength
- **Red Flag Questions**: Critical health indicators
  - Gynecomastia
  - Testicular atrophy
  - Infertility
  - Pituitary disease
  - Head trauma
  - Chemotherapy/radiation exposure

#### Lifestyle Assessment
- **AUDIT Questionnaire**: Alcohol use disorders identification test (score 0-40)
- **Smoking Status**: Never/Current/Ex-smoker/Vaping
  - Age started smoking (10-80 years)
  - Cigarettes per day (1-100)
  - Auto-calculated pack years: (cigarettes per day ÷ 20) × years smoked
- **Vaping Information** (Appendix 1):
  - Device information
  - Nicotine concentration (mg/mL)
  - PG/VG ratio
  - Usage pattern
  - PSECDI score (0-20)
  - Readiness to quit (0-10)
- **IPAQ Questionnaire**: International Physical Activity Questionnaire (score 0-100)

#### Anthropometrics & Vitals
- **Weight**: Kilograms (mandatory)
- **Height**: Meters (mandatory)
- **Circumference Measurements** (optional):
  - Waist circumference (cm)
  - Hip circumference (cm)
  - Neck circumference (cm)
- **Blood Pressure**: Formatted as "120/80 mmHg"

### 2. Auto-Calculations

The system automatically calculates and displays:

- **BMI**: Weight / height²
- **WHR**: Waist ÷ hip
- **WHtR**: Waist ÷ height
- **BRI**: Body Roundness Index using the formula from MDCalc
- **Pack Years**: For smoking history
- **Age**: Current age to one decimal place

### 3. Dashboard Features

#### Overview Tab
- Key metrics cards with color-coded risk categories
- Age, BMI, AUDIT score, and IPAQ score summaries
- Risk categorization for each metric

#### Demographics Tab
- Complete demographic information
- Sex-specific health data display
- Conditional rendering based on biological sex

#### Health Metrics Tab
- Calculated health indices with risk categories
- Vital signs and measurements
- Color-coded risk indicators

#### Lifestyle Tab
- Alcohol and exercise assessments
- Smoking and vaping information
- Risk categorization for AUDIT and IPAQ scores

#### Anthropometrics Tab
- Raw measurements and calculated ratios
- Risk categories for WHR based on biological sex
- Comprehensive measurement overview

### 4. CSV Export

The dashboard provides comprehensive CSV export functionality including:
- All demographic data
- Female/male-specific health information
- Lifestyle assessment scores
- Anthropometric measurements
- Calculated health indices
- Risk categories

### 5. Medication Library Integration

#### BNF Integration
- British National Formulary (BNF) medication search
- BNF codes and direct links to BNF website
- Comprehensive medication information

#### EMC Integration
- Electronic Medicines Compendium (EMC) integration
- Product IDs and direct links to EMC website
- Detailed prescribing information

#### Search Features
- Medication name, type, and description search
- Recent search history
- Detailed medication profiles
- External source linking

## Technical Implementation

### Backend
- **Prisma Schema**: Comprehensive database model with all required fields
- **Validation**: Joi schema validation with conditional requirements
- **Calculations**: Utility functions for health metrics
- **API Endpoints**: Registration completion, status checking, and updates

### Frontend
- **React Components**: Modular, reusable components
- **Form Validation**: Step-by-step validation with error handling
- **Responsive Design**: Mobile-first approach with Tailwind CSS
- **State Management**: React hooks for form state and navigation
- **Animations**: Framer Motion for smooth transitions

### Database Schema
The system uses a comprehensive Prisma schema with:
- User demographics and health data
- Conditional fields based on biological sex
- Calculated fields for performance
- JSON fields for complex data structures

## Risk Categorization

### BMI Categories
- Underweight: < 18.5
- Normal weight: 18.5 - 24.9
- Overweight: 25.0 - 29.9
- Obese Class I: 30.0 - 34.9
- Obese Class II: 35.0 - 39.9
- Obese Class III: ≥ 40.0

### WHR Risk Categories
- **Male**: Low risk (< 0.9), Moderate risk (0.9 - 1.0), High risk (> 1.0)
- **Female**: Low risk (< 0.8), Moderate risk (0.8 - 0.85), High risk (> 0.85)

### AUDIT Risk Categories
- Low risk: ≤ 7
- Medium risk: 8 - 15
- High risk: 16 - 19
- Very high risk: ≥ 20

### IPAQ Activity Categories
- Low activity: < 600
- Moderate activity: 600 - 2999
- High activity: ≥ 3000

## Usage Instructions

### For Patients
1. Complete the 5-step registration process
2. Answer all required questions based on biological sex
3. Provide accurate measurements for calculations
4. Review dashboard for health insights
5. Export data for healthcare providers

### For Healthcare Professionals
1. Access patient dashboard for comprehensive health overview
2. Review risk categorizations and calculated indices
3. Export patient data for clinical records
4. Use medication library for prescribing decisions
5. Monitor patient progress over time

## Data Privacy & Security

- All data is stored securely with encryption
- User authentication required for access
- Data export limited to authenticated users
- Compliance with healthcare data protection regulations

## Future Enhancements

- Integration with electronic health records (EHR)
- Real-time data synchronization
- Advanced analytics and trend analysis
- Mobile application development
- API integration with external health systems
- Machine learning for risk prediction

## Support & Documentation

For technical support or feature requests, please refer to the main application documentation or contact the development team.

---

*MedTrack is designed to provide comprehensive health assessment and tracking capabilities while maintaining the highest standards of data security and user experience.*
