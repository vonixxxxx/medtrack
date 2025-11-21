const express = require('express');
const axios = require('axios');
const router = express.Router();
require('dotenv').config();

// AI Backend URLs
const AI_BACKEND_URL = process.env.AI_BACKEND_URL || 'http://localhost:5003';
const HUGGINGFACE_API_KEY = process.env.HUGGINGFACE_API_KEY;
const HUGGINGFACE_API_URL = 'https://api-inference.huggingface.co/models/microsoft/BioGPT-Large';

// Comprehensive medication database with enhanced matching
const medicationDatabase = [
  // Pain Relief & Anti-inflammatory
  { generic_name: 'acetaminophen', brand_names: ['Tylenol', 'Panadol', 'Excedrin'], drug_class: 'Analgesic/Antipyretic', dosage_forms: ['tablet', 'capsule', 'liquid', 'suppository'], typical_strengths: ['250mg', '500mg', '650mg', '1000mg'], indications: ['pain', 'fever', 'headache'] },
  { generic_name: 'ibuprofen', brand_names: ['Advil', 'Motrin', 'Nurofen'], drug_class: 'NSAID', dosage_forms: ['tablet', 'capsule', 'liquid', 'gel'], typical_strengths: ['200mg', '400mg', '600mg', '800mg'], indications: ['pain', 'inflammation', 'fever'] },
  { generic_name: 'naproxen', brand_names: ['Aleve', 'Naprosyn'], drug_class: 'NSAID', dosage_forms: ['tablet', 'capsule', 'gel'], typical_strengths: ['220mg', '440mg', '500mg'], indications: ['pain', 'inflammation', 'arthritis'] },
  { generic_name: 'aspirin', brand_names: ['Bayer', 'Bufferin', 'Ecotrin'], drug_class: 'NSAID/Antiplatelet', dosage_forms: ['tablet', 'chewable', 'suppository'], typical_strengths: ['81mg', '325mg', '500mg'], indications: ['pain', 'fever', 'heart protection'] },
  { generic_name: 'celecoxib', brand_names: ['Celebrex'], drug_class: 'COX-2 Inhibitor', dosage_forms: ['capsule'], typical_strengths: ['100mg', '200mg', '400mg'], indications: ['arthritis', 'pain', 'inflammation'] },
  
  // Diabetes Medications
  { generic_name: 'metformin', brand_names: ['Glucophage', 'Fortamet', 'Glumetza'], drug_class: 'Biguanide', dosage_forms: ['tablet', 'extended-release'], typical_strengths: ['500mg', '850mg', '1000mg'], indications: ['type 2 diabetes', 'prediabetes'] },
  { generic_name: 'insulin', brand_names: ['Humalog', 'Novolog', 'Lantus', 'Tresiba'], drug_class: 'Hormone', dosage_forms: ['injection', 'pen', 'pump'], typical_strengths: ['10 units', '20 units', '30 units', '40 units', '50 units'], indications: ['diabetes', 'hyperglycemia'] },
  { generic_name: 'sitagliptin', brand_names: ['Januvia'], drug_class: 'DPP-4 Inhibitor', dosage_forms: ['tablet'], typical_strengths: ['25mg', '50mg', '100mg'], indications: ['type 2 diabetes'] },
  { generic_name: 'pioglitazone', brand_names: ['Actos'], drug_class: 'Thiazolidinedione', dosage_forms: ['tablet'], typical_strengths: ['15mg', '30mg', '45mg'], indications: ['type 2 diabetes'] },
  { generic_name: 'glipizide', brand_names: ['Glucotrol'], drug_class: 'Sulfonylurea', dosage_forms: ['tablet', 'extended-release'], typical_strengths: ['5mg', '10mg'], indications: ['type 2 diabetes'] },
  { generic_name: 'semaglutide', brand_names: ['Ozempic', 'Wegovy', 'Rybelsus'], drug_class: 'GLP-1 Receptor Agonist', dosage_forms: ['injection', 'tablet'], typical_strengths: ['0.25mg', '0.5mg', '1mg', '2.4mg', '3mg', '7mg', '14mg'], indications: ['type 2 diabetes', 'weight management', 'obesity'] },
  
  // Cardiovascular
  { generic_name: 'lisinopril', brand_names: ['Prinivil', 'Zestril'], drug_class: 'ACE Inhibitor', dosage_forms: ['tablet'], typical_strengths: ['5mg', '10mg', '20mg', '40mg'], indications: ['hypertension', 'heart failure', 'heart attack prevention'] },
  { generic_name: 'atorvastatin', brand_names: ['Lipitor'], drug_class: 'Statin', dosage_forms: ['tablet'], typical_strengths: ['10mg', '20mg', '40mg', '80mg'], indications: ['high cholesterol', 'heart disease prevention'] },
  { generic_name: 'simvastatin', brand_names: ['Zocor'], drug_class: 'Statin', dosage_forms: ['tablet'], typical_strengths: ['10mg', '20mg', '40mg', '80mg'], indications: ['high cholesterol', 'heart disease prevention'] },
  { generic_name: 'amlodipine', brand_names: ['Norvasc'], drug_class: 'Calcium Channel Blocker', dosage_forms: ['tablet'], typical_strengths: ['2.5mg', '5mg', '10mg'], indications: ['hypertension', 'chest pain'] },
  { generic_name: 'metoprolol', brand_names: ['Lopressor', 'Toprol-XL'], drug_class: 'Beta Blocker', dosage_forms: ['tablet', 'extended-release'], typical_strengths: ['25mg', '50mg', '100mg', '200mg'], indications: ['hypertension', 'heart failure', 'chest pain'] },
  { generic_name: 'warfarin', brand_names: ['Coumadin', 'Jantoven'], drug_class: 'Anticoagulant', dosage_forms: ['tablet'], typical_strengths: ['1mg', '2mg', '2.5mg', '3mg', '4mg', '5mg', '6mg', '7.5mg', '10mg'], indications: ['blood clots', 'stroke prevention', 'atrial fibrillation'] },
  
  // Gastrointestinal
  { generic_name: 'omeprazole', brand_names: ['Prilosec'], drug_class: 'Proton Pump Inhibitor', dosage_forms: ['capsule', 'tablet'], typical_strengths: ['10mg', '20mg', '40mg'], indications: ['acid reflux', 'ulcers', 'GERD'] },
  { generic_name: 'esomeprazole', brand_names: ['Nexium'], drug_class: 'Proton Pump Inhibitor', dosage_forms: ['capsule', 'tablet'], typical_strengths: ['20mg', '40mg'], indications: ['acid reflux', 'ulcers', 'GERD'] },
  { generic_name: 'ranitidine', brand_names: ['Zantac'], drug_class: 'H2 Blocker', dosage_forms: ['tablet', 'liquid'], typical_strengths: ['75mg', '150mg', '300mg'], indications: ['acid reflux', 'ulcers'] },
  { generic_name: 'famotidine', brand_names: ['Pepcid'], drug_class: 'H2 Blocker', dosage_forms: ['tablet', 'liquid'], typical_strengths: ['10mg', '20mg', '40mg'], indications: ['acid reflux', 'ulcers'] },
  
  // Mental Health
  { generic_name: 'sertraline', brand_names: ['Zoloft'], drug_class: 'SSRI', dosage_forms: ['tablet', 'liquid'], typical_strengths: ['25mg', '50mg', '100mg', '200mg'], indications: ['depression', 'anxiety', 'panic disorder', 'PTSD'] },
  { generic_name: 'amphetamine/dextroamphetamine', brand_names: ['Adderall', 'Adderall XR'], drug_class: 'Stimulant', dosage_forms: ['tablet', 'extended-release capsule'], typical_strengths: ['5mg', '10mg', '15mg', '20mg', '25mg', '30mg'], indications: ['ADHD', 'narcolepsy'] },
  { generic_name: 'fluoxetine', brand_names: ['Prozac'], drug_class: 'SSRI', dosage_forms: ['tablet', 'capsule', 'liquid'], typical_strengths: ['10mg', '20mg', '40mg'], indications: ['depression', 'anxiety', 'OCD', 'bulimia'] },
  { generic_name: 'escitalopram', brand_names: ['Lexapro'], drug_class: 'SSRI', dosage_forms: ['tablet', 'liquid'], typical_strengths: ['5mg', '10mg', '20mg'], indications: ['depression', 'anxiety'] },
  { generic_name: 'venlafaxine', brand_names: ['Effexor'], drug_class: 'SNRI', dosage_forms: ['tablet', 'extended-release'], typical_strengths: ['37.5mg', '75mg', '150mg', '225mg'], indications: ['depression', 'anxiety', 'panic disorder'] },
  { generic_name: 'bupropion', brand_names: ['Wellbutrin'], drug_class: 'NDRI', dosage_forms: ['tablet', 'extended-release'], typical_strengths: ['75mg', '100mg', '150mg', '300mg'], indications: ['depression', 'smoking cessation', 'ADHD'] },
  { generic_name: 'alprazolam', brand_names: ['Xanax'], drug_class: 'Benzodiazepine', dosage_forms: ['tablet', 'extended-release'], typical_strengths: ['0.25mg', '0.5mg', '1mg', '2mg'], indications: ['anxiety', 'panic disorder'] },
  { generic_name: 'lorazepam', brand_names: ['Ativan'], drug_class: 'Benzodiazepine', dosage_forms: ['tablet', 'liquid', 'injection'], typical_strengths: ['0.5mg', '1mg', '2mg'], indications: ['anxiety', 'seizures', 'insomnia'] },
  
  // Antibiotics
  { generic_name: 'amoxicillin', brand_names: ['Amoxil', 'Trimox'], drug_class: 'Penicillin Antibiotic', dosage_forms: ['capsule', 'tablet', 'liquid', 'chewable'], typical_strengths: ['250mg', '500mg', '875mg'], indications: ['bacterial infections', 'respiratory infections', 'urinary tract infections'] },
  { generic_name: 'azithromycin', brand_names: ['Zithromax', 'Z-Pak'], drug_class: 'Macrolide Antibiotic', dosage_forms: ['tablet', 'liquid', 'injection'], typical_strengths: ['250mg', '500mg'], indications: ['bacterial infections', 'respiratory infections', 'skin infections'] },
  { generic_name: 'ciprofloxacin', brand_names: ['Cipro'], drug_class: 'Fluoroquinolone Antibiotic', dosage_forms: ['tablet', 'liquid', 'injection'], typical_strengths: ['250mg', '500mg', '750mg'], indications: ['bacterial infections', 'urinary tract infections', 'respiratory infections'] },
  { generic_name: 'doxycycline', brand_names: ['Vibramycin', 'Doryx'], drug_class: 'Tetracycline Antibiotic', dosage_forms: ['capsule', 'tablet', 'liquid'], typical_strengths: ['50mg', '100mg'], indications: ['bacterial infections', 'acne', 'malaria prevention'] },
  
  // Respiratory
  { generic_name: 'albuterol', brand_names: ['Proventil', 'Ventolin'], drug_class: 'Bronchodilator', dosage_forms: ['inhaler', 'nebulizer', 'tablet'], typical_strengths: ['90mcg', '180mcg'], indications: ['asthma', 'COPD', 'bronchospasm'] },
  { generic_name: 'fluticasone', brand_names: ['Flonase', 'Flovent'], drug_class: 'Corticosteroid', dosage_forms: ['inhaler', 'nasal spray'], typical_strengths: ['44mcg', '110mcg', '220mcg'], indications: ['asthma', 'allergic rhinitis'] },
  { generic_name: 'montelukast', brand_names: ['Singulair'], drug_class: 'Leukotriene Receptor Antagonist', dosage_forms: ['tablet', 'chewable'], typical_strengths: ['4mg', '5mg', '10mg'], indications: ['asthma', 'allergic rhinitis'] },
  
  // Sleep & Sedation
  { generic_name: 'zolpidem', brand_names: ['Ambien'], drug_class: 'Non-benzodiazepine Hypnotic', dosage_forms: ['tablet', 'extended-release'], typical_strengths: ['5mg', '10mg', '12.5mg'], indications: ['insomnia', 'sleep disorders'] },
  { generic_name: 'trazodone', brand_names: ['Desyrel'], drug_class: 'SARI', dosage_forms: ['tablet'], typical_strengths: ['50mg', '100mg', '150mg', '300mg'], indications: ['depression', 'insomnia'] },
  { generic_name: 'melatonin', brand_names: ['Melatonin'], drug_class: 'Hormone', dosage_forms: ['tablet', 'capsule', 'liquid'], typical_strengths: ['1mg', '3mg', '5mg', '10mg'], indications: ['insomnia', 'jet lag', 'sleep disorders'] },
  
  // Vitamins & Supplements
  { generic_name: 'vitamin d', brand_names: ['Vitamin D3', 'D3'], drug_class: 'Vitamin', dosage_forms: ['tablet', 'capsule', 'liquid', 'gummy'], typical_strengths: ['400 IU', '1000 IU', '2000 IU', '5000 IU'], indications: ['vitamin d deficiency', 'bone health', 'immune support'] },
  { generic_name: 'vitamin b12', brand_names: ['B12', 'Cyanocobalamin'], drug_class: 'Vitamin', dosage_forms: ['tablet', 'capsule', 'sublingual', 'injection'], typical_strengths: ['500mcg', '1000mcg', '2500mcg', '5000mcg'], indications: ['b12 deficiency', 'anemia', 'nerve health'] },
  { generic_name: 'iron', brand_names: ['Iron', 'Ferrous Sulfate'], drug_class: 'Mineral', dosage_forms: ['tablet', 'capsule', 'liquid'], typical_strengths: ['18mg', '28mg', '65mg', '325mg'], indications: ['iron deficiency', 'anemia'] },
  { generic_name: 'calcium', brand_names: ['Calcium', 'Calcium Carbonate'], drug_class: 'Mineral', dosage_forms: ['tablet', 'chewable', 'liquid'], typical_strengths: ['500mg', '600mg', '1000mg', '1200mg'], indications: ['bone health', 'osteoporosis prevention'] },
  
  // Additional common medications
  { generic_name: 'diphenhydramine', brand_names: ['Benadryl'], drug_class: 'Antihistamine', dosage_forms: ['tablet', 'capsule', 'liquid', 'gel'], typical_strengths: ['25mg', '50mg'], indications: ['allergies', 'insomnia', 'motion sickness'] },
  { generic_name: 'loratadine', brand_names: ['Claritin'], drug_class: 'Antihistamine', dosage_forms: ['tablet', 'liquid', 'chewable'], typical_strengths: ['10mg'], indications: ['allergies', 'hay fever'] },
  { generic_name: 'fexofenadine', brand_names: ['Allegra'], drug_class: 'Antihistamine', dosage_forms: ['tablet', 'capsule', 'liquid'], typical_strengths: ['30mg', '60mg', '120mg', '180mg'], indications: ['allergies', 'hay fever'] },
  { generic_name: 'cetirizine', brand_names: ['Zyrtec'], drug_class: 'Antihistamine', dosage_forms: ['tablet', 'liquid', 'chewable'], typical_strengths: ['5mg', '10mg'], indications: ['allergies', 'hay fever'] },
  { generic_name: 'pseudoephedrine', brand_names: ['Sudafed'], drug_class: 'Decongestant', dosage_forms: ['tablet', 'liquid'], typical_strengths: ['30mg', '60mg', '120mg'], indications: ['nasal congestion', 'sinus pressure'] },
  { generic_name: 'phenylephrine', brand_names: ['Sudafed PE'], drug_class: 'Decongestant', dosage_forms: ['tablet', 'liquid'], typical_strengths: ['5mg', '10mg'], indications: ['nasal congestion', 'sinus pressure'] },
  { generic_name: 'dextromethorphan', brand_names: ['Robitussin', 'Delsym'], drug_class: 'Cough Suppressant', dosage_forms: ['tablet', 'liquid', 'syrup'], typical_strengths: ['15mg', '30mg'], indications: ['cough', 'cold symptoms'] },
  { generic_name: 'guaifenesin', brand_names: ['Mucinex'], drug_class: 'Expectorant', dosage_forms: ['tablet', 'liquid', 'syrup'], typical_strengths: ['200mg', '400mg', '600mg'], indications: ['cough', 'chest congestion'] },
  { generic_name: 'acetaminophen/codeine', brand_names: ['Tylenol with Codeine'], drug_class: 'Opioid Analgesic', dosage_forms: ['tablet', 'liquid'], typical_strengths: ['300mg/15mg', '300mg/30mg', '300mg/60mg'], indications: ['pain', 'cough'] },
  { generic_name: 'hydrocodone/acetaminophen', brand_names: ['Vicodin', 'Norco'], drug_class: 'Opioid Analgesic', dosage_forms: ['tablet'], typical_strengths: ['5mg/325mg', '7.5mg/325mg', '10mg/325mg'], indications: ['pain'] },
  { generic_name: 'oxycodone', brand_names: ['OxyContin', 'Percocet'], drug_class: 'Opioid Analgesic', dosage_forms: ['tablet', 'capsule'], typical_strengths: ['5mg', '10mg', '15mg', '20mg', '30mg'], indications: ['pain'] },
  { generic_name: 'morphine', brand_names: ['MS Contin', 'Roxanol'], drug_class: 'Opioid Analgesic', dosage_forms: ['tablet', 'liquid', 'injection'], typical_strengths: ['15mg', '30mg', '60mg', '100mg'], indications: ['pain'] },
  { generic_name: 'fentanyl', brand_names: ['Duragesic', 'Actiq'], drug_class: 'Opioid Analgesic', dosage_forms: ['patch', 'lozenge', 'injection'], typical_strengths: ['12.5mcg', '25mcg', '50mcg', '75mcg', '100mcg'], indications: ['severe pain'] },
  { generic_name: 'tramadol', brand_names: ['Ultram'], drug_class: 'Opioid Analgesic', dosage_forms: ['tablet', 'extended-release'], typical_strengths: ['50mg', '100mg', '200mg', '300mg'], indications: ['pain'] },
  { generic_name: 'gabapentin', brand_names: ['Neurontin'], drug_class: 'Anticonvulsant', dosage_forms: ['capsule', 'tablet', 'liquid'], typical_strengths: ['100mg', '300mg', '400mg', '600mg', '800mg'], indications: ['seizures', 'nerve pain', 'anxiety'] },
  { generic_name: 'pregabalin', brand_names: ['Lyrica'], drug_class: 'Anticonvulsant', dosage_forms: ['capsule', 'liquid'], typical_strengths: ['25mg', '50mg', '75mg', '100mg', '150mg', '200mg', '225mg', '300mg'], indications: ['nerve pain', 'fibromyalgia', 'anxiety'] },
  { generic_name: 'duloxetine', brand_names: ['Cymbalta'], drug_class: 'SNRI', dosage_forms: ['capsule'], typical_strengths: ['20mg', '30mg', '60mg'], indications: ['depression', 'anxiety', 'nerve pain', 'fibromyalgia'] },
  { generic_name: 'venlafaxine', brand_names: ['Effexor'], drug_class: 'SNRI', dosage_forms: ['tablet', 'extended-release'], typical_strengths: ['37.5mg', '75mg', '150mg', '225mg'], indications: ['depression', 'anxiety', 'panic disorder'] },
  { generic_name: 'mirtazapine', brand_names: ['Remeron'], drug_class: 'Tetracyclic Antidepressant', dosage_forms: ['tablet'], typical_strengths: ['7.5mg', '15mg', '30mg', '45mg'], indications: ['depression', 'anxiety', 'insomnia'] },
  { generic_name: 'quetiapine', brand_names: ['Seroquel'], drug_class: 'Atypical Antipsychotic', dosage_forms: ['tablet', 'extended-release'], typical_strengths: ['25mg', '50mg', '100mg', '200mg', '300mg', '400mg'], indications: ['schizophrenia', 'bipolar disorder', 'depression'] },
  { generic_name: 'aripiprazole', brand_names: ['Abilify'], drug_class: 'Atypical Antipsychotic', dosage_forms: ['tablet', 'liquid', 'injection'], typical_strengths: ['2mg', '5mg', '10mg', '15mg', '20mg', '30mg'], indications: ['schizophrenia', 'bipolar disorder', 'depression', 'Tourette syndrome'] },
  { generic_name: 'olanzapine', brand_names: ['Zyprexa'], drug_class: 'Atypical Antipsychotic', dosage_forms: ['tablet', 'injection'], typical_strengths: ['2.5mg', '5mg', '7.5mg', '10mg', '15mg', '20mg'], indications: ['schizophrenia', 'bipolar disorder'] },
  { generic_name: 'risperidone', brand_names: ['Risperdal'], drug_class: 'Atypical Antipsychotic', dosage_forms: ['tablet', 'liquid', 'injection'], typical_strengths: ['0.25mg', '0.5mg', '1mg', '2mg', '3mg', '4mg'], indications: ['schizophrenia', 'bipolar disorder', 'autism'] },
  { generic_name: 'haloperidol', brand_names: ['Haldol'], drug_class: 'Typical Antipsychotic', dosage_forms: ['tablet', 'liquid', 'injection'], typical_strengths: ['0.5mg', '1mg', '2mg', '5mg', '10mg', '20mg'], indications: ['schizophrenia', 'Tourette syndrome', 'agitation'] },
  { generic_name: 'chlorpromazine', brand_names: ['Thorazine'], drug_class: 'Typical Antipsychotic', dosage_forms: ['tablet', 'liquid', 'injection'], typical_strengths: ['10mg', '25mg', '50mg', '100mg', '200mg'], indications: ['schizophrenia', 'bipolar disorder', 'nausea', 'vomiting'] },
  { generic_name: 'thioridazine', brand_names: ['Mellaril'], drug_class: 'Typical Antipsychotic', dosage_forms: ['tablet', 'liquid'], typical_strengths: ['10mg', '25mg', '50mg', '100mg', '150mg', '200mg'], indications: ['schizophrenia', 'agitation'] },
  { generic_name: 'fluphenazine', brand_names: ['Prolixin'], drug_class: 'Typical Antipsychotic', dosage_forms: ['tablet', 'liquid', 'injection'], typical_strengths: ['1mg', '2.5mg', '5mg', '10mg'], indications: ['schizophrenia', 'agitation'] },
  { generic_name: 'perphenazine', brand_names: ['Trilafon'], drug_class: 'Typical Antipsychotic', dosage_forms: ['tablet', 'liquid'], typical_strengths: ['2mg', '4mg', '8mg', '16mg'], indications: ['schizophrenia', 'agitation', 'nausea', 'vomiting'] },
  { generic_name: 'trifluoperazine', brand_names: ['Stelazine'], drug_class: 'Typical Antipsychotic', dosage_forms: ['tablet', 'liquid'], typical_strengths: ['1mg', '2mg', '5mg', '10mg'], indications: ['schizophrenia', 'agitation'] },
  { generic_name: 'mesoridazine', brand_names: ['Serentil'], drug_class: 'Typical Antipsychotic', dosage_forms: ['tablet', 'liquid'], typical_strengths: ['25mg', '50mg', '100mg'], indications: ['schizophrenia', 'agitation'] },
  { generic_name: 'prochlorperazine', brand_names: ['Compazine'], drug_class: 'Typical Antipsychotic', dosage_forms: ['tablet', 'suppository', 'injection'], typical_strengths: ['5mg', '10mg', '15mg', '25mg'], indications: ['nausea', 'vomiting', 'agitation'] },
  { generic_name: 'promethazine', brand_names: ['Phenergan'], drug_class: 'Phenothiazine', dosage_forms: ['tablet', 'suppository', 'injection'], typical_strengths: ['12.5mg', '25mg', '50mg'], indications: ['nausea', 'vomiting', 'allergies', 'sedation'] },
  { generic_name: 'hydroxyzine', brand_names: ['Atarax', 'Vistaril'], drug_class: 'Antihistamine', dosage_forms: ['tablet', 'capsule', 'liquid', 'injection'], typical_strengths: ['10mg', '25mg', '50mg', '100mg'], indications: ['anxiety', 'allergies', 'nausea', 'vomiting'] },
  { generic_name: 'diphenhydramine', brand_names: ['Benadryl'], drug_class: 'Antihistamine', dosage_forms: ['tablet', 'capsule', 'liquid', 'gel'], typical_strengths: ['25mg', '50mg'], indications: ['allergies', 'insomnia', 'motion sickness'] },
  { generic_name: 'doxylamine', brand_names: ['Unisom'], drug_class: 'Antihistamine', dosage_forms: ['tablet', 'capsule'], typical_strengths: ['25mg'], indications: ['insomnia', 'motion sickness'] },
  { generic_name: 'cyclizine', brand_names: ['Marezine'], drug_class: 'Antihistamine', dosage_forms: ['tablet', 'injection'], typical_strengths: ['25mg', '50mg'], indications: ['motion sickness', 'nausea', 'vomiting'] },
  { generic_name: 'meclizine', brand_names: ['Antivert', 'Bonine'], drug_class: 'Antihistamine', dosage_forms: ['tablet', 'chewable'], typical_strengths: ['12.5mg', '25mg', '50mg'], indications: ['motion sickness', 'vertigo'] },
  { generic_name: 'dimenhydrinate', brand_names: ['Dramamine'], drug_class: 'Antihistamine', dosage_forms: ['tablet', 'chewable', 'liquid'], typical_strengths: ['50mg'], indications: ['motion sickness', 'nausea', 'vomiting'] },
  { generic_name: 'scopolamine', brand_names: ['Transderm Scop'], drug_class: 'Anticholinergic', dosage_forms: ['patch'], typical_strengths: ['1.5mg'], indications: ['motion sickness', 'nausea', 'vomiting'] },
  { generic_name: 'hyoscyamine', brand_names: ['Levsin'], drug_class: 'Anticholinergic', dosage_forms: ['tablet', 'liquid', 'injection'], typical_strengths: ['0.125mg', '0.25mg'], indications: ['irritable bowel syndrome', 'stomach cramps'] },
  { generic_name: 'atropine', brand_names: ['Atropine'], drug_class: 'Anticholinergic', dosage_forms: ['tablet', 'liquid', 'injection'], typical_strengths: ['0.4mg', '0.6mg'], indications: ['bradycardia', 'antidote for poisoning'] },
  { generic_name: 'glycopyrrolate', brand_names: ['Robinul'], drug_class: 'Anticholinergic', dosage_forms: ['tablet', 'injection'], typical_strengths: ['1mg', '2mg'], indications: ['peptic ulcers', 'excessive sweating'] },
  { generic_name: 'propantheline', brand_names: ['Pro-Banthine'], drug_class: 'Anticholinergic', dosage_forms: ['tablet'], typical_strengths: ['7.5mg', '15mg'], indications: ['peptic ulcers', 'irritable bowel syndrome'] },
  { generic_name: 'dicyclomine', brand_names: ['Bentyl'], drug_class: 'Anticholinergic', dosage_forms: ['capsule', 'liquid', 'injection'], typical_strengths: ['10mg', '20mg'], indications: ['irritable bowel syndrome', 'stomach cramps'] },
  { generic_name: 'belladonna', brand_names: ['Belladonna'], drug_class: 'Anticholinergic', dosage_forms: ['tablet', 'liquid'], typical_strengths: ['0.125mg'], indications: ['stomach cramps', 'irritable bowel syndrome'] },
  { generic_name: 'phenobarbital', brand_names: ['Phenobarbital'], drug_class: 'Barbiturate', dosage_forms: ['tablet', 'liquid', 'injection'], typical_strengths: ['15mg', '30mg', '60mg', '100mg'], indications: ['seizures', 'insomnia', 'anxiety'] },
  { generic_name: 'phenytoin', brand_names: ['Dilantin'], drug_class: 'Anticonvulsant', dosage_forms: ['capsule', 'tablet', 'liquid', 'injection'], typical_strengths: ['30mg', '100mg'], indications: ['seizures', 'epilepsy'] },
  { generic_name: 'carbamazepine', brand_names: ['Tegretol'], drug_class: 'Anticonvulsant', dosage_forms: ['tablet', 'chewable', 'liquid'], typical_strengths: ['100mg', '200mg', '400mg'], indications: ['seizures', 'epilepsy', 'bipolar disorder', 'nerve pain'] },
  { generic_name: 'oxcarbazepine', brand_names: ['Trileptal'], drug_class: 'Anticonvulsant', dosage_forms: ['tablet', 'liquid'], typical_strengths: ['150mg', '300mg', '600mg'], indications: ['seizures', 'epilepsy'] },
  { generic_name: 'valproic acid', brand_names: ['Depakene'], drug_class: 'Anticonvulsant', dosage_forms: ['capsule', 'liquid'], typical_strengths: ['250mg'], indications: ['seizures', 'epilepsy', 'bipolar disorder', 'migraine prevention'] },
  { generic_name: 'divalproex', brand_names: ['Depakote'], drug_class: 'Anticonvulsant', dosage_forms: ['tablet', 'sprinkle'], typical_strengths: ['125mg', '250mg', '500mg'], indications: ['seizures', 'epilepsy', 'bipolar disorder', 'migraine prevention'] },
  { generic_name: 'gabapentin', brand_names: ['Neurontin'], drug_class: 'Anticonvulsant', dosage_forms: ['capsule', 'tablet', 'liquid'], typical_strengths: ['100mg', '300mg', '400mg', '600mg', '800mg'], indications: ['seizures', 'nerve pain', 'anxiety'] },
  { generic_name: 'pregabalin', brand_names: ['Lyrica'], drug_class: 'Anticonvulsant', dosage_forms: ['capsule', 'liquid'], typical_strengths: ['25mg', '50mg', '75mg', '100mg', '150mg', '200mg', '225mg', '300mg'], indications: ['nerve pain', 'fibromyalgia', 'anxiety'] },
  { generic_name: 'topiramate', brand_names: ['Topamax'], drug_class: 'Anticonvulsant', dosage_forms: ['tablet', 'capsule'], typical_strengths: ['25mg', '50mg', '100mg', '200mg'], indications: ['seizures', 'epilepsy', 'migraine prevention', 'bipolar disorder'] },
  { generic_name: 'lamotrigine', brand_names: ['Lamictal'], drug_class: 'Anticonvulsant', dosage_forms: ['tablet', 'chewable'], typical_strengths: ['25mg', '50mg', '100mg', '150mg', '200mg'], indications: ['seizures', 'epilepsy', 'bipolar disorder'] },
  { generic_name: 'levetiracetam', brand_names: ['Keppra'], drug_class: 'Anticonvulsant', dosage_forms: ['tablet', 'liquid', 'injection'], typical_strengths: ['250mg', '500mg', '750mg', '1000mg'], indications: ['seizures', 'epilepsy'] },
  { generic_name: 'zonisamide', brand_names: ['Zonegran'], drug_class: 'Anticonvulsant', dosage_forms: ['capsule'], typical_strengths: ['25mg', '50mg', '100mg'], indications: ['seizures', 'epilepsy'] },
  { generic_name: 'tiagabine', brand_names: ['Gabitril'], drug_class: 'Anticonvulsant', dosage_forms: ['tablet'], typical_strengths: ['2mg', '4mg', '12mg', '16mg'], indications: ['seizures', 'epilepsy'] },
  { generic_name: 'vigabatrin', brand_names: ['Sabril'], drug_class: 'Anticonvulsant', dosage_forms: ['tablet', 'powder'], typical_strengths: ['500mg'], indications: ['seizures', 'epilepsy'] },
  { generic_name: 'felbamate', brand_names: ['Felbatol'], drug_class: 'Anticonvulsant', dosage_forms: ['tablet', 'liquid'], typical_strengths: ['400mg', '600mg'], indications: ['seizures', 'epilepsy'] },
  { generic_name: 'ethosuximide', brand_names: ['Zarontin'], drug_class: 'Anticonvulsant', dosage_forms: ['capsule', 'liquid'], typical_strengths: ['250mg'], indications: ['seizures', 'epilepsy'] },
  { generic_name: 'methsuximide', brand_names: ['Celontin'], drug_class: 'Anticonvulsant', dosage_forms: ['capsule'], typical_strengths: ['150mg', '300mg'], indications: ['seizures', 'epilepsy'] },
  { generic_name: 'phensuximide', brand_names: ['Milontin'], drug_class: 'Anticonvulsant', dosage_forms: ['capsule'], typical_strengths: ['500mg'], indications: ['seizures', 'epilepsy'] },
  { generic_name: 'trimethadione', brand_names: ['Tridione'], drug_class: 'Anticonvulsant', dosage_forms: ['tablet', 'capsule'], typical_strengths: ['150mg', '300mg'], indications: ['seizures', 'epilepsy'] },
  { generic_name: 'paramethadione', brand_names: ['Paradione'], drug_class: 'Anticonvulsant', dosage_forms: ['tablet', 'capsule'], typical_strengths: ['150mg', '300mg'], indications: ['seizures', 'epilepsy'] },
  { generic_name: 'ethotoin', brand_names: ['Peganone'], drug_class: 'Anticonvulsant', dosage_forms: ['tablet'], typical_strengths: ['250mg', '500mg'], indications: ['seizures', 'epilepsy'] },
  { generic_name: 'mephenytoin', brand_names: ['Mesantoin'], drug_class: 'Anticonvulsant', dosage_forms: ['tablet'], typical_strengths: ['100mg'], indications: ['seizures', 'epilepsy'] },
  { generic_name: 'fosphenytoin', brand_names: ['Cerebyx'], drug_class: 'Anticonvulsant', dosage_forms: ['injection'], typical_strengths: ['50mg PE', '75mg PE'], indications: ['seizures', 'epilepsy'] },
  { generic_name: 'lacosamide', brand_names: ['Vimpat'], drug_class: 'Anticonvulsant', dosage_forms: ['tablet', 'liquid', 'injection'], typical_strengths: ['50mg', '100mg', '150mg', '200mg'], indications: ['seizures', 'epilepsy'] },
  { generic_name: 'rufinamide', brand_names: ['Banzel'], drug_class: 'Anticonvulsant', dosage_forms: ['tablet'], typical_strengths: ['200mg', '400mg'], indications: ['seizures', 'epilepsy'] },
  { generic_name: 'ezogabine', brand_names: ['Potiga'], drug_class: 'Anticonvulsant', dosage_forms: ['tablet'], typical_strengths: ['50mg', '200mg', '300mg', '400mg'], indications: ['seizures', 'epilepsy'] },
  { generic_name: 'perampanel', brand_names: ['Fycompa'], drug_class: 'Anticonvulsant', dosage_forms: ['tablet'], typical_strengths: ['2mg', '4mg', '6mg', '8mg', '10mg', '12mg'], indications: ['seizures', 'epilepsy'] },
  { generic_name: 'brivaracetam', brand_names: ['Briviact'], drug_class: 'Anticonvulsant', dosage_forms: ['tablet', 'liquid', 'injection'], typical_strengths: ['10mg', '25mg', '50mg', '75mg', '100mg'], indications: ['seizures', 'epilepsy'] },
  { generic_name: 'cenobamate', brand_names: ['Xcopri'], drug_class: 'Anticonvulsant', dosage_forms: ['tablet'], typical_strengths: ['12.5mg', '25mg', '50mg', '100mg', '150mg', '200mg'], indications: ['seizures', 'epilepsy'] },
  { generic_name: 'ganaxolone', brand_names: ['Ztalmy'], drug_class: 'Anticonvulsant', dosage_forms: ['liquid'], typical_strengths: ['200mg/ml'], indications: ['seizures', 'epilepsy'] },
  { generic_name: 'stiripentol', brand_names: ['Diacomit'], drug_class: 'Anticonvulsant', dosage_forms: ['capsule', 'powder'], typical_strengths: ['250mg', '500mg'], indications: ['seizures', 'epilepsy'] },
  { generic_name: 'cannabidiol', brand_names: ['Epidiolex'], drug_class: 'Anticonvulsant', dosage_forms: ['liquid'], typical_strengths: ['100mg/ml'], indications: ['seizures', 'epilepsy'] },
  { generic_name: 'clobazam', brand_names: ['Onfi'], drug_class: 'Benzodiazepine', dosage_forms: ['tablet', 'liquid'], typical_strengths: ['5mg', '10mg', '20mg'], indications: ['seizures', 'epilepsy'] },
  { generic_name: 'clonazepam', brand_names: ['Klonopin'], drug_class: 'Benzodiazepine', dosage_forms: ['tablet', 'disintegrating'], typical_strengths: ['0.5mg', '1mg', '2mg'], indications: ['seizures', 'epilepsy', 'panic disorder'] },
  { generic_name: 'diazepam', brand_names: ['Valium'], drug_class: 'Benzodiazepine', dosage_forms: ['tablet', 'liquid', 'injection'], typical_strengths: ['2mg', '5mg', '10mg'], indications: ['anxiety', 'seizures', 'muscle spasms'] },
  { generic_name: 'lorazepam', brand_names: ['Ativan'], drug_class: 'Benzodiazepine', dosage_forms: ['tablet', 'liquid', 'injection'], typical_strengths: ['0.5mg', '1mg', '2mg'], indications: ['anxiety', 'seizures', 'insomnia'] },
  { generic_name: 'midazolam', brand_names: ['Versed'], drug_class: 'Benzodiazepine', dosage_forms: ['liquid', 'injection'], typical_strengths: ['1mg/ml', '5mg/ml'], indications: ['sedation', 'seizures', 'anxiety'] },
  { generic_name: 'phenobarbital', brand_names: ['Phenobarbital'], drug_class: 'Barbiturate', dosage_forms: ['tablet', 'liquid', 'injection'], typical_strengths: ['15mg', '30mg', '60mg', '100mg'], indications: ['seizures', 'insomnia', 'anxiety'] },
  { generic_name: 'primidone', brand_names: ['Mysoline'], drug_class: 'Anticonvulsant', dosage_forms: ['tablet'], typical_strengths: ['50mg', '250mg'], indications: ['seizures', 'epilepsy'] },
  { generic_name: 'acetazolamide', brand_names: ['Diamox'], drug_class: 'Carbonic Anhydrase Inhibitor', dosage_forms: ['tablet', 'capsule', 'injection'], typical_strengths: ['125mg', '250mg'], indications: ['glaucoma', 'seizures', 'altitude sickness'] },
  { generic_name: 'methazolamide', brand_names: ['Neptazane'], drug_class: 'Carbonic Anhydrase Inhibitor', dosage_forms: ['tablet'], typical_strengths: ['25mg', '50mg'], indications: ['glaucoma'] },
  { generic_name: 'dichlorphenamide', brand_names: ['Keveyis'], drug_class: 'Carbonic Anhydrase Inhibitor', dosage_forms: ['tablet'], typical_strengths: ['25mg', '50mg'], indications: ['periodic paralysis'] },
  { generic_name: 'zonisamide', brand_names: ['Zonegran'], drug_class: 'Anticonvulsant', dosage_forms: ['capsule'], typical_strengths: ['25mg', '50mg', '100mg'], indications: ['seizures', 'epilepsy'] },
  { generic_name: 'sulthiame', brand_names: ['Ospolot'], drug_class: 'Anticonvulsant', dosage_forms: ['tablet'], typical_strengths: ['50mg'], indications: ['seizures', 'epilepsy'] },
  { generic_name: 'sultiame', brand_names: ['Ospolot'], drug_class: 'Anticonvulsant', dosage_forms: ['tablet'], typical_strengths: ['50mg'], indications: ['seizures', 'epilepsy'] },
  { generic_name: 'sulthiam', brand_names: ['Ospolot'], drug_class: 'Anticonvulsant', dosage_forms: ['tablet'], typical_strengths: ['50mg'], indications: ['seizures', 'epilepsy'] },
  { generic_name: 'sulthiame', brand_names: ['Ospolot'], drug_class: 'Anticonvulsant', dosage_forms: ['tablet'], typical_strengths: ['50mg'], indications: ['seizures', 'epilepsy'] },
  { generic_name: 'sultiame', brand_names: ['Ospolot'], drug_class: 'Anticonvulsant', dosage_forms: ['tablet'], typical_strengths: ['50mg'], indications: ['seizures', 'epilepsy'] },
  { generic_name: 'sulthiam', brand_names: ['Ospolot'], drug_class: 'Anticonvulsant', dosage_forms: ['tablet'], typical_strengths: ['50mg'], indications: ['seizures', 'epilepsy'] }
];

// Levenshtein distance for fuzzy matching
const levenshteinDistance = (str1, str2) => {
  const matrix = [];
  for (let i = 0; i <= str2.length; i++) {
    matrix[i] = [i];
  }
  for (let j = 0; j <= str1.length; j++) {
    matrix[0][j] = j;
  }
  for (let i = 1; i <= str2.length; i++) {
    for (let j = 1; j <= str1.length; j++) {
      if (str2.charAt(i - 1) === str1.charAt(j - 1)) {
        matrix[i][j] = matrix[i - 1][j - 1];
      } else {
        matrix[i][j] = Math.min(
          matrix[i - 1][j - 1] + 1,
          matrix[i][j - 1] + 1,
          matrix[i - 1][j] + 1
        );
      }
    }
  }
  return 1 - (matrix[str2.length][str1.length] / Math.max(str1.length, str2.length));
};

// Enhanced medication search with multiple strategies
const enhancedMedicationSearch = (medicationName) => {
  const normalizedName = medicationName.toLowerCase().trim();
  
  // Enhanced acronym and nickname mapping
  const acronymMap = {
    'asa': 'aspirin',
    'tylenol': 'acetaminophen',
    'advil': 'ibuprofen',
    'motrin': 'ibuprofen',
    'aleve': 'naproxen',
    'bayer': 'aspirin',
    'ozempic': 'semaglutide',
    'wegovy': 'semaglutide',
    'rybelsus': 'semaglutide',
    'adderall': 'amphetamine/dextroamphetamine',
    'xanax': 'alprazolam',
    'prozac': 'fluoxetine',
    'zoloft': 'sertraline',
    'lexapro': 'escitalopram',
    'wellbutrin': 'bupropion',
    'lipitor': 'atorvastatin',
    'zocor': 'simvastatin',
    'prilosec': 'omeprazole',
    'nexium': 'esomeprazole',
    'pepcid': 'famotidine',
    'zantac': 'ranitidine',
    'glucophage': 'metformin',
    'januvia': 'sitagliptin',
    'actos': 'pioglitazone',
    'glucotrol': 'glipizide',
    'prinivil': 'lisinopril',
    'zestril': 'lisinopril',
    'norvasc': 'amlodipine',
    'lopressor': 'metoprolol',
    'toprol': 'metoprolol',
    'coumadin': 'warfarin',
    'jantoven': 'warfarin',
    'humalog': 'insulin',
    'novolog': 'insulin',
    'lantus': 'insulin',
    'tresiba': 'insulin'
  };
  
  // Check acronym/nickname mapping first
  const mappedName = acronymMap[normalizedName];
  if (mappedName) {
    const mappedMatch = medicationDatabase.find(med => 
      med.generic_name.toLowerCase() === mappedName
    );
    if (mappedMatch) {
      return {
        success: true,
        data: {
          ...mappedMatch,
          confidence: 0.95,
          match_type: 'acronym_mapped',
          original_input: medicationName
        }
      };
    }
  }
  
  // Strategy 1: Exact match
  let exactMatch = medicationDatabase.find(med => 
    med.generic_name.toLowerCase() === normalizedName ||
    med.brand_names.some(brand => brand.toLowerCase() === normalizedName)
  );
  
  if (exactMatch) {
    return {
      success: true,
      data: {
        ...exactMatch,
        confidence: 1.0,
        match_type: 'exact'
      }
    };
  }
  
  // Strategy 2: Partial match (brand names)
  let partialMatch = medicationDatabase.find(med => 
    med.brand_names.some(brand => 
      brand.toLowerCase().includes(normalizedName) ||
      normalizedName.includes(brand.toLowerCase())
    )
  );
  
  if (partialMatch) {
    return {
      success: true,
      data: {
        ...partialMatch,
        confidence: 0.9,
        match_type: 'partial_brand'
      }
    };
  }
  
  // Strategy 3: Enhanced fuzzy matching with capitalization tolerance
  let bestMatch = null;
  let bestScore = 0;
  
  for (const med of medicationDatabase) {
    // Check generic name with multiple case variations
    const genericVariations = [
      med.generic_name.toLowerCase(),
      med.generic_name.toUpperCase(),
      med.generic_name.charAt(0).toUpperCase() + med.generic_name.slice(1).toLowerCase()
    ];
    
    const genericScores = genericVariations.map(variation => 
      levenshteinDistance(normalizedName, variation)
    );
    const bestGenericScore = Math.max(...genericScores);
    
    // Check brand names with multiple case variations
    const brandScores = med.brand_names.flatMap(brand => [
      levenshteinDistance(normalizedName, brand.toLowerCase()),
      levenshteinDistance(normalizedName, brand.toUpperCase()),
      levenshteinDistance(normalizedName, brand.charAt(0).toUpperCase() + brand.slice(1).toLowerCase())
    ]);
    const maxBrandScore = Math.max(...brandScores);
    
    // Enhanced substring matching with case tolerance
    const genericSubstring = genericVariations.some(variation => 
      variation.includes(normalizedName) || normalizedName.includes(variation)
    ) ? 0.95 : 0;
    
    const brandSubstring = med.brand_names.some(brand => {
      const brandLower = brand.toLowerCase();
      return brandLower.includes(normalizedName) || normalizedName.includes(brandLower);
    }) ? 0.95 : 0;
    
    // Check for common misspellings and abbreviations
    const commonMisspellings = {
      'acetaminophen': ['tylenol', 'paracetamol', 'acetaminophen'],
      'ibuprofen': ['advil', 'motrin', 'ibuprofen'],
      'aspirin': ['asa', 'bayer', 'aspirin'],
      'semaglutide': ['ozempic', 'wegovy', 'rybelsus', 'semaglutide'],
      'amphetamine': ['adderall', 'amphetamine', 'dextroamphetamine'],
      'alprazolam': ['xanax', 'alprazolam'],
      'fluoxetine': ['prozac', 'fluoxetine'],
      'sertraline': ['zoloft', 'sertraline']
    };
    
    let misspellingScore = 0;
    for (const [correct, variations] of Object.entries(commonMisspellings)) {
      if (variations.includes(normalizedName) && med.generic_name.toLowerCase() === correct) {
        misspellingScore = 0.9;
        break;
      }
    }
    
    const maxScore = Math.max(bestGenericScore, maxBrandScore, genericSubstring, brandSubstring, misspellingScore);
    if (maxScore > bestScore && maxScore > 0.6) { // Lowered threshold for better matching
      bestScore = maxScore;
      bestMatch = med;
    }
  }
  
  if (bestMatch) {
    return {
      success: true,
      data: {
        ...bestMatch,
        confidence: bestScore,
        match_type: 'fuzzy'
      }
    };
  }
  
  // Strategy 4: Common misspellings and variations
  const commonVariations = {
    'tylenol': 'acetaminophen',
    'advil': 'ibuprofen',
    'motrin': 'ibuprofen',
    'aleve': 'naproxen',
    'aspirin': 'aspirin',
    'panadol': 'acetaminophen',
    'excedrin': 'acetaminophen',
    'benadryl': 'diphenhydramine',
    'claritin': 'loratadine',
    'zantac': 'ranitidine',
    'prilosec': 'omeprazole',
    'nexium': 'esomeprazole',
    'lipitor': 'atorvastatin',
    'zocor': 'simvastatin',
    'crestor': 'rosuvastatin',
    'glucophage': 'metformin',
    'januvia': 'sitagliptin',
    'actos': 'pioglitazone',
    'glipizide': 'glipizide',
    'insulin': 'insulin',
    'lisinopril': 'lisinopril',
    'amlodipine': 'amlodipine',
    'metoprolol': 'metoprolol',
    'atenolol': 'atenolol',
    'propranolol': 'propranolol',
    'warfarin': 'warfarin',
    'coumadin': 'warfarin',
    'plavix': 'clopidogrel',
    'effexor': 'venlafaxine',
    'prozac': 'fluoxetine',
    'zoloft': 'sertraline',
    'lexapro': 'escitalopram',
    'celexa': 'citalopram',
    'wellbutrin': 'bupropion',
    'xanax': 'alprazolam',
    'valium': 'diazepam',
    'ativan': 'lorazepam',
    'klonopin': 'clonazepam',
    'ambien': 'zolpidem',
    'lunesta': 'eszopiclone',
    'trazodone': 'trazodone',
    'mirtazapine': 'mirtazapine',
    'quetiapine': 'quetiapine',
    'olanzapine': 'olanzapine',
    'risperidone': 'risperidone',
    'aripiprazole': 'aripiprazole',
    'ziprasidone': 'ziprasidone',
    'haloperidol': 'haloperidol',
    'chlorpromazine': 'chlorpromazine',
    'thioridazine': 'thioridazine',
    'fluphenazine': 'fluphenazine',
    'perphenazine': 'perphenazine',
    'trifluoperazine': 'trifluoperazine',
    'mesoridazine': 'mesoridazine',
    'prochlorperazine': 'prochlorperazine',
    'promethazine': 'promethazine',
    'hydroxyzine': 'hydroxyzine',
    'diphenhydramine': 'diphenhydramine',
    'doxylamine': 'doxylamine',
    'cyclizine': 'cyclizine',
    'meclizine': 'meclizine',
    'dimenhydrinate': 'dimenhydrinate',
    'scopolamine': 'scopolamine',
    'hyoscyamine': 'hyoscyamine',
    'atropine': 'atropine',
    'glycopyrrolate': 'glycopyrrolate',
    'propantheline': 'propantheline',
    'dicyclomine': 'dicyclomine',
    'belladonna': 'belladonna',
    'phenobarbital': 'phenobarbital',
    'phenytoin': 'phenytoin',
    'carbamazepine': 'carbamazepine',
    'oxcarbazepine': 'oxcarbazepine',
    'valproic acid': 'valproic acid',
    'divalproex': 'divalproex',
    'gabapentin': 'gabapentin',
    'pregabalin': 'pregabalin',
    'topiramate': 'topiramate',
    'lamotrigine': 'lamotrigine',
    'levetiracetam': 'levetiracetam',
    'zonisamide': 'zonisamide',
    'tiagabine': 'tiagabine',
    'vigabatrin': 'vigabatrin',
    'felbamate': 'felbamate',
    'ethosuximide': 'ethosuximide',
    'methsuximide': 'methsuximide',
    'phensuximide': 'phensuximide',
    'trimethadione': 'trimethadione',
    'paramethadione': 'paramethadione',
    'ethotoin': 'ethotoin',
    'mephenytoin': 'mephenytoin',
    'fosphenytoin': 'fosphenytoin',
    'lacosamide': 'lacosamide',
    'rufinamide': 'rufinamide',
    'ezogabine': 'ezogabine',
    'perampanel': 'perampanel',
    'brivaracetam': 'brivaracetam',
    'cenobamate': 'cenobamate',
    'ganaxolone': 'ganaxolone',
    'stiripentol': 'stiripentol',
    'cannabidiol': 'cannabidiol',
    'clobazam': 'clobazam',
    'clonazepam': 'clonazepam',
    'diazepam': 'diazepam',
    'lorazepam': 'lorazepam',
    'midazolam': 'midazolam',
    'primidone': 'primidone',
    'acetazolamide': 'acetazolamide',
    'methazolamide': 'methazolamide',
    'dichlorphenamide': 'dichlorphenamide',
    'sulthiame': 'sulthiame',
    'sultiame': 'sultiame',
    'sulthiam': 'sulthiam'
  };
  
  const variation = commonVariations[normalizedName];
  if (variation) {
    const variationMatch = medicationDatabase.find(med => 
      med.generic_name.toLowerCase() === variation.toLowerCase()
    );
    
    if (variationMatch) {
      return {
        success: true,
        data: {
          ...variationMatch,
          confidence: 0.85,
          match_type: 'variation',
          original_input: medicationName,
          suggested_name: variationMatch.generic_name
        }
      };
    }
  }
  
  return {
    success: false,
    error: 'No medication found'
  };
};

// Test endpoint
router.get('/test', (req, res) => {
  res.json({ message: 'Test endpoint working' });
});

// Enhanced medication validation endpoint
router.post('/validateMedication', async (req, res) => {
  try {
    const { medication_name } = req.body;
    
    if (!medication_name) {
      return res.status(400).json({
        success: false,
        error: 'Medication name is required'
      });
    }

    // Use enhanced search with comprehensive database
    const searchResult = enhancedMedicationSearch(medication_name);
    
    if (searchResult.success) {
      const medication = searchResult.data;
      
      // Get suggested metrics based on drug class and indications
      const suggestedMetrics = getSuggestedMetrics(medication);
      
      return res.json({
        success: true,
        data: {
          generic_name: medication.generic_name,
          brand_names: medication.brand_names,
          drug_class: medication.drug_class,
          dosage_forms: medication.dosage_forms,
          typical_strengths: medication.typical_strengths,
          indications: medication.indications,
          confidence: medication.confidence,
          match_type: medication.match_type,
          suggested_metrics: suggestedMetrics,
          medication_name: medication.generic_name, // For compatibility
          name: medication.generic_name,
          brandName: medication.brand_names[0] || medication.generic_name
        }
      });
    } else {
      return res.json({
        success: false,
        error: searchResult.error || 'Medication not found',
        suggestions: getSimilarMedications(medication_name)
      });
    }
  } catch (error) {
    console.error('Error validating medication:', error);
    return res.status(500).json({
      success: false,
      error: 'Internal server error'
    });
  }
});

// Get suggested metrics for medication
const getSuggestedMetrics = (medication) => {
  const metricSuggestions = {
    'acetaminophen': ['Pain Level', 'Sleep Quality', 'Temperature'],
    'ibuprofen': ['Pain Level', 'Inflammation Markers', 'Digestive Health'],
    'naproxen': ['Pain Level', 'Inflammation Markers', 'Digestive Health'],
    'aspirin': ['Pain Level', 'Blood Pressure', 'Heart Rate'],
    'metformin': ['Blood Glucose', 'Weight', 'Digestive Health'],
    'insulin': ['Blood Glucose', 'Weight', 'Hypoglycemia Symptoms'],
    'lisinopril': ['Blood Pressure', 'Kidney Function', 'Heart Rate'],
    'atorvastatin': ['Cholesterol', 'Liver Function', 'Muscle Health'],
    'simvastatin': ['Cholesterol', 'Liver Function', 'Muscle Health'],
    'amlodipine': ['Blood Pressure', 'Heart Rate', 'Ankle Swelling'],
    'metoprolol': ['Blood Pressure', 'Heart Rate', 'Energy Level'],
    'warfarin': ['Blood Count', 'Bleeding Risk', 'Blood Pressure'],
    'omeprazole': ['Digestive Health', 'Stomach Pain', 'Acid Reflux'],
    'esomeprazole': ['Digestive Health', 'Stomach Pain', 'Acid Reflux'],
    'sertraline': ['Mood', 'Anxiety Level', 'Sleep Quality'],
    'fluoxetine': ['Mood', 'Anxiety Level', 'Sleep Quality'],
    'escitalopram': ['Mood', 'Anxiety Level', 'Sleep Quality'],
    'venlafaxine': ['Mood', 'Anxiety Level', 'Blood Pressure'],
    'bupropion': ['Mood', 'Energy Level', 'Sleep Quality'],
    'alprazolam': ['Anxiety Level', 'Sleep Quality', 'Mood'],
    'lorazepam': ['Anxiety Level', 'Sleep Quality', 'Mood'],
    'zolpidem': ['Sleep Quality', 'Mood', 'Energy Level'],
    'trazodone': ['Sleep Quality', 'Mood', 'Blood Pressure'],
    'melatonin': ['Sleep Quality', 'Mood', 'Energy Level'],
    'vitamin d': ['Bone Health', 'Immune Function', 'Mood'],
    'vitamin b12': ['Energy Level', 'Mood', 'Nerve Health'],
    'iron': ['Energy Level', 'Blood Count', 'Mood'],
    'calcium': ['Bone Health', 'Muscle Health', 'Heart Rate']
  };

  const genericName = medication.generic_name?.toLowerCase() || '';
  const drugClass = medication.drug_class?.toLowerCase() || '';
  
  // Check for exact matches
  for (const [key, metrics] of Object.entries(metricSuggestions)) {
    if (genericName.includes(key)) {
      return metrics;
    }
  }

  // Check for drug class matches
  if (drugClass.includes('analgesic') || drugClass.includes('pain')) {
    return ['Pain Level', 'Sleep Quality', 'Mood'];
  } else if (drugClass.includes('diabetes') || drugClass.includes('glucose')) {
    return ['Blood Glucose', 'Weight', 'Energy Level'];
  } else if (drugClass.includes('cardiovascular') || drugClass.includes('heart')) {
    return ['Blood Pressure', 'Heart Rate', 'Weight'];
  } else if (drugClass.includes('mental') || drugClass.includes('psychiatric')) {
    return ['Mood', 'Anxiety Level', 'Sleep Quality'];
  } else if (drugClass.includes('statin') || drugClass.includes('cholesterol')) {
    return ['Cholesterol', 'Liver Function', 'Muscle Health'];
  } else if (drugClass.includes('acid') || drugClass.includes('reflux')) {
    return ['Digestive Health', 'Stomach Pain', 'Acid Reflux'];
  }

  return ['General Health', 'Side Effects', 'Medication Adherence'];
};

// Get similar medications for suggestions
const getSimilarMedications = (medicationName) => {
  const normalizedName = medicationName.toLowerCase().trim();
  const suggestions = [];
  
  for (const med of medicationDatabase) {
    const genericScore = levenshteinDistance(normalizedName, med.generic_name.toLowerCase());
    const brandScores = med.brand_names.map(brand => 
      levenshteinDistance(normalizedName, brand.toLowerCase())
    );
    const maxBrandScore = Math.max(...brandScores);
    const maxScore = Math.max(genericScore, maxBrandScore);
    
    if (maxScore > 0.5 && maxScore < 0.8) {
      suggestions.push({
        name: med.generic_name,
        brand: med.brand_names[0],
        confidence: maxScore
      });
    }
  }
  
  return suggestions
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 5);
};

module.exports = router;