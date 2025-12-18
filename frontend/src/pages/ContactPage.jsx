import React, { useState, useRef } from 'react';
import { motion, useInView } from 'framer-motion';
import LandingHeader from '../components/landing/LandingHeader';
import LandingFooter from '../components/landing/LandingFooter';
import {
  Mail, Send, MessageSquare, Clock,
  CheckCircle2, AlertCircle, ArrowRight, Shield, Users, FileText
} from 'lucide-react';

// Medical-grade color system
const colors = {
  primary: {
    600: '#0284c7', // Primary blue
  },
  teal: {
    600: '#0d9488', // Secondary teal
  },
  gray: {
    50: '#f9fafb',
    100: '#f3f4f6',
    200: '#e5e7eb',
    300: '#d1d5db',
    400: '#9ca3af',
    500: '#6b7280',
    600: '#4b5563',
    700: '#374151',
    800: '#1f2937',
    900: '#111827',
  }
};

const contactMethods = [
  {
    icon: Mail,
    title: 'Email Support',
    description: 'For general inquiries and support',
    value: 'alex@medtrack.uk',
    action: 'mailto:alex@medtrack.uk',
    availability: '24/7 response within 24 hours'
  }
];

const faqs = [
  {
    question: 'How does medication validation work?',
    answer: 'MedTrack validates medications against the FDA drug-label database containing 3+ million records. Our system uses fuzzy matching algorithms, RxNorm API integration, and AI-powered parsing to identify drug classes, recommend dosages, and verify safety. Validation occurs in real-time with 99.9% accuracy.'
  },
  {
    question: 'Is my health data secure and HIPAA compliant?',
    answer: 'Yes. MedTrack is fully HIPAA and GDPR compliant. We use end-to-end encryption, role-based access control, and comprehensive audit logging. Patient data is encrypted at rest and in transit. We never share data with third parties without explicit consent, and all data processing follows strict healthcare privacy regulations.'
  },
  {
    question: 'Can healthcare providers and institutions use MedTrack?',
    answer: 'Absolutely. MedTrack offers comprehensive clinical dashboards designed for healthcare providers, including patient management, real-time analytics, medication adherence tracking, and reporting tools. We support EMR integration and offer enterprise deployment options for hospitals and healthcare networks.'
  },
  {
    question: 'How accurate is the pill recognition feature?',
    answer: 'Our pill recognition uses advanced CNN (Convolutional Neural Network) models trained on thousands of medication images. The system achieves 94%+ accuracy in identifying medications from photographs. All identifications include confidence scores and can be manually verified for critical medications.'
  },
  {
    question: 'What makes MedTrack different from other medication tracking apps?',
    answer: 'MedTrack is built on clinical-grade technology: FDA database integration (3M+ records), real-time drug interaction checking, AI-powered health analytics, and enterprise security. We prioritize clinical accuracy, evidence-based recommendations, and healthcare provider workflows over consumer-focused features.'
  },
  {
    question: 'Do you offer API access for integration?',
    answer: 'Yes. MedTrack provides RESTful APIs for healthcare institutions and developers. Our APIs support medication validation, health metrics tracking, and clinical data integration. Enterprise customers can access comprehensive API documentation and dedicated support for custom integrations.'
  }
];

const ContactMethodCard = ({ method, index }) => {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-50px' });
  const Icon = method.icon;

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 30 }}
      animate={isInView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.6, delay: index * 0.1, ease: [0.22, 1, 0.36, 1] }}
      className="bg-white rounded-2xl p-8 border border-gray-200 hover:border-blue-300 transition-all shadow-sm hover:shadow-lg"
    >
      <div className="flex items-start gap-4 mb-6">
        <div className="p-3 bg-blue-50 rounded-xl">
          <Icon className="text-blue-600" size={24} />
        </div>
        <div className="flex-1">
          <h3 className="text-xl font-bold text-gray-900 mb-2">{method.title}</h3>
          <p className="text-gray-600 text-sm mb-4">{method.description}</p>
          {method.action ? (
            <a
              href={method.action}
              className="text-blue-600 font-semibold hover:text-blue-700 transition-colors block mb-2"
            >
              {method.value}
            </a>
          ) : (
            <p className="text-gray-900 font-medium mb-2">{method.value}</p>
          )}
          <p className="text-xs text-gray-500">{method.availability}</p>
        </div>
      </div>
    </motion.div>
  );
};

const FAQCard = ({ faq, index }) => {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-50px' });
  const [isOpen, setIsOpen] = useState(false);

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 20 }}
      animate={isInView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.6, delay: index * 0.1, ease: [0.22, 1, 0.36, 1] }}
      className="bg-white rounded-2xl border border-gray-200 overflow-hidden"
    >
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full p-6 text-left flex items-start justify-between gap-4 hover:bg-gray-50 transition-colors"
      >
        <h3 className="text-lg font-semibold text-gray-900 flex-1">{faq.question}</h3>
        <div className={`flex-shrink-0 transition-transform ${isOpen ? 'rotate-180' : ''}`}>
          <ArrowRight className="text-gray-400" size={20} />
        </div>
      </button>
      <motion.div
        initial={false}
        animate={{ height: isOpen ? 'auto' : 0 }}
        transition={{ duration: 0.3 }}
        className="overflow-hidden"
      >
        <div className="px-6 pb-6 text-gray-600 leading-relaxed">
          {faq.answer}
        </div>
      </motion.div>
    </motion.div>
  );
};

export default function ContactPage() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    organization: '',
    subject: '',
    message: ''
  });
  const [status, setStatus] = useState({ type: null, message: '' });

  const handleSubmit = async (e) => {
    e.preventDefault();
    setStatus({ type: 'loading', message: 'Sending message...' });
    
    // Simulate form submission
    setTimeout(() => {
      setStatus({ 
        type: 'success', 
        message: 'Thank you for your message. We\'ll respond within 24 hours during business days.' 
      });
      setFormData({ name: '', email: '', organization: '', subject: '', message: '' });
    }, 1500);
  };

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  return (
    <div className="min-h-screen bg-white">
      <LandingHeader />
      
      {/* Hero Section */}
      <section className="relative pt-32 pb-24 lg:pt-40 lg:pb-32 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-blue-50 via-white to-teal-50" />
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
            className="text-center max-w-4xl mx-auto"
          >
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-blue-50 rounded-full mb-8">
              <MessageSquare className="text-blue-600" size={16} />
              <span className="text-sm font-medium text-blue-900">Contact Us</span>
            </div>
            
            <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold text-gray-900 mb-6 leading-tight tracking-tight">
              Get in Touch
            </h1>
            
            <p className="text-xl lg:text-2xl text-gray-600 mb-12 leading-relaxed max-w-3xl mx-auto">
              Have questions about MedTrack? We're here to help. Whether you're a healthcare 
              provider, researcher, or patient, our team is ready to assist you.
            </p>

            <div className="flex flex-wrap justify-center gap-6 text-sm">
              <div className="flex items-center gap-2 text-gray-600">
                <CheckCircle2 className="text-green-600" size={18} />
                <span>24/7 Support</span>
              </div>
              <div className="flex items-center gap-2 text-gray-600">
                <CheckCircle2 className="text-green-600" size={18} />
                <span>Enterprise Solutions</span>
              </div>
              <div className="flex items-center gap-2 text-gray-600">
                <CheckCircle2 className="text-green-600" size={18} />
                <span>Clinical Integration</span>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Contact Methods */}
      <section className="py-24 lg:py-32 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl lg:text-5xl font-bold text-gray-900 mb-4">
              Contact Methods
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Choose the method that works best for you
            </p>
          </motion.div>

          <div className="grid md:grid-cols-1 max-w-2xl mx-auto gap-8 mb-24">
            {contactMethods.map((method, index) => (
              <ContactMethodCard key={method.title} method={method} index={index} />
            ))}
          </div>

          {/* Contact Form */}
          <div className="max-w-3xl mx-auto">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
              className="bg-white rounded-2xl p-8 lg:p-12 border border-gray-200 shadow-lg"
            >
              <div className="mb-8">
                <div className="flex items-center gap-3 mb-2">
                  <MessageSquare className="text-blue-600" size={24} />
                  <h2 className="text-3xl font-bold text-gray-900">Send a Message</h2>
                </div>
                <p className="text-gray-600">
                  Fill out the form below and we'll get back to you as soon as possible.
                </p>
              </div>

              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <label htmlFor="name" className="block text-sm font-semibold text-gray-900 mb-2">
                      Name *
                    </label>
                    <input
                      type="text"
                      id="name"
                      name="name"
                      required
                      value={formData.name}
                      onChange={handleChange}
                      className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all bg-white"
                      placeholder="Your full name"
                    />
                  </div>
                  <div>
                    <label htmlFor="email" className="block text-sm font-semibold text-gray-900 mb-2">
                      Email *
                    </label>
                    <input
                      type="email"
                      id="email"
                      name="email"
                      required
                      value={formData.email}
                      onChange={handleChange}
                      className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all bg-white"
                      placeholder="your@email.com"
                    />
                  </div>
                </div>

                <div>
                  <label htmlFor="organization" className="block text-sm font-semibold text-gray-900 mb-2">
                    Organization
                  </label>
                  <input
                    type="text"
                    id="organization"
                    name="organization"
                    value={formData.organization}
                    onChange={handleChange}
                    className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all bg-white"
                    placeholder="Your organization (optional)"
                  />
                </div>

                <div>
                  <label htmlFor="subject" className="block text-sm font-semibold text-gray-900 mb-2">
                    Subject *
                  </label>
                  <input
                    type="text"
                    id="subject"
                    name="subject"
                    required
                    value={formData.subject}
                    onChange={handleChange}
                    className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all bg-white"
                    placeholder="What is this regarding?"
                  />
                </div>

                <div>
                  <label htmlFor="message" className="block text-sm font-semibold text-gray-900 mb-2">
                    Message *
                  </label>
                  <textarea
                    id="message"
                    name="message"
                    required
                    rows={6}
                    value={formData.message}
                    onChange={handleChange}
                    className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all resize-none bg-white"
                    placeholder="Tell us how we can help..."
                  />
                </div>

                {status.message && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={`p-4 rounded-xl flex items-start gap-3 ${
                      status.type === 'success'
                        ? 'bg-green-50 text-green-900 border border-green-200'
                        : status.type === 'error'
                        ? 'bg-red-50 text-red-900 border border-red-200'
                        : 'bg-blue-50 text-blue-900 border border-blue-200'
                    }`}
                  >
                    {status.type === 'success' ? (
                      <CheckCircle2 size={20} className="flex-shrink-0 mt-0.5" />
                    ) : status.type === 'error' ? (
                      <AlertCircle size={20} className="flex-shrink-0 mt-0.5" />
                    ) : (
                      <Clock size={20} className="flex-shrink-0 mt-0.5" />
                    )}
                    <span className="text-sm font-medium">{status.message}</span>
                  </motion.div>
                )}

                <motion.button
                  type="submit"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  disabled={status.type === 'loading'}
                  className="w-full px-6 py-4 bg-blue-600 text-white rounded-xl font-semibold hover:bg-blue-700 transition-colors flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-blue-600/25"
                >
                  <Send size={20} />
                  {status.type === 'loading' ? 'Sending...' : 'Send Message'}
                </motion.button>
              </form>
            </motion.div>
          </div>
        </div>
      </section>

      {/* FAQ Section */}
      <section className="py-24 lg:py-32 bg-gray-50">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl lg:text-5xl font-bold text-gray-900 mb-4">
              Frequently Asked Questions
            </h2>
            <p className="text-xl text-gray-600">
              Common questions about MedTrack's platform and capabilities
            </p>
          </motion.div>

          <div className="space-y-4">
            {faqs.map((faq, index) => (
              <FAQCard key={index} faq={faq} index={index} />
            ))}
          </div>
        </div>
      </section>

      {/* Additional Info */}
      <section className="py-24 lg:py-32 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                icon: Shield,
                title: 'Enterprise Security',
                description: 'HIPAA and GDPR compliant with enterprise-grade security architecture.'
              },
              {
                icon: Users,
                title: 'Clinical Integration',
                description: 'Seamless integration with EMR systems and clinical workflows.'
              },
              {
                icon: FileText,
                title: 'API Access',
                description: 'Comprehensive RESTful APIs for custom integrations and development.'
              }
            ].map((item, index) => {
              const Icon = item.icon;
              return (
                <motion.div
                  key={item.title}
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.1, duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
                  className="text-center p-6"
                >
                  <div className="inline-flex p-4 bg-blue-50 rounded-xl mb-4">
                    <Icon className="text-blue-600" size={24} />
                  </div>
                  <h3 className="text-lg font-bold text-gray-900 mb-2">{item.title}</h3>
                  <p className="text-gray-600 text-sm">{item.description}</p>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      <LandingFooter />
    </div>
  );
}
