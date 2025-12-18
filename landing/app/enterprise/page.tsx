"use client";

import { motion } from "framer-motion";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import { Shield, Lock, Server, Users, CheckCircle2, ArrowRight } from "lucide-react";
import Link from "next/link";

export default function EnterprisePage() {
  return (
    <main className="min-h-screen">
      <Header />
      <section className="pt-32 pb-16 bg-gradient-to-b from-white to-gray-50">
        <div className="container mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16 max-w-3xl mx-auto"
          >
            <h1 className="text-4xl lg:text-5xl font-bold mb-6">Enterprise Solutions</h1>
            <p className="text-xl text-gray-600 leading-relaxed">
              Security, compliance, and deployment options designed for healthcare institutions
            </p>
          </motion.div>

          {/* Security & Compliance */}
          <motion.div
            id="security"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="mb-24"
          >
            <h2 className="text-3xl font-bold mb-8 text-center">Security & Compliance</h2>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {[
                {
                  icon: Shield,
                  title: "HIPAA Compliance",
                  description: "Full compliance with Health Insurance Portability and Accountability Act. Regular audits and certifications.",
                },
                {
                  icon: Lock,
                  title: "GDPR Ready",
                  description: "Meets all General Data Protection Regulation requirements for EU data protection and patient rights.",
                },
                {
                  icon: Server,
                  title: "End-to-End Encryption",
                  description: "Bank-level encryption for data at rest and in transit. AES-256 encryption standard.",
                },
                {
                  icon: Users,
                  title: "Role-Based Access Control",
                  description: "Granular permissions and access controls with audit trails for all data access.",
                },
                {
                  icon: CheckCircle2,
                  title: "SOC 2 Type II",
                  description: "Undergoing SOC 2 Type II certification for security, availability, and confidentiality.",
                },
                {
                  icon: Lock,
                  title: "Data Residency Options",
                  description: "Choose your data storage location with on-premise and private cloud deployment options.",
                },
              ].map((item, index) => {
                const Icon = item.icon;
                return (
                  <div key={index} className="p-6 bg-white rounded-xl border border-gray-200 shadow-sm hover:shadow-md transition-shadow">
                    <Icon className="w-8 h-8 text-primary-600 mb-4" />
                    <h3 className="text-xl font-semibold mb-3">{item.title}</h3>
                    <p className="text-gray-600">{item.description}</p>
                  </div>
                );
              })}
            </div>
          </motion.div>

          {/* SSO & SAML */}
          <motion.div
            id="sso"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="mb-24"
          >
            <h2 className="text-3xl font-bold mb-8 text-center">Single Sign-On (SSO) & SAML</h2>
            <div className="bg-white rounded-xl border border-gray-200 p-8 lg:p-12 shadow-sm">
              <div className="grid md:grid-cols-2 gap-8">
                <div>
                  <h3 className="text-2xl font-semibold mb-4">Enterprise Authentication</h3>
                  <ul className="space-y-3">
                    <li className="flex items-start gap-3">
                      <CheckCircle2 className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
                      <span className="text-gray-700">SAML 2.0 support for identity providers</span>
                    </li>
                    <li className="flex items-start gap-3">
                      <CheckCircle2 className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
                      <span className="text-gray-700">OAuth 2.0 and OpenID Connect</span>
                    </li>
                    <li className="flex items-start gap-3">
                      <CheckCircle2 className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
                      <span className="text-gray-700">Active Directory integration</span>
                    </li>
                    <li className="flex items-start gap-3">
                      <CheckCircle2 className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
                      <span className="text-gray-700">Multi-factor authentication (MFA)</span>
                    </li>
                  </ul>
                </div>
                <div>
                  <h3 className="text-2xl font-semibold mb-4">Supported Providers</h3>
                  <ul className="space-y-3">
                    <li className="flex items-start gap-3">
                      <CheckCircle2 className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
                      <span className="text-gray-700">Okta, Azure AD, Google Workspace</span>
                    </li>
                    <li className="flex items-start gap-3">
                      <CheckCircle2 className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
                      <span className="text-gray-700">OneLogin, Auth0, Ping Identity</span>
                    </li>
                    <li className="flex items-start gap-3">
                      <CheckCircle2 className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
                      <span className="text-gray-700">Custom SAML providers</span>
                    </li>
                  </ul>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Deployment Options */}
          <motion.div
            id="deployment"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="mb-24"
          >
            <h2 className="text-3xl font-bold mb-8 text-center">Deployment Options</h2>
            <div className="grid md:grid-cols-3 gap-6">
              {[
                {
                  title: "Cloud (SaaS)",
                  description: "Fully managed cloud deployment with automatic updates and scaling.",
                  features: ["99.9% uptime SLA", "Automatic backups", "Global CDN", "24/7 monitoring"],
                },
                {
                  title: "Private Cloud",
                  description: "Dedicated cloud infrastructure in your preferred region or provider.",
                  features: ["Data residency control", "Custom networking", "Dedicated resources", "Hybrid options"],
                },
                {
                  title: "On-Premise",
                  description: "Deploy in your own data center with full control over infrastructure.",
                  features: ["Complete data control", "Air-gapped deployment", "Custom integrations", "Full audit access"],
                },
              ].map((option, index) => (
                <div key={index} className="p-6 bg-white rounded-xl border border-gray-200 shadow-sm">
                  <h3 className="text-2xl font-semibold mb-3">{option.title}</h3>
                  <p className="text-gray-600 mb-4">{option.description}</p>
                  <ul className="space-y-2">
                    {option.features.map((feature, i) => (
                      <li key={i} className="flex items-start gap-2">
                        <CheckCircle2 className="w-4 h-4 text-primary-600 flex-shrink-0 mt-1" />
                        <span className="text-sm text-gray-700">{feature}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </motion.div>

          {/* SLA & Pricing */}
          <motion.div
            id="pricing"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="mb-24"
          >
            <h2 className="text-3xl font-bold mb-8 text-center">SLA & Pricing Tiers</h2>
            <div className="grid md:grid-cols-3 gap-6">
              {[
                {
                  tier: "Professional",
                  sla: "99.5%",
                  price: "Custom",
                  features: ["Up to 1,000 users", "Standard support", "Cloud deployment", "Basic integrations"],
                },
                {
                  tier: "Enterprise",
                  sla: "99.9%",
                  price: "Custom",
                  features: ["Unlimited users", "Priority support", "Private cloud option", "Advanced integrations", "SSO/SAML"],
                },
                {
                  tier: "Enterprise Plus",
                  sla: "99.99%",
                  price: "Custom",
                  features: [
                    "Unlimited users",
                    "24/7 dedicated support",
                    "On-premise option",
                    "Custom integrations",
                    "Dedicated account manager",
                    "SLA guarantees",
                  ],
                },
              ].map((tier, index) => (
                <div
                  key={index}
                  className={`p-6 rounded-xl border-2 shadow-sm ${
                    index === 1
                      ? "bg-gradient-to-br from-primary-50 to-blue-50 border-primary-300"
                      : "bg-white border-gray-200"
                  }`}
                >
                  <h3 className="text-2xl font-semibold mb-2">{tier.tier}</h3>
                  <div className="mb-4">
                    <span className="text-3xl font-bold">{tier.price}</span>
                    <span className="text-gray-600 ml-2">/month</span>
                  </div>
                  <div className="mb-4">
                    <span className="text-sm text-gray-600">Uptime SLA: </span>
                    <span className="font-semibold">{tier.sla}</span>
                  </div>
                  <ul className="space-y-2 mb-6">
                    {tier.features.map((feature, i) => (
                      <li key={i} className="flex items-start gap-2">
                        <CheckCircle2 className="w-4 h-4 text-primary-600 flex-shrink-0 mt-1" />
                        <span className="text-sm text-gray-700">{feature}</span>
                      </li>
                    ))}
                  </ul>
                  <Link
                    href="/contact"
                    className="block w-full text-center px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors font-semibold"
                  >
                    Contact Sales <ArrowRight className="w-4 h-4 inline ml-2" />
                  </Link>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Privacy Section */}
          <motion.div
            id="privacy"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="mb-24"
          >
            <div className="bg-gradient-to-br from-primary-50 to-blue-50 rounded-xl p-8 lg:p-12 border border-primary-200">
              <div className="flex items-center gap-4 mb-6">
                <Lock className="w-8 h-8 text-primary-600" />
                <h2 className="text-3xl font-bold">Privacy-First Architecture</h2>
              </div>
              <p className="text-lg text-gray-700 mb-6">
                MedTrack is built with privacy at its core. We offer local AI options using LLaMA for on-premise 
                inference, ensuring your sensitive health data never leaves your infrastructure. All data processing 
                follows the principle of data minimization, and we provide comprehensive anonymization tools for 
                research purposes.
              </p>
              <ul className="space-y-3">
                <li className="flex items-start gap-3">
                  <CheckCircle2 className="w-5 h-5 text-primary-600 flex-shrink-0 mt-0.5" />
                  <span className="text-gray-700">Local AI inference option (LLaMA) for complete data residency</span>
                </li>
                <li className="flex items-start gap-3">
                  <CheckCircle2 className="w-5 h-5 text-primary-600 flex-shrink-0 mt-0.5" />
                  <span className="text-gray-700">End-to-end encryption with zero-knowledge architecture</span>
                </li>
                <li className="flex items-start gap-3">
                  <CheckCircle2 className="w-5 h-5 text-primary-600 flex-shrink-0 mt-0.5" />
                  <span className="text-gray-700">Comprehensive audit logs and compliance reporting</span>
                </li>
              </ul>
            </div>
          </motion.div>
        </div>
      </section>
      <Footer />
    </main>
  );
}





