import { motion } from "framer-motion";
import { Building2, Award, Users } from "lucide-react";

const partners = [
  {
    name: "Imperial College London",
    department: "Research Department in Metabolic Medicine",
    description: "Collaborating on cutting-edge research and data analysis",
  },
  {
    name: "Imperial College Healthcare",
    department: "NHS Trust",
    description: "Potential adoption across healthcare facilities",
  },
];

export default function LandingCollaboration() {
  return (
    <section className="py-24 lg:py-32 bg-gradient-to-br from-gray-50 to-blue-50/30">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-gray-900 mb-4">
            Trusted by Leading Institutions
          </h2>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Partnering with world-class healthcare organizations to advance medical research and patient care
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
          {partners.map((partner, index) => (
            <motion.div
              key={partner.name}
              initial={{ opacity: 0, y: 50 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: index * 0.2 }}
              whileHover={{ y: -8 }}
              className="group p-8 bg-white rounded-3xl border border-gray-200 hover:border-blue-300 transition-all shadow-sm hover:shadow-xl"
            >
              <div className="flex items-start gap-4 mb-4">
                <div className="p-3 bg-blue-100 rounded-xl group-hover:bg-blue-200 transition-colors">
                  <Building2 className="text-blue-600" size={24} />
                </div>
                <div>
                  <h3 className="text-2xl font-bold text-gray-900 mb-1">
                    {partner.name}
                  </h3>
                  <p className="text-blue-600 font-medium">
                    {partner.department}
                  </p>
                </div>
              </div>
              <p className="text-gray-600 leading-relaxed">
                {partner.description}
              </p>
            </motion.div>
          ))}
        </div>

        {/* Stats */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="mt-16 grid grid-cols-3 gap-8 max-w-3xl mx-auto text-center"
        >
          {[
            { icon: Users, value: "10K+", label: "Active Users" },
            { icon: Building2, value: "50+", label: "Institutions" },
            { icon: Award, value: "99.9%", label: "Uptime" },
          ].map((stat, index) => {
            const Icon = stat.icon;
            return (
              <div key={stat.label} className="p-6">
                <Icon className="text-blue-600 mx-auto mb-3" size={32} />
                <div className="text-4xl font-bold text-gray-900 mb-2">
                  {stat.value}
                </div>
                <div className="text-gray-600">{stat.label}</div>
              </div>
            );
          })}
        </motion.div>
      </div>
    </section>
  );
}


