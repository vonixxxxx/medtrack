import { motion } from "framer-motion";
import Card from "./card";
import FadeIn from "./FadeIn";

/**
 * Reusable feature grid with consistent layout
 */
export default function FeatureGrid({ 
  features, 
  columns = { mobile: 1, tablet: 2, desktop: 3 },
  className = ""
}) {
  const gridClasses = `grid grid-cols-${columns.mobile} sm:grid-cols-${columns.tablet} lg:grid-cols-${columns.desktop} gap-6 lg:gap-8`;

  return (
    <div className={`${gridClasses} ${className}`}>
      {features.map((feature, index) => (
        <FadeIn key={feature.id || index} delay={index * 0.1}>
          <Card hover={!!feature.onClick || !!feature.href} href={feature.href} onClick={feature.onClick}>
            {feature.icon && (
              <div className={`inline-flex p-3 rounded-lg ${feature.iconBg || "bg-blue-100"} mb-4`}>
                <feature.icon 
                  className={feature.iconColor ? `text-${feature.iconColor}` : "text-blue-600"} 
                  size={24} 
                />
              </div>
            )}
            {feature.title && (
              <h3 className="font-bold text-xl mb-2 text-gray-900">{feature.title}</h3>
            )}
            {feature.description && (
              <p className="text-gray-600 text-sm leading-relaxed">{feature.description}</p>
            )}
            {feature.children}
          </Card>
        </FadeIn>
      ))}
    </div>
  );
}

