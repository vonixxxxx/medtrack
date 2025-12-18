import { motion } from "framer-motion";

/**
 * Reusable section wrapper with consistent spacing and transitions
 */
export default function Section({ 
  children, 
  className = "", 
  id,
  background = "white",
  padding = "py-16 lg:py-24",
  divider = false,
  dividerText = null
}) {
  const bgClasses = {
    white: "bg-white",
    gray: "bg-gray-50",
    "gray-to-white": "bg-gradient-to-b from-gray-50 to-white",
    "white-to-gray": "bg-gradient-to-b from-white to-gray-50",
    blue: "bg-blue-50/30",
  };

  return (
    <>
      {divider && (
        <div className="relative py-8">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-gray-200"></div>
          </div>
          {dividerText && (
            <div className="relative flex justify-center">
              <span className={`${bgClasses[background]} px-4 text-sm text-gray-500 font-medium`}>
                {dividerText}
              </span>
            </div>
          )}
        </div>
      )}
      <section 
        id={id} 
        className={`${bgClasses[background]} ${padding} ${className}`}
      >
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          {children}
        </div>
      </section>
    </>
  );
}



