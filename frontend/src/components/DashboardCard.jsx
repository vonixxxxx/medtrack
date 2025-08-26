import React from 'react';

const DashboardCard = ({ children, className = '', ...props }) => {
  return (
    <div 
      className={`bg-white border border-gray-200 rounded-2xl transition-all duration-300 hover:border-blue-300 ${className}`}
      {...props}
    >
      {children}
    </div>
  );
};

export default DashboardCard;
