import { useState, useEffect } from 'react';

const DebugInfo = () => {
  const [debugInfo, setDebugInfo] = useState({});

  useEffect(() => {
    const token = localStorage.getItem('token');
    const userString = localStorage.getItem('user');
    const user = userString && userString !== 'undefined' ? JSON.parse(userString) : {};
    
    setDebugInfo({
      token: token ? 'Present' : 'Missing',
      userString,
      user,
      hasRole: !!user.role,
      role: user.role
    });
  }, []);

  return (
    <div style={{ 
      position: 'fixed', 
      top: 0, 
      left: 0, 
      background: 'black', 
      color: 'white', 
      padding: '10px', 
      zIndex: 9999,
      fontSize: '12px',
      maxWidth: '300px'
    }}>
      <h3>Debug Info:</h3>
      <pre>{JSON.stringify(debugInfo, null, 2)}</pre>
    </div>
  );
};

export default DebugInfo;
