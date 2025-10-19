// Utility to clear corrupted localStorage data
export const clearCorruptedStorage = () => {
  const userString = localStorage.getItem('user');
  if (userString === 'undefined' || userString === 'null') {
    localStorage.removeItem('user');
    console.log('Cleared corrupted user data from localStorage');
  }
  
  const token = localStorage.getItem('token');
  if (!token || token === 'undefined' || token === 'null') {
    localStorage.removeItem('token');
    console.log('Cleared corrupted token from localStorage');
  }
};

// Call this function on app startup
clearCorruptedStorage();
