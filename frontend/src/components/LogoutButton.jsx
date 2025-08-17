export default function LogoutButton() {
  return (
    <button
      onClick={() => {
        localStorage.clear();
        window.location.href = '/login';
      }}
      className="fixed top-4 right-4 text-transparent bg-clip-text bg-gradient-to-r from-blue-500 to-indigo-600 font-medium hover:from-blue-600 hover:to-indigo-700 transition-all duration-300 text-sm"
      title="Logout"
    >
      Logout
    </button>
  );
}
