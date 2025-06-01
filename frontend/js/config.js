// frontend/js/config.js
// This file defines global configuration variables for the frontend.
// In a production environment, when the frontend is served by Nginx (or another web server)
// on the same domain as the API (with API requests proxied, e.g., under /api/v1/),
// BACKEND_API_BASE_URL should be an empty string. This makes API calls use relative paths.
// Example: fetch('/api/v1/users/me') instead of fetch('http://domain.com/api/v1/users/me')
window.BACKEND_API_BASE_URL = "";

// For local development, or if the backend is on a completely different domain/port,
// you might temporarily change this to the specific URL, e.g.:
// window.BACKEND_API_BASE_URL = "http://127.0.0.1:8000";
// However, the primary configuration pushed to production should be "".
