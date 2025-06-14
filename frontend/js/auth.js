// frontend/js/auth.js
console.log("auth.js loaded");

// const BACKEND_API_BASE_URL = 'http://127.0.0.1:8000'; // This will now be set globally via HTML script tag

document.addEventListener('DOMContentLoaded', () => {
    const registrationForm = document.getElementById('registrationForm');
    const loginForm = document.getElementById('loginForm'); // For regular users
    const adminLoginForm = document.getElementById('adminLoginForm'); // For admin users
    
    const logoutButton = document.getElementById('logoutButtonSidebar'); 

    if (registrationForm) {
        registrationForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            const username = event.target.username.value;
            const email = event.target.email.value;
            const password = event.target.password.value;
            const confirm_password = event.target.confirm_password.value;
            const referral_code_used = event.target.referral_code.value.trim(); // Get referral code

            if (password !== confirm_password) {
                alert("Passwords do not match!");
                return;
            }

            const registrationData = { username, email, password };
            if (referral_code_used) {
                registrationData.referral_code_used = referral_code_used;
            }

            console.log("Registering:", registrationData);
            
            try {
                const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/auth/register`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(registrationData)
                });

                const result = await response.json();

                if (response.ok) { // Check if the HTTP status code is in the 2xx range
                    alert(result.message);
                    // Redirect to login page after successful registration
                    window.location.href = 'login.html'; 
                } else {
                    // Handle backend errors (e.g., validation errors, user already exists)
                    alert('Registration failed: ' + (result.detail || result.message || "Unknown error."));
                }
            } catch (error) {
                console.error("Registration error:", error);
                alert("An error occurred during registration. Please try again.");
            }
        });
    }

    if (loginForm) {
        loginForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            const username_or_email = event.target.username_or_email.value;
            const password = event.target.password.value;

            console.log("Logging in:", { username_or_email });
            
            try {
                const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/auth/login`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, // Use form-urlencoded for OAuth2PasswordRequestForm
                    body: new URLSearchParams({ username: username_or_email, password: password }) // Use URLSearchParams for form data
                });

                const result = await response.json();

                if (response.ok) { // Check if the HTTP status code is in the 2xx range
                    // Store the token and user info
                    localStorage.setItem('authToken', result.access_token);
                    localStorage.setItem('tokenType', result.token_type);
                    localStorage.setItem('userId', result.user_id);
                    localStorage.setItem('username', result.username);
                    localStorage.setItem('isAdmin', result.is_admin.toString());
                    
                    alert("Login successful.");
                    // Redirect based on admin status
                    if (result.is_admin) {
                        window.location.href = 'admin/dashboard.html'; // Redirect to admin dashboard
                    } else {
                        window.location.href = 'dashboard.html'; // Redirect to regular user dashboard
                    }
                } else {
                    // Handle backend errors (e.g., invalid credentials)
                    alert('Login failed: ' + (result.detail || result.message || "Invalid credentials."));
                }
            } catch (error) {
                console.error("Login error:", error);
                alert("An error occurred during login. Please try again.");
            }
        });
    }

    if (adminLoginForm) {
        adminLoginForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            const username = event.target.admin_username.value;
            const password = event.target.admin_password.value;

            console.log("Admin logging in:", { username });
            
            try {
                // Assuming admin login uses the same /api/v1/auth/login endpoint but checks for admin status in response
                 const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/auth/login`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: new URLSearchParams({ username: username, password: password })
                });

                const result = await response.json();

                if (response.ok && result.is_admin) { // Check for both success status and admin flag
                    localStorage.setItem('authToken', result.access_token);
                    localStorage.setItem('tokenType', result.token_type);
                    localStorage.setItem('userId', result.user_id);
                    localStorage.setItem('username', result.username);
                    localStorage.setItem('isAdmin', 'true');
                    alert("Admin login successful. Redirecting to admin dashboard.");
                    // Assuming admin dashboard is frontend/admin/dashboard.html
                    // Need to adjust the path relative to the current page (auth.js is in frontend/js)
                    // The correct path from frontend/js to frontend/admin/dashboard.html is ../admin/dashboard.html
                    window.location.href = '../admin/dashboard.html'; 
                } else if (response.ok && !result.is_admin) {
                    alert('Login successful, but you do not have admin privileges for the admin panel.');
                }
                else {
                    // Handle backend errors
                    alert('Admin login failed: ' + (result.detail || result.message || "Invalid credentials or not an admin."));
                }
            } catch (error) {
                console.error("Admin login error:", error);
                alert("An error occurred during admin login. Please try again.");
            }
        });
    }

    if (logoutButton) {
        logoutButton.addEventListener('click', () => {
            console.log("Logging out...");
            // In a real app, you might also send a request to a backend logout endpoint
            // to invalidate the token server-side if using a mechanism like refresh tokens or a token blacklist.
            // For JWT in Authorization header, clearing client-side storage is often sufficient
            // as the token will eventually expire.

            localStorage.removeItem('authToken');
            localStorage.removeItem('tokenType');
            localStorage.removeItem('userId');
            localStorage.removeItem('username');
            localStorage.removeItem('isAdmin');
            alert("Logged out successfully.");
            
            // Determine if current page is in admin area to redirect to correct login
            if (window.location.pathname.includes('/admin/')) {
                // Redirect to admin login page (relative path from admin directory)
                window.location.href = 'login.html'; 
            } else {
                // Redirect to main user login page (relative path from root frontend directory)
                window.location.href = 'login.html'; 
            }
        });
    }
});
