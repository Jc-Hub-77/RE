// frontend/admin/js/admin_dashboard.js
console.log("admin_dashboard.js loaded");

document.addEventListener('DOMContentLoaded', () => {
    const authToken = localStorage.getItem('authToken');
    const isAdmin = localStorage.getItem('isAdmin') === 'true';

    // Basic Admin Auth Check (more robust checks should be on backend for API calls)
    // if (!isAdmin || !authToken) {
    //     alert("Access Denied. You are not authorized to view this page or your session has expired.");
    //     // Adjust path if auth.js is in a different relative location from admin pages
    //     window.location.href = 'login.html'; // Redirect to admin login
    //     return;
    // }

    const adminTotalUsers = document.getElementById('adminTotalUsers');
    const adminActiveSubscriptions = document.getElementById('adminActiveSubscriptions');
    const adminTotalStrategies = document.getElementById('adminTotalStrategies');
    const adminTotalRevenue = document.getElementById('adminTotalRevenue');
    const adminRecentActivity = document.getElementById('adminRecentActivity');

    async function fetchAdminDashboardData() {
        console.log("Fetching admin dashboard data...");
        
        try {
            // Fetch data from the new backend endpoint
            const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/admin/dashboard-summary`, {
                headers: { 'Authorization': `Bearer ${authToken}` }
            });

            if (!response.ok) {
                if (response.status === 401 || response.status === 403) { // Unauthorized or Forbidden
                    alert("Session expired or unauthorized. Please log in again.");
                    window.location.href = 'login.html';
                }
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();

            if (data.status === "success" && data.summary) {
                const summary = data.summary;
                if (adminTotalUsers) adminTotalUsers.textContent = summary.totalUsers.toLocaleString();
                if (adminActiveSubscriptions) adminActiveSubscriptions.textContent = summary.activeSubscriptions.toLocaleString();
                if (adminTotalStrategies) adminTotalStrategies.textContent = summary.totalStrategies.toLocaleString();
                if (adminTotalRevenue) adminTotalRevenue.textContent = `$${summary.totalRevenueLast30d.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
                
                if (adminRecentActivity && summary.recentActivities) {
                    adminRecentActivity.innerHTML = '<ul></ul>';
                    const ul = adminRecentActivity.querySelector('ul');
                    summary.recentActivities.forEach(activity => {
                        const li = document.createElement('li');
                        li.textContent = activity;
                        li.style.padding = '3px 0';
                        ul.appendChild(li);
                    });
                }
            } else {
                throw new Error(data.message || "Failed to parse dashboard summary.");
            }
        } catch (error) {
            console.error("Failed to fetch admin dashboard data:", error);
            if (adminTotalUsers) adminTotalUsers.textContent = "Error";
            if (adminActiveSubscriptions) adminActiveSubscriptions.textContent = "Error";
            if (adminTotalStrategies) adminTotalStrategies.textContent = "Error";
            if (adminTotalRevenue) adminTotalRevenue.textContent = "Error";
            if (adminRecentActivity) adminRecentActivity.innerHTML = `<p class="error-message">Could not load recent activity: ${error.message}</p>`;
        }
    }

    fetchAdminDashboardData();
});
