// frontend/js/dashboard.js
console.log("dashboard.js loaded");

// const BACKEND_API_BASE_URL = 'http://127.0.0.1:8000'; // This will now be set globally via HTML script tag

document.addEventListener('DOMContentLoaded', () => {
    const authToken = localStorage.getItem('authToken');
    const userId = localStorage.getItem('userId');

    if (!authToken || !userId) {
        console.warn("User not authenticated. Redirecting to login.");
        window.location.href = 'login.html';
        return; // Stop execution if not authenticated
    }

    // DOM Elements for dashboard data
    const dashUsername = document.getElementById('dashUsername');
    const dashEmail = document.getElementById('dashEmail');

    const dashOverallPnl = document.getElementById('dashOverallPnl');
    const dashActiveBots = document.getElementById('dashActiveBots');
    const dashSubPlanMain = document.getElementById('dashSubPlanMain');
    const dashSubExpiryMain = document.getElementById('dashSubExpiryMain');

    const activeSubscriptionsList = document.getElementById('activeSubscriptionsList');
    const dashExchangeList = document.getElementById('dashExchangeList');

    // Referral Section Elements
    const userReferralCodeElem = document.getElementById('userReferralCode');
    const copyReferralCodeBtn = document.getElementById('copyReferralCodeBtn');
    const totalReferralsCountElem = document.getElementById('totalReferralsCount');
    const activeReferralsCountElem = document.getElementById('activeReferralsCount');
    const pendingCommissionAmountElem = document.getElementById('pendingCommissionAmount');
    const totalCommissionEarnedElem = document.getElementById('totalCommissionEarned');

    const updateProfileBtn = document.getElementById('updateProfileBtn');
    const changePasswordBtn = document.getElementById('changePasswordBtn');

    // Logout button is handled by auth.js if it's included on the page and targets the same ID.
    // If auth.js is not on dashboard.html, or if sidebar logout is specific, keep this:
    // const logoutButtonSidebar = document.getElementById('logoutButtonSidebar');
    // if (logoutButtonSidebar) { ... }


    async function initializeDashboard() {
        if (!userId) {
            console.error("User ID not found. Cannot load dashboard.");
            if(dashUsername) dashUsername.textContent = "Error: Not Logged In";
            // Potentially redirect to login - already handled above
            return;
        }
        console.log(`Initializing dashboard for user ID: ${userId}`);

        // Fetch all necessary data in parallel or sequentially
        await fetchUserProfile(); // Basic profile info
        await fetchActiveUserStrategySubscriptions(); // For "Active Subscriptions" card
        await fetchConnectedExchangesSummary();
        await fetchPerformanceSummary();
        await fetchUserReferralStats(); // New function for referral stats
        // await fetchRunningStrategiesStatus(); // Removed: Strategy status will be shown within fetchActiveUserStrategySubscriptions
        await fetchUserPlatformSubscription(); // Fetch and display platform subscription
    }

    async function fetchUserProfile() {
        if (!dashUsername || !dashEmail) return;
        try {
            const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/auth/users/me`, { headers: { 'Authorization': `Bearer ${authToken}` } });
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json(); // Expects { status, username, email, ... }

            if (data.status === "success") {
                if (dashUsername) dashUsername.textContent = data.username || "N/A";
                if (dashEmail) dashEmail.textContent = data.email || "N/A";
            } else { throw new Error(data.message || "Failed to parse profile."); }
        } catch (error) {
            console.error("Error fetching user profile:", error);
            if (dashUsername) dashUsername.textContent = "Error";
            if (dashEmail) dashEmail.textContent = "Error";
        }
    }

    async function fetchPerformanceSummary() {
        if (!dashOverallPnl || !dashActiveBots) return;
        try {
            // Corrected API path
            const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/user_data/users/${userId}/performance-summary`, { headers: { 'Authorization': `Bearer ${authToken}` } });
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json(); // Expects { status, overall_pnl_30d, active_bots_count, ... }

            if (data.status === "success") {
                if (dashOverallPnl) dashOverallPnl.textContent = `$${data.overall_pnl_30d?.toFixed(2) || '0.00'}`;
                if (dashActiveBots) dashActiveBots.textContent = data.active_bots_count || 0;
            } else { throw new Error(data.message || "Failed to parse performance summary."); }
        } catch (error) {
            console.error("Error fetching performance summary:", error);
            if (dashOverallPnl) dashOverallPnl.textContent = "$Error";
            if (dashActiveBots) dashActiveBots.textContent = "Error";
        }
    }

    async function fetchActiveUserStrategySubscriptions() {
        if (!activeSubscriptionsList) return;
        activeSubscriptionsList.innerHTML = '<p>Loading active subscriptions...</p>';
        try {
            // Corrected API path
            const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/user_data/users/${userId}/strategy_subscriptions`, { headers: { 'Authorization': `Bearer ${authToken}` } });
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json(); // Expects { status, subscriptions: [...] }

            activeSubscriptionsList.innerHTML = ''; // Clear loading message
            if (data.status === "success" && data.subscriptions && data.subscriptions.length > 0) {
                const ul = document.createElement('ul');
                ul.style.listStyle = 'none'; ul.style.paddingLeft = '0';
                data.subscriptions.forEach(sub => {
                    const li = document.createElement('li');
                    // Assuming strategy_name, api_key_id, expires_at, status_message, is_active are available
                    const expiresDate = sub.expires_at ? new Date(sub.expires_at).toLocaleDateString() : 'Never';
                    const statusText = sub.is_active ? (sub.status_message || 'Running') : (sub.status_message || 'Inactive');
                    const statusClass = sub.is_active ? 'status-active' : 'status-inactive';
                    
                    li.innerHTML = `
                        <strong>${sub.strategy_name || 'Unnamed Strategy'}</strong> (API Key ID: ${sub.api_key_id || 'N/A'})<br>
                        Status: <span class="${statusClass}">${statusText}</span><br>
                        Expires: ${expiresDate}
                    `;
                    li.style.padding = '8px 0';
                    li.style.borderBottom = '1px solid var(--border-color)';
                    ul.appendChild(li);
                });
                activeSubscriptionsList.appendChild(ul);
            } else {
                activeSubscriptionsList.innerHTML = '<p>No active strategy subscriptions.</p>';
            }
        } catch (error) {
            console.error("Error fetching active subscriptions:", error);
            activeSubscriptionsList.innerHTML = `<p class="error-message">Error: ${error.message}</p>`;
        }
    }

    async function fetchConnectedExchangesSummary() {
        if (!dashExchangeList) return;
        dashExchangeList.innerHTML = '<li>Loading connected exchanges...</li>';
        try {
            // Corrected API path - fetching all keys for the user
            const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/exchanges/api-keys`, { headers: { 'Authorization': `Bearer ${authToken}` } });
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json(); // Expects { status, keys: [...] }

            dashExchangeList.innerHTML = ''; // Clear loading message
            if (data.status === "success" && data.keys && data.keys.length > 0) {
                data.keys.forEach(exKey => {
                    const listItem = document.createElement('li');
                    // Assuming label, exchange_name, status are available in the response
                    listItem.innerHTML = `<strong>${exKey.label || 'Unnamed Key'}</strong> (${exKey.exchange_name || 'N/A'}) - Status: <span class="status-${exKey.status || 'unknown'}">${exKey.status || 'unknown'}</span>`;
                    listItem.style.padding = '5px 0';
                    dashExchangeList.appendChild(listItem);
                });
            } else {
                dashExchangeList.innerHTML = '<li>No exchanges connected.</li>';
            }
        } catch (error) {
            console.error("Error fetching exchange summary:", error);
            dashExchangeList.innerHTML = `<li>Error: ${error.message}</li>`;
        }
    }

    async function fetchUserReferralStats() {
        if (!userReferralCodeElem) return; // Check if referral section elements exist
        try {
            // Corrected API path
            const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/user_data/users/${userId}/referral-stats`, { headers: { 'Authorization': `Bearer ${authToken}` } });
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json(); // Expects { status, referral_code, total_referrals, ... }

            if (data.status === "success") {
                if(userReferralCodeElem) userReferralCodeElem.textContent = data.referral_code || "N/A";
                if(totalReferralsCountElem) totalReferralsCountElem.textContent = data.total_referrals?.toLocaleString() || '0';
                if(activeReferralsCountElem) activeReferralsCountElem.textContent = data.active_referrals?.toLocaleString() || '0';
                if(pendingCommissionAmountElem) pendingCommissionAmountElem.textContent = `$${data.pending_commission_payout?.toFixed(2) || '0.00'}`;
                if(totalCommissionEarnedElem) totalCommissionEarnedElem.textContent = `$${data.total_commission_earned?.toFixed(2) || '0.00'}`;
            } else {
                throw new Error(data.message || "Failed to load referral stats.");
            }
        } catch (error) {
            console.error("Error fetching referral stats:", error);
            if(userReferralCodeElem) userReferralCodeElem.textContent = "Error";
            if(totalReferralsCountElem) totalReferralsCountElem.textContent = "Error";
            if(activeReferralsCountElem) activeReferralsCountElem.textContent = "Error";
            if(pendingCommissionAmountElem) pendingCommissionAmountElem.textContent = "$Error";
            if(totalCommissionEarnedElem) totalCommissionEarnedElem.textContent = "$Error";
        }
    }

    async function fetchRunningStrategiesStatus() {
        // This assumes there's an element on the dashboard to display this, e.g., <div id="runningStrategiesStatus"></div>
        const runningStrategiesStatusElem = document.getElementById('runningStrategiesStatus');
        if (!runningStrategiesStatusElem) {
            console.warn("Element #runningStrategiesStatus not found. Cannot display running strategies status.");
            return;
        }
        runningStrategiesStatusElem.innerHTML = '<p>Loading running strategies status...</p>';

        try {
            // Corrected API path
            const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/live_trading/strategies/status`, {
                headers: { 'Authorization': `Bearer ${authToken}` }
            });
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json(); // Expects { status, running_strategies: [...] }

            if (data.status === "success" && data.running_strategies) {
                if (data.running_strategies.length > 0) {
                    runningStrategiesStatusElem.innerHTML = `<h4>Running Strategies:</h4><ul>${data.running_strategies.map(s => `<li>${s.strategy_name || 'Unnamed Strategy'} (ID: ${s.subscription_id || 'N/A'}) - Status: ${s.status || 'unknown'}</li>`).join('')}</ul>`;
                } else {
                    runningStrategiesStatusElem.innerHTML = '<p>No strategies are currently running.</p>';
                }
            } else {
                throw new Error(data.message || "Failed to load running strategies status.");
            }
        } catch (error) {
            console.error("Error fetching running strategies status:", error);
            runningStrategiesStatusElem.innerHTML = `<p class="error-message">Error loading running strategies status: ${error.message}</p>`;
        }
    }

    async function fetchUserPlatformSubscription() {
         if (!dashSubPlanMain || !dashSubExpiryMain) return;
         try {
             // Corrected API path
             const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/user_data/users/${userId}/platform_subscription`, { headers: { 'Authorization': `Bearer ${authToken}` } });
             if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
             const data = await response.json(); // Expects { status, subscription: { plan_name, is_active, expires_at } }

             if (data.status === "success" && data.subscription) {
                 if (dashSubPlanMain) dashSubPlanMain.textContent = data.subscription.plan_name || "Free Tier";
                 if (dashSubExpiryMain) {
                     if (data.subscription.is_active && data.subscription.expires_at) {
                         const expiryDate = new Date(data.subscription.expires_at).toLocaleDateString();
                         dashSubExpiryMain.textContent = `Expires: ${expiryDate}`;
                     } else if (data.subscription.is_active) {
                          dashSubExpiryMain.textContent = "Active (No Expiry)";
                     }
                     else {
                         dashSubExpiryMain.textContent = "Inactive";
                     }
                 }
             } else {
                 // Handle case where user has no platform subscription or error
                 if (dashSubPlanMain) dashSubPlanMain.textContent = "Free Tier";
                 if (dashSubExpiryMain) dashSubExpiryMain.textContent = "N/A";
                 if (data.status === "error") throw new Error(data.message || "Failed to load platform subscription.");
             }
         } catch (error) {
             console.error("Error fetching platform subscription:", error);
             if (dashSubPlanMain) dashSubPlanMain.textContent = "Error";
             if (dashSubExpiryMain) dashSubExpiryMain.textContent = "Error";
         }
    }


    if (copyReferralCodeBtn && userReferralCodeElem) {
        copyReferralCodeBtn.addEventListener('click', () => {
            const codeToCopy = userReferralCodeElem.textContent;
            if (codeToCopy && codeToCopy !== "Loading..." && codeToCopy !== "Error") {
                navigator.clipboard.writeText(codeToCopy).then(() => {
                    alert("Referral code copied to clipboard!");
                }).catch(err => {
                    console.error('Failed to copy referral code: ', err);
                    alert("Failed to copy code. Please copy manually.");
                });
            } else {
                alert("No referral code to copy.");
            }
        });
    }

    if (updateProfileBtn) {
        updateProfileBtn.addEventListener('click', () => alert("Update Profile (to be implemented)."));
    }
    if (changePasswordBtn) {
        changePasswordBtn.addEventListener('click', () => alert("Change Password (to be implemented)."));
    }

    initializeDashboard();
});
