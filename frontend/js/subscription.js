// frontend/js/subscription.js
console.log("subscription.js loaded");

// const BACKEND_API_BASE_URL = 'http://127.0.0.1:8000'; // This will now be set globally via HTML script tag

document.addEventListener('DOMContentLoaded', () => {
    const authToken = localStorage.getItem('authToken');
    const userId = localStorage.getItem('userId');

    if (!authToken || !userId) {
        console.warn("User not authenticated. Redirecting to login from subscription.js");
        window.location.href = 'login.html'; // Adjust path if needed
        return;
    }

    const activeStrategySubsList = document.getElementById('activeStrategySubscriptionsList');
    const platformPlanName = document.getElementById('platformPlanName');
    const platformPlanStatus = document.getElementById('platformPlanStatus');
    const platformPlanExpiry = document.getElementById('platformPlanExpiry');
    const renewPlatformSubBtn = document.getElementById('renewPlatformSubBtn');
    const paymentSection = document.getElementById('paymentSection'); 
    const paymentHistoryTableBody = document.getElementById('paymentHistoryTableBody');

    async function initializeSubscriptionPage() {
        await fetchActiveStrategySubscriptions();
        await fetchPlatformSubscriptionDetails(); 
        await fetchPaymentHistory();

        if(renewPlatformSubBtn) renewPlatformSubBtn.addEventListener('click', handleGenericRenewInitiation);
        
        if(paymentSection) paymentSection.style.display = 'none'; 
    }

    async function fetchActiveStrategySubscriptions() {
        if (!activeStrategySubsList) return;
        activeStrategySubsList.innerHTML = '<p>Loading your strategy subscriptions...</p>';
        try {
            const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/strategies/subscriptions/me`, { 
                headers: { 'Authorization': `Bearer ${authToken}` } 
            });
            if (!response.ok) {
                if (response.status === 401) { window.location.href = 'login.html'; return; }
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json(); // Expects { status: "success", subscriptions: [...] }
            
            activeStrategySubsList.innerHTML = '';
            if (data.status === "success" && data.subscriptions && data.subscriptions.length > 0) {
                data.subscriptions.forEach(sub => {
                    const itemDiv = document.createElement('div');
                    itemDiv.className = 'subscription-item card mb-2'; 
                    const isActive = sub.is_active; 
                    const statusClass = isActive ? 'status-active' : 'status-expired';
                    const timeRemaining = isActive ? formatTimeRemaining(sub.time_remaining_seconds) : 'Expired';
                    const statusMessageText = sub.status_message || (isActive ? 'Running' : 'Inactive/Expired');

                    itemDiv.innerHTML = `
                        <h4>${sub.strategy_name}</h4>
                        <div class="subscription-details">
                            <p><strong>Status:</strong> <span class="${statusClass}">${statusMessageText}</span></p>
                            <p><strong>Expires:</strong> ${new Date(sub.expires_at).toLocaleString()} (${timeRemaining})</p>
                            <p><small>Subscription ID: ${sub.subscription_id} | API Key ID: ${sub.api_key_id}</small></p>
                            <p><small>Celery Task ID: ${sub.celery_task_id || 'N/A'}</small></p>
                        </div>
                        <button class="btn btn-sm renew-strategy-sub-btn mt-1" 
                                data-sub-id="${sub.subscription_id}" 
                                data-item-name="${sub.strategy_name} Renewal"
                                data-item-type="renew_strategy_subscription"
                                data-item-description="1 Month Renewal for ${sub.strategy_name}"
                                data-amount-usd="10.00" 
                                data-subscription-months="1"> 
                            Renew ($10.00)
                        </button>
                    `; // Added data-subscription-months
                    activeStrategySubsList.appendChild(itemDiv);
                });
                document.querySelectorAll('.renew-strategy-sub-btn').forEach(button => {
                    button.addEventListener('click', handleGenericRenewInitiation);
                });
            } else {
                activeStrategySubsList.innerHTML = '<p>No active strategy subscriptions found.</p>';
                if (data.status !== "success") throw new Error(data.message || "Failed to parse subscriptions.");
            }
        } catch (error) {
            console.error("Error loading strategy subscriptions:", error);
            activeStrategySubsList.innerHTML = `<p class="error-message">Error: ${error.message}</p>`;
        }
    }

    async function fetchPlatformSubscriptionDetails() {
        if (!platformPlanName && !platformPlanStatus && !platformPlanExpiry) return;
        try {
            const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/user_data/users/${userId}/platform_subscription`, { 
                headers: { 'Authorization': `Bearer ${authToken}` } 
            });
            if (!response.ok) {
                if (response.status === 401) { window.location.href = 'login.html'; return; }
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();

            if (data.status === "success" && data.subscription) {
                const sub = data.subscription;
                if(platformPlanName) platformPlanName.textContent = sub.plan_name || "N/A";
                const isActive = sub.is_active && sub.expires_at && new Date(sub.expires_at) > new Date();
                if(platformPlanStatus) {
                    platformPlanStatus.textContent = isActive ? 'Active' : 'Expired/None';
                    platformPlanStatus.className = isActive ? 'status-active' : 'status-expired';
                }
                if(platformPlanExpiry) platformPlanExpiry.textContent = sub.expires_at ? new Date(sub.expires_at).toLocaleDateString() : "N/A";
                if(renewPlatformSubBtn) {
                    renewPlatformSubBtn.style.display = 'inline-block'; // Make it visible
                    renewPlatformSubBtn.dataset.itemId = sub.plan_id || "platform_basic_annual"; // Assuming plan_id exists
                    renewPlatformSubBtn.dataset.itemName = `${sub.plan_name || "Platform"} Renewal`;
                    renewPlatformSubBtn.dataset.itemType = "platform_subscription_renewal"; // Specific item type
                    renewPlatformSubBtn.dataset.itemDescription = `1 Year Renewal for ${sub.plan_name || "Platform"}`;
                    renewPlatformSubBtn.dataset.amountUsd = "99.00"; // Example, should come from plan details
                    renewPlatformSubBtn.dataset.subscriptionMonths = "12";
                }
            } else { 
                if(platformPlanName) platformPlanName.textContent = "Free Tier / None";
                if(platformPlanStatus) { platformPlanStatus.textContent = "N/A"; platformPlanStatus.className = "";}
                if(platformPlanExpiry) platformPlanExpiry.textContent = "N/A";
                if (data.status === "error" && data.message !== "User has no active platform subscription.") { // Don't throw error for no sub
                    throw new Error(data.message || "Failed to load platform subscription.");
                } else if (data.message === "User has no active platform subscription.") {
                    console.info("User has no active platform subscription.");
                }
            }
        } catch (error) {
            console.error("Error loading platform subscription:", error);
            if(platformPlanName) platformPlanName.textContent = "Error";
        }
    }

    async function fetchPaymentHistory() {
        if (!paymentHistoryTableBody) return;
        paymentHistoryTableBody.innerHTML = '<tr><td colspan="6" style="text-align:center;">Loading payment history...</td></tr>';
        try {
            const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/payment/history/me?page=1&per_page=10`, { // Added default pagination
                headers: { 'Authorization': `Bearer ${authToken}` } 
            });
            if (!response.ok) {
                if (response.status === 401) { window.location.href = 'login.html'; return; }
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json(); 

            paymentHistoryTableBody.innerHTML = '';
            if (data.status === "success" && data.payment_history && data.payment_history.length > 0) {
                data.payment_history.forEach(p => {
                    const row = paymentHistoryTableBody.insertRow();
                    row.insertCell().textContent = new Date(p.date).toLocaleDateString();
                    row.insertCell().textContent = p.description || `Sub ID: ${p.subscription_id || 'N/A'}`;
                    // Assuming amount_crypto and crypto_currency are main fields, usd_equivalent is secondary
                    row.insertCell().textContent = `${p.amount_crypto ? p.amount_crypto.toFixed(p.crypto_currency === "USDC" ? 2 : 8) : (p.usd_equivalent ? p.usd_equivalent.toFixed(2) : 'N/A')}`;
                    row.insertCell().textContent = p.crypto_currency || (p.usd_equivalent ? 'USD (Equivalent)' : 'N/A');
                    row.insertCell().innerHTML = `<span class="status-${p.status.toLowerCase()}">${p.status}</span>`;
                    row.insertCell().textContent = p.gateway_id || p.internal_reference;
                });
            } else {
                paymentHistoryTableBody.innerHTML = '<tr><td colspan="6" style="text-align:center;">No payment history found.</td></tr>';
                if (data.status !== "success") throw new Error(data.message || "Failed to parse payment history.");
            }
        } catch (error) {
            console.error("Error loading payment history:", error);
            paymentHistoryTableBody.innerHTML = `<tr><td colspan="6" style="text-align:center;">Error: ${error.message}</td></tr>`;
        }
    }

    async function handleGenericRenewInitiation(event) {
        const button = event.target;
        const itemId = button.dataset.subId || button.dataset.itemId; 
        const itemName = button.dataset.itemName;
        const itemType = button.dataset.itemType;
        const itemDescription = button.dataset.itemDescription || itemName;
        const amountUsd = parseFloat(button.dataset.amountUsd);
        const subscriptionMonths = parseInt(button.dataset.subscriptionMonths || "1");

        if (!itemId || !itemName || !itemType || isNaN(amountUsd) || isNaN(subscriptionMonths)) {
            alert("Error: Missing required information for renewal/payment.");
            return;
        }
        
        // console.info(`Initiating payment for ${itemType} ID ${itemId}: ${itemName}, Amount: $${amountUsd}`); // Changed from logger to console
        console.log(`Initiating payment for ${itemType} ID ${itemId}: ${itemName}, Amount: $${amountUsd}`);
        button.disabled = true; button.textContent = "Processing...";

        try {
            const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/payment/charges`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` },
                body: JSON.stringify({ 
                    item_id: parseInt(itemId), // Ensure item_id is int if it's a DB ID
                    item_type: itemType, 
                    item_name: itemName, 
                    item_description: itemDescription, 
                    amount_usd: amountUsd, 
                    subscription_months: subscriptionMonths,
                    metadata: { // Pass necessary metadata for webhook processing
                        user_id: userId, // Already string from localStorage
                        item_id: parseInt(itemId), // strategy_db_id or user_strategy_subscription_id
                        item_type: itemType,
                        subscription_months: subscriptionMonths
                        // For new_strategy_subscription, api_key_id and custom_parameters_json would be needed here.
                        // For renewals, backend can fetch existing sub details.
                    }
                })
            });
            const chargeData = await response.json(); 
            if (!response.ok || chargeData.status === "error") {
                 throw new Error(chargeData.message || chargeData.detail || `HTTP error! status: ${response.status}`);
            }
            
            if ((chargeData.status === "success" || chargeData.status === "success_simulated") && chargeData.payment_page_url) {
                alert(`You will be redirected to complete your payment for ${itemName}.`);
                window.location.href = chargeData.payment_page_url; 
            } else if (chargeData.status === "success_simulated") {
                 alert("Payment simulation successful. Subscription should update shortly.");
                 initializeSubscriptionPage(); // Refresh the page data
            }
            else {
                throw new Error(chargeData.message || "Failed to create payment charge: No payment_page_url.");
            }
        } catch (error) {
            console.error("Error initiating payment:", error);
            alert("Error initiating payment: " + error.message);
            button.disabled = false; 
            button.textContent = `Renew ($${amountUsd.toFixed(2)})`; 
        }
    }
    
    function formatTimeRemaining(totalSeconds) {
        if (totalSeconds <= 0) return "Expired";
        const days = Math.floor(totalSeconds / (24 * 60 * 60));
        const hours = Math.floor((totalSeconds % (24 * 60 * 60)) / (60 * 60));
        const minutes = Math.floor((totalSeconds % (60 * 60)) / 60);
        let parts = [];
        if (days > 0) parts.push(`${days}d`);
        if (hours > 0 && days < 3) parts.push(`${hours}h`); 
        if (minutes > 0 && days === 0 && hours < 1) parts.push(`${minutes}m`); // Show minutes if less than an hour left
        if (parts.length === 0 && totalSeconds > 0) return "<1m"; // Show <1m if very little time left but not expired
        return parts.join(' ') || "Expired"; // Default to Expired if no parts (shouldn't happen if totalSeconds > 0)
    }

    if (authToken && userId) {
        initializeSubscriptionPage();
    } else {
        // This case should ideally be handled by a global auth check that redirects to login.html
        console.error("User not authenticated. Cannot display subscription page.");
        // Optionally, clear parts of the page or show a login prompt.
        if(activeStrategySubsList) activeStrategySubsList.innerHTML = '<p>Please login to view your subscriptions.</p>';
    }
});
