// frontend/admin/js/admin_subscriptions.js
console.log("admin_subscriptions.js loaded");

document.addEventListener('DOMContentLoaded', () => {
    const authToken = localStorage.getItem('authToken');
    const isAdmin = localStorage.getItem('isAdmin') === 'true';

    if (!isAdmin && !authToken) { 
        alert("Access Denied. You are not authorized to view this page or your session has expired.");
        window.location.href = 'login.html'; 
        return;
    }
    if (!authToken) { 
        alert("Session expired. Please log in again.");
        window.location.href = 'login.html';
        return;
    }

    const subscriptionsTableBody = document.getElementById('subscriptionsTableBody');
    const prevPageButton = document.getElementById('prevPageSubscriptions'); // Assuming these IDs
    const nextPageButton = document.getElementById('nextPageSubscriptions');
    const pageInfoSubscriptions = document.getElementById('pageInfoSubscriptions');
    // Assuming filter inputs might be added later
    // const filterUserInput = document.getElementById('filterUserSubscriptions');
    // const filterStrategyInput = document.getElementById('filterStrategySubscriptions');
    // const filterSubscriptionsButton = document.getElementById('filterSubscriptionsButton');

    let currentSubscriptionsPage = 1;
    const subscriptionsPerPage = 15;

    async function fetchAdminSubscriptions(page = 1) {
        if (!subscriptionsTableBody) return;
        subscriptionsTableBody.innerHTML = `<tr><td colspan="9" style="text-align:center;">Loading subscriptions...</td></tr>`; // Updated colspan to 9

        currentSubscriptionsPage = page;
        // Add filters if corresponding HTML inputs exist
        // const userIdFilter = filterUserInput ? filterUserInput.value : '';
        // const strategyIdFilter = filterStrategyInput ? filterStrategyInput.value : '';
        // let queryParams = `page=${page}&per_page=${subscriptionsPerPage}`;
        // if (userIdFilter) queryParams += `&user_id=${encodeURIComponent(userIdFilter)}`;
        // if (strategyIdFilter) queryParams += `&strategy_id=${encodeURIComponent(strategyIdFilter)}`;
        
        // Assuming a general admin endpoint for all subscriptions exists or will be created
        // For now, using a placeholder. This endpoint would need to be implemented in the backend.
        // Based on admin_service.py, it should be something like:
        let queryParams = `page=${page}&per_page=${subscriptionsPerPage}`;


        try {
            // ASSUMED BACKEND ENDPOINT: /api/v1/admin/subscriptions (needs to be created in admin_router.py)
            // This endpoint would call admin_service.list_all_subscriptions_admin
            const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/admin/all-subscriptions?${queryParams}`, { 
                headers: { 'Authorization': `Bearer ${authToken}` } 
            });

            if (!response.ok) {
                if (response.status === 401 || response.status === 403) { window.location.href = 'login.html'; return; }
                const errData = await response.json().catch(()=>({})); // Try to parse error
                throw new Error(errData.detail || errData.message || `HTTP error! status: ${response.status}`);
            }
            const data = await response.json(); // Expects { status, subscriptions, total_subscriptions, page, per_page, total_pages }
            
            const subscriptions = data.subscriptions; // Assuming the list is in 'subscriptions'
            const totalPages = data.total_pages;
            currentSubscriptionsPage = data.page;

            if (data.status === "success" && subscriptions) {
                subscriptionsTableBody.innerHTML = '';
                if (subscriptions.length === 0) {
                    subscriptionsTableBody.innerHTML = '<tr><td colspan="9" style="text-align:center;">No subscriptions found.</td></tr>';
                } else {
                    subscriptions.forEach(sub => {
                        const row = subscriptionsTableBody.insertRow();
                        row.insertCell().textContent = sub.id;
                        row.insertCell().textContent = `${sub.user_id} (${sub.username || 'N/A'})`;
                        row.insertCell().textContent = `${sub.strategy_id} (${sub.strategy_name || 'N/A'})`;
                        row.insertCell().textContent = sub.api_key_id;
                        
                        const isActive = sub.is_active && new Date(sub.expires_at) > new Date();
                        row.insertCell().innerHTML = `<span class="status-${isActive ? 'active' : 'inactive'}">${sub.status_message || (isActive ? 'Active' : 'Inactive/Expired')}</span>`;
                        
                        row.insertCell().textContent = sub.subscribed_at ? new Date(sub.subscribed_at).toLocaleString() : 'N/A';
                        row.insertCell().textContent = sub.expires_at ? new Date(sub.expires_at).toLocaleString() : 'N/A';
                        row.insertCell().textContent = sub.celery_task_id || 'N/A';
                        
                        const actionsCell = row.insertCell();
                        const viewButton = document.createElement('button');
                        viewButton.className = 'btn btn-sm btn-outline';
                        viewButton.textContent = 'Details';
                        viewButton.onclick = () => handleViewSubscription(sub); // Pass full sub object
                        actionsCell.appendChild(viewButton);

                        if (isActive) {
                            const deactivateButton = document.createElement('button');
                            deactivateButton.className = 'btn btn-sm btn-warning'; // Warning for deactivation
                            deactivateButton.textContent = 'Deactivate';
                            deactivateButton.style.marginLeft = '5px';
                            deactivateButton.onclick = () => handleAdminDeactivateSubscription(sub.id);
                            actionsCell.appendChild(deactivateButton);
                        }
                        // Add manual edit button if needed (e.g., to change expiry, status_message)
                        const editButton = document.createElement('button');
                        editButton.className = 'btn btn-sm btn-info';
                        editButton.textContent = 'Edit';
                        editButton.style.marginLeft = '5px';
                        editButton.onclick = () => handleAdminEditSubscription(sub);
                        actionsCell.appendChild(editButton);

                    });
                }
                updateSubscriptionPaginationControls(totalPages);
            } else {
                throw new Error(data.message || "Failed to parse subscriptions list.");
            }
        } catch (error) {
            console.error("Error fetching admin subscriptions:", error);
            subscriptionsTableBody.innerHTML = `<tr><td colspan="9" style="text-align:center;">Error loading subscriptions: ${error.message}</td></tr>`;
            updateSubscriptionPaginationControls(0);
        }
    }

    function updateSubscriptionPaginationControls(totalPages) {
        if (pageInfoSubscriptions) pageInfoSubscriptions.textContent = `Page ${currentSubscriptionsPage} of ${totalPages || 0}`;
        if (prevPageButton) prevPageButton.disabled = currentSubscriptionsPage <= 1;
        if (nextPageButton) nextPageButton.disabled = currentSubscriptionsPage >= totalPages;
    }

    function handleViewSubscription(subData) {
        // This should ideally populate a modal with more details, including custom_parameters
        alert(`View Subscription Details (Admin):\nID: ${subData.id}\nUser: ${subData.user_id} (${subData.username})\nStrategy: ${subData.strategy_id} (${subData.strategy_name})\nIs Active: ${subData.is_active}\nExpires: ${new Date(subData.expires_at).toLocaleString()}\nStatus Msg: ${subData.status_message}\nCustom Params: ${JSON.stringify(subData.custom_parameters)}`);
        console.log("Subscription Details:", subData);
    }
    
    async function handleAdminEditSubscription(subData) {
        // Example: Prompt for new status message and expiry. A modal form is better.
        const newStatusMessage = prompt("Enter new status message (or leave blank):", subData.status_message);
        const newExpiresAtStr = prompt("Enter new expiry date (YYYY-MM-DDTHH:MM:SS, or leave blank):", subData.expires_at ? subData.expires_at.substring(0,19) : "");
        const newIsActiveStr = prompt("Set active status? (true/false, or leave blank to keep current):", String(subData.is_active));

        const payload = {};
        if (newStatusMessage !== null && newStatusMessage !== subData.status_message) payload.new_status_message = newStatusMessage;
        if (newExpiresAtStr !== null && newExpiresAtStr !== (subData.expires_at ? subData.expires_at.substring(0,19) : "")) payload.new_expires_at_str = newExpiresAtStr;
        if (newIsActiveStr !== null && newIsActiveStr !== String(subData.is_active)) payload.new_is_active = (newIsActiveStr.toLowerCase() === 'true');

        if (Object.keys(payload).length === 0) {
            alert("No changes made."); return;
        }

        if (!confirm(`Update subscription ${subData.id} with these changes?\n${JSON.stringify(payload, null, 2)}`)) return;
        
        try {
            // ASSUMED BACKEND ENDPOINT: PUT /api/v1/admin/subscriptions/{sub_id}/details 
            const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/admin/subscriptions/${subData.id}/details`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` },
                body: JSON.stringify(payload)
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.detail || result.message || "Failed to update subscription details.");
            alert(result.message || "Subscription details updated.");
            fetchAdminSubscriptions(currentSubscriptionsPage);
        } catch (error) {
            console.error("Error updating subscription details:", error);
            alert("Error: " + error.message);
        }
    }


    async function handleAdminDeactivateSubscription(subscriptionId) {
        if (!confirm(`Are you sure you want to deactivate subscription ID ${subscriptionId}? This will stop its live trading bot.`)) return;

        try {
            // ASSUMED BACKEND ENDPOINT: POST /api/v1/admin/subscriptions/{subscription_id}/deactivate
            const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/admin/subscriptions/${subscriptionId}/deactivate`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${authToken}` }
            });
            const result = await response.json();
            if (!response.ok) {
                 const errorDetail = result.detail || result.message || `HTTP error! status: ${response.status}`;
                 throw new Error(errorDetail);
            }
            alert(result.message || `Subscription ${subscriptionId} deactivation process initiated.`);
            fetchAdminSubscriptions(currentSubscriptionsPage); 
        } catch (error) {
            console.error(`Error deactivating subscription ${subscriptionId}:`, error);
            alert("Error: " + error.message);
        }
    }
    
    // Add event listeners for pagination and filters if HTML elements are added
    // e.g., filterButton.addEventListener('click', () => fetchAdminSubscriptions(1));
    // prevPageButton.addEventListener('click', ...); nextPageButton.addEventListener('click', ...);

    if (authToken && isAdmin) {
        fetchAdminSubscriptions(); // Initial fetch
    } else {
        if(subscriptionsTableBody) subscriptionsTableBody.innerHTML = '<tr><td colspan="9" style="text-align:center;">Access Denied. Please login as admin.</td></tr>';
    }
});
