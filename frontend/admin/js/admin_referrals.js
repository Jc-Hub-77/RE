// frontend/admin/js/admin_referrals.js
console.log("admin_referrals.js loaded");

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

    const referralsTableBody = document.getElementById('referralsTableBody');
    const prevPageButton = document.getElementById('prevPageReferrals'); 
    const nextPageButton = document.getElementById('nextPageReferrals');
    const pageInfoReferrals = document.getElementById('pageInfoReferrals');
    const filterReferrerInput = document.getElementById('filterReferrer');
    const filterReferredInput = document.getElementById('filterReferred');
    const filterReferralsButton = document.getElementById('filterReferralsButton');

    let currentReferralsPage = 1;
    const referralsPerPage = 15; // Number of items per page

    async function fetchAdminReferrals(page = 1) {
        if (!referralsTableBody) return;
        referralsTableBody.innerHTML = `<tr><td colspan="10" style="text-align:center;">Loading referral data...</td></tr>`;
        
        currentReferralsPage = page;
        const referrerSearch = filterReferrerInput ? filterReferrerInput.value : '';
        const referredSearch = filterReferredInput ? filterReferredInput.value : '';

        let queryParams = `page=${page}&per_page=${referralsPerPage}`;
        if (referrerSearch) queryParams += `&referrer_search=${encodeURIComponent(referrerSearch)}`;
        if (referredSearch) queryParams += `&referred_search=${encodeURIComponent(referredSearch)}`;
        // Add default sorting for consistency, can be made dynamic later
        queryParams += `&sort_by=pending_payout&sort_order=desc`;


        try {
            const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/referral?${queryParams}`, { 
                headers: { 'Authorization': `Bearer ${authToken}` } 
            });
            if (!response.ok) {
                if (response.status === 401 || response.status === 403) { window.location.href = 'login.html'; return; }
                const errData = await response.json().catch(()=>({}));
                throw new Error(errData.detail || errData.message || `HTTP error! status: ${response.status}`);
            }
            const data = await response.json(); // Expects referral_schemas.AdminReferralListResponse
            
            const referrals = data.referrals; 
            const totalPages = data.total_pages;
            currentReferralsPage = data.page; // Update current page from response

            if (data.status === "success" && referrals) {
                referralsTableBody.innerHTML = '';
                if (referrals.length === 0) {
                    referralsTableBody.innerHTML = '<tr><td colspan="10" style="text-align:center;">No referral records found for the current filters.</td></tr>';
                } else {
                    referrals.forEach(ref => {
                        const row = referralsTableBody.insertRow();
                        row.insertCell().textContent = ref.referral_id;
                        row.insertCell().textContent = `${ref.referrer_user_id} (${ref.referrer_username || 'N/A'})`;
                        row.insertCell().textContent = `${ref.referred_user_id} (${ref.referred_username || 'N/A'})`;
                        row.insertCell().textContent = ref.signed_up_at ? new Date(ref.signed_up_at).toLocaleDateString() : 'N/A';
                        row.insertCell().textContent = ref.first_payment_at ? new Date(ref.first_payment_at).toLocaleDateString() : 'N/A';
                        row.insertCell().textContent = `$${(ref.commission_earned_total || 0).toFixed(2)}`;
                        row.insertCell().textContent = `$${(ref.commission_pending_payout || 0).toFixed(2)}`;
                        row.insertCell().textContent = `$${(ref.commission_paid_out_total || 0).toFixed(2)}`;
                        row.insertCell().textContent = ref.last_payout_date ? new Date(ref.last_payout_date).toLocaleDateString() : 'N/A';
                        
                        const actionsCell = row.insertCell();
                        if (ref.commission_pending_payout > 0) {
                            const payButton = document.createElement('button');
                            payButton.className = 'btn btn-sm btn-success';
                            payButton.textContent = 'Mark Paid';
                            payButton.onclick = () => handleMarkCommissionPaid(ref.referral_id, ref.commission_pending_payout);
                            actionsCell.appendChild(payButton);
                        } else {
                            actionsCell.textContent = 'N/A';
                        }
                    });
                }
                updateReferralPaginationControls(totalPages);
            } else {
                throw new Error(data.message || "Failed to parse referral data.");
            }
        } catch (error) {
            console.error("Error fetching admin referrals:", error);
            referralsTableBody.innerHTML = `<tr><td colspan="10" style="text-align:center;">Error loading referral data: ${error.message}</td></tr>`;
            updateReferralPaginationControls(0);
        }
    }
    
    function updateReferralPaginationControls(totalPages) {
        if (pageInfoReferrals) pageInfoReferrals.textContent = `Page ${currentReferralsPage} of ${totalPages || 0}`;
        if (prevPageButton) prevPageButton.disabled = currentReferralsPage <= 1;
        if (nextPageButton) nextPageButton.disabled = currentReferralsPage >= totalPages;
    }

    async function handleMarkCommissionPaid(referralId, pendingAmount) {
        const amountToPayStr = prompt(`Enter amount paid for referral ID ${referralId} (pending: $${pendingAmount.toFixed(2)}):`, pendingAmount.toFixed(2));
        if (amountToPayStr === null) return; 

        const parsedAmount = parseFloat(amountToPayStr);
        if (isNaN(parsedAmount) || parsedAmount <= 0 || parsedAmount > pendingAmount) {
            alert("Invalid amount entered. Please enter a positive number up to the pending amount.");
            return;
        }

        const notes = prompt("Enter any notes for this payout (optional):");

        if (!confirm(`Mark $${parsedAmount.toFixed(2)} as paid for referral ID ${referralId}?`)) return;

        try {
            const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/referral/${referralId}/mark-paid`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` },
                body: JSON.stringify({ amount_paid: parsedAmount, notes: notes || "" })
            });
            
            const result = await response.json();
            if (!response.ok) { // Check HTTP status code for errors
                 const errorDetail = result.detail || result.message || `HTTP error! status: ${response.status}`;
                 if (Array.isArray(errorDetail)) { throw new Error(errorDetail.map(e => `${e.loc.join('->')}: ${e.msg}`).join(', ')); }
                 throw new Error(errorDetail);
            }
            // No need to check result.status if response.ok, assume success structure
            alert(result.message || "Commission payout recorded successfully.");
            fetchAdminReferrals(currentReferralsPage); 
        } catch (error) {
            console.error(`Error marking commission paid for referral ${referralId}:`, error);
            alert("Error: " + error.message);
        }
    }
    
    if (filterReferralsButton) {
        filterReferralsButton.addEventListener('click', () => fetchAdminReferrals(1)); // Reset to page 1 on new filter
    }
    if (prevPageButton) {
        prevPageButton.addEventListener('click', () => {
            if (currentReferralsPage > 1) fetchAdminReferrals(currentReferralsPage - 1);
        });
    }
    if (nextPageButton) {
        nextPageButton.addEventListener('click', () => {
            // The fetchAdminReferrals function will handle disabled state if totalPages is reached
            fetchAdminReferrals(currentReferralsPage + 1);
        });
    }

    if (authToken && isAdmin) {
        fetchAdminReferrals(); // Initial fetch
    } else {
         if(referralsTableBody) referralsTableBody.innerHTML = '<tr><td colspan="10" style="text-align:center;">Access Denied. Please login as admin.</td></tr>';
    }
});
