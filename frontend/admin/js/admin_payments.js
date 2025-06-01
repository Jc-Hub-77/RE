// frontend/admin/js/admin_payments.js
console.log("admin_payments.js loaded");

document.addEventListener('DOMContentLoaded', () => {
    const authToken = localStorage.getItem('authToken');
    const isAdmin = localStorage.getItem('isAdmin') === 'true';

    if (!isAdmin && !authToken) { // Check !isAdmin specifically
        alert("Access Denied. You are not authorized to view this page or your session has expired.");
        window.location.href = 'login.html'; 
        return;
    }
    if (!authToken) { // General token check if somehow isAdmin was true without token
        alert("Session expired. Please log in again.");
        window.location.href = 'login.html';
        return;
    }

    const paymentsTableBody = document.getElementById('paymentsTableBody');
    const prevPageButton = document.getElementById('prevPage');
    const nextPageButton = document.getElementById('nextPage');
    const pageInfo = document.getElementById('pageInfo');
    let currentPage = 1;
    const perPage = 15; // Or make this configurable

    async function fetchAdminPayments(page = 1) {
        if (!paymentsTableBody) return;
        paymentsTableBody.innerHTML = `<tr><td colspan="10" style="text-align:center;">Loading payments...</td></tr>`; // Increased colspan

        // Filters - get values from filter inputs if they exist
        const filterUserId = document.getElementById('filterUserId') ? document.getElementById('filterUserId').value : '';
        const filterStatus = document.getElementById('filterStatus') ? document.getElementById('filterStatus').value : '';
        const filterGateway = document.getElementById('filterGateway') ? document.getElementById('filterGateway').value : '';

        let queryParams = `page=${page}&per_page=${perPage}`;
        if (filterUserId) queryParams += `&user_id=${encodeURIComponent(filterUserId)}`;
        if (filterStatus) queryParams += `&status=${encodeURIComponent(filterStatus)}`;
        if (filterGateway) queryParams += `&gateway=${encodeURIComponent(filterGateway)}`;


        try {
            const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/payment/admin/transactions?${queryParams}`, { 
                headers: { 'Authorization': `Bearer ${authToken}` } 
            });
            if (!response.ok) {
                if (response.status === 401 || response.status === 403) { window.location.href = 'login.html'; return; }
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json(); 
            
            // The actual list of transactions is in data.transactions based on AdminPaymentListResponse (similar to UserPaymentHistoryResponse)
            const transactions = data.transactions; 
            const totalPages = data.total_pages;
            currentPage = data.page;


            if (data.status === "success" && transactions) {
                paymentsTableBody.innerHTML = '';
                if (transactions.length === 0) {
                    paymentsTableBody.innerHTML = '<tr><td colspan="10" style="text-align:center;">No payment transactions found for the current filters.</td></tr>';
                } else {
                    transactions.forEach(p => {
                        const row = paymentsTableBody.insertRow();
                        row.insertCell().textContent = p.id; 
                        row.insertCell().textContent = p.user_id;
                        row.insertCell().textContent = p.subscription_id || 'N/A'; // from user_strategy_subscription_id
                        row.insertCell().textContent = p.description || 'N/A';
                        row.insertCell().textContent = `${p.usd_equivalent ? p.usd_equivalent.toFixed(2) : (p.amount_crypto ? p.amount_crypto.toFixed(p.crypto_currency === "USDC" || p.crypto_currency === "USDT" ? 2 : 8) : 'N/A')}`;
                        row.insertCell().textContent = p.crypto_currency || (p.usd_equivalent ? 'USD (Equivalent)' : 'N/A');
                        row.insertCell().textContent = p.gateway || 'N/A';
                        row.insertCell().textContent = p.gateway_id || p.internal_reference || 'N/A'; // Show gateway_id or internal_ref
                        row.insertCell().innerHTML = `<span class="status-${(p.status || 'unknown').toLowerCase().replace(/\s+/g, '-')}">${p.status || 'Unknown'}</span>`;
                        row.insertCell().textContent = new Date(p.date).toLocaleString(); // 'date' is from created_at
                        
                        const actionsCell = row.insertCell();
                        const viewButton = document.createElement('button');
                        viewButton.className = 'btn btn-sm btn-outline';
                        viewButton.textContent = 'View';
                        viewButton.onclick = () => handleViewPayment(p.id); // Pass only ID
                        actionsCell.appendChild(viewButton);
                    });
                }
                updatePaginationControls(totalPages);
            } else {
                throw new Error(data.message || "Failed to parse payments list.");
            }
        } catch (error) {
            console.error("Error fetching admin payments:", error);
            paymentsTableBody.innerHTML = `<tr><td colspan="10" style="text-align:center;">Error loading payments: ${error.message}</td></tr>`;
            updatePaginationControls(0);
        }
    }

    function updatePaginationControls(totalPages) {
        if (pageInfo) pageInfo.textContent = `Page ${currentPage} of ${totalPages}`;
        if (prevPageButton) prevPageButton.disabled = currentPage <= 1;
        if (nextPageButton) nextPageButton.disabled = currentPage >= totalPages;
    }

    async function handleViewPayment(paymentId) {
        console.log(`Viewing details for payment ID: ${paymentId}`);
        try {
            const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/payment/admin/transactions/${paymentId}`, {
                 headers: { 'Authorization': `Bearer ${authToken}` } 
            });
            if (!response.ok) {
                if (response.status === 401 || response.status === 403) { window.location.href = 'login.html'; return; }
                const errData = await response.json().catch(()=>({}));
                throw new Error(errData.detail || errData.message || `HTTP error! status: ${response.status}`);
            }
            const result = await response.json();
            if (result.status === "success" && result.transaction) {
                // In a real app, display this in a modal or detailed view
                console.log("Payment Details:", result.transaction);
                alert(`Payment Details for Tx ID ${paymentId}:\nUser ID: ${result.transaction.user_id}\nAmount: ${result.transaction.usd_equivalent || result.transaction.amount_crypto} ${result.transaction.crypto_currency || 'USD'}\nStatus: ${result.transaction.status}\nDate: ${new Date(result.transaction.date).toLocaleString()}\nGateway: ${result.transaction.gateway}\nGateway ID: ${result.transaction.gateway_id}`);
            } else {
                throw new Error(result.message || "Could not fetch payment details.");
            }
        } catch (error) {
            console.error(`Error fetching details for payment ID ${paymentId}:`, error);
            alert(`Error fetching payment details: ${error.message}`);
        }
    }
    
    // Setup event listeners for filters and pagination
    const filterButton = document.getElementById('filterButton');
    if (filterButton) {
        filterButton.addEventListener('click', () => fetchAdminPayments(1));
    }
    if (prevPageButton) {
        prevPageButton.addEventListener('click', () => {
            if (currentPage > 1) fetchAdminPayments(currentPage - 1);
        });
    }
    if (nextPageButton) {
        nextPageButton.addEventListener('click', () => {
            fetchAdminPayments(currentPage + 1); // Backend handles max page
        });
    }

    fetchAdminPayments(); // Initial fetch
});
