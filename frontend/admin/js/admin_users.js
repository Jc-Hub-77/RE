// frontend/admin/js/admin_users.js
console.log("admin_users.js loaded");

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

    const usersTableBody = document.getElementById('usersTableBody');
    const searchUserInput = document.getElementById('searchUserInput'); // Assumed ID for search input
    const searchUserButton = document.getElementById('searchUserButton'); // Assumed ID for search button
    const prevPageButtonUsers = document.getElementById('prevPageUsers'); // Assumed ID
    const nextPageButtonUsers = document.getElementById('nextPageUsers'); // Assumed ID
    const pageInfoUsers = document.getElementById('pageInfoUsers'); // Assumed ID

    let currentUsersPage = 1;
    const usersPerPage = 15; 

    async function fetchUsers(page = 1, searchTerm = '') {
        if (!usersTableBody) return;
        usersTableBody.innerHTML = `<tr><td colspan="7" style="text-align:center;">Loading users...</td></tr>`; // Colspan updated to 7

        currentUsersPage = page;
        let queryParams = `page=${page}&per_page=${usersPerPage}`;
        if (searchTerm) {
            queryParams += `&search_term=${encodeURIComponent(searchTerm)}`;
        }
        // Add sorting params if HTML controls are added: &sort_by=...&sort_order=...

        try {
            const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/admin/users?${queryParams}`, { 
                headers: { 'Authorization': `Bearer ${authToken}` }
            });
            if (!response.ok) {
                if (response.status === 401 || response.status === 403) { window.location.href = 'login.html'; return; }
                const errData = await response.json().catch(()=>({}));
                throw new Error(errData.detail || errData.message || `HTTP error! status: ${response.status}`);
            }
            const data = await response.json(); 
            
            const users = data.users;
            const totalPages = data.total_pages;
            currentUsersPage = data.page;

            if (data.status === "success" && users) {
                usersTableBody.innerHTML = ''; 
                if (users.length === 0) {
                    usersTableBody.innerHTML = '<tr><td colspan="7" style="text-align:center;">No users found.</td></tr>';
                } else {
                    users.forEach(user => {
                        const row = usersTableBody.insertRow();
                        row.insertCell().textContent = user.id;
                        row.insertCell().textContent = user.username;
                        row.insertCell().textContent = user.email;
                        row.insertCell().innerHTML = user.is_admin ? '<span class="status-active" style="font-weight:bold;">Yes</span>' : 'No';
                        row.insertCell().innerHTML = user.email_verified ? '<span class="status-active">Yes</span>' : '<span class="status-inactive">No</span>';
                        row.insertCell().textContent = new Date(user.created_at).toLocaleDateString();
                        
                        const actionsCell = row.insertCell();
                        
                        const toggleAdminButton = document.createElement('button');
                        toggleAdminButton.className = 'btn btn-sm';
                        toggleAdminButton.textContent = user.is_admin ? 'Revoke Admin' : 'Make Admin';
                        toggleAdminButton.classList.add(user.is_admin ? 'btn-warning' : 'btn-success');
                        toggleAdminButton.style.marginRight = '5px';
                        toggleAdminButton.onclick = () => handleToggleAdmin(user.id, !user.is_admin);
                        actionsCell.appendChild(toggleAdminButton);

                        const toggleActiveButton = document.createElement('button');
                        toggleActiveButton.className = 'btn btn-sm';
                        toggleActiveButton.textContent = user.is_active ? 'Deactivate' : 'Activate';
                        toggleActiveButton.classList.add(user.is_active ? 'btn-danger' : 'btn-success');
                        toggleActiveButton.style.marginRight = '5px';
                        toggleActiveButton.onclick = () => handleToggleActive(user.id, !user.is_active);
                        actionsCell.appendChild(toggleActiveButton);
                        
                        // Email verification toggle - Assuming backend has admin_service.toggle_user_email_verified
                        const toggleEmailVerifiedButton = document.createElement('button');
                        toggleEmailVerifiedButton.className = 'btn btn-sm btn-outline';
                        toggleEmailVerifiedButton.textContent = user.email_verified ? 'Mark Unverified' : 'Mark Verified';
                        toggleEmailVerifiedButton.onclick = () => handleToggleEmailVerified(user.id, !user.email_verified);
                        actionsCell.appendChild(toggleEmailVerifiedButton);

                    });
                }
                updateUserPaginationControls(totalPages);
            } else {
                throw new Error(data.message || "Failed to parse user list.");
            }
        } catch (error) {
            console.error("Error fetching users:", error);
            usersTableBody.innerHTML = `<tr><td colspan="7" style="text-align:center;">Error loading users: ${error.message}</td></tr>`;
            updateUserPaginationControls(0);
        }
    }

    function updateUserPaginationControls(totalPages) {
        if (pageInfoUsers) pageInfoUsers.textContent = `Page ${currentUsersPage} of ${totalPages || 0}`;
        if (prevPageButtonUsers) prevPageButtonUsers.disabled = currentUsersPage <= 1;
        if (nextPageButtonUsers) nextPageButtonUsers.disabled = currentUsersPage >= totalPages;
    }

    async function handleToggleAdmin(userId, makeAdmin) {
        if (!confirm(`Are you sure you want to ${makeAdmin ? 'grant admin rights to' : 'revoke admin rights from'} user ID ${userId}?`)) return;
        
        try {
            const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/admin/users/set-admin-status`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` },
                body: JSON.stringify({ user_id: userId, make_admin: makeAdmin })
            });
            const result = await response.json();
            if (!response.ok) {
                const errorDetail = result.detail || result.message || `HTTP error! status: ${response.status}`;
                throw new Error(errorDetail);
            }
            alert(result.message || "Admin status updated.");
            fetchUsers(currentUsersPage, searchUserInput ? searchUserInput.value : ''); 
        } catch (error) {
            console.error("Error toggling admin status:", error);
            alert("Error: " + error.message);
        }
    }

    async function handleToggleActive(userId, setActive) {
        if (!confirm(`Are you sure you want to ${setActive ? 'activate' : 'deactivate'} user ID ${userId}?`)) return;
        try {
            // Note: Backend uses 'make_admin' field in AdminSetAdminStatusRequest for the 'activate' boolean
            const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/admin/users/toggle-active-status`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` },
                body: JSON.stringify({ user_id: userId, make_admin: setActive }) 
            });
            const result = await response.json();
            if (!response.ok) {
                const errorDetail = result.detail || result.message || `HTTP error! status: ${response.status}`;
                throw new Error(errorDetail);
            }
            alert(result.message || "User active status updated.");
            fetchUsers(currentUsersPage, searchUserInput ? searchUserInput.value : '');
        } catch (error) {
            console.error("Error toggling active status:", error);
            alert("Error: " + error.message);
        }
    }
    
    async function handleToggleEmailVerified(userId, setEmailVerified) {
        // This function assumes the backend /admin/users/toggle-email-verified endpoint
        // uses the same AdminSetAdminStatusRequest schema, where 'make_admin' field is used for 'set_email_verified'.
        // The backend router for this needs to be checked or adjusted if it expects a different schema.
        // For now, assuming it mirrors toggle-active-status's re-use of the schema.
        if (!confirm(`Are you sure you want to mark email as ${setEmailVerified ? 'VERIFIED' : 'NOT VERIFIED'} for user ID ${userId}?`)) return;
        try {
            const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/admin/users/toggle-email-verified`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` },
                // Assuming backend reuses 'make_admin' field for the boolean value
                body: JSON.stringify({ user_id: userId, make_admin: setEmailVerified }) 
            });
            const result = await response.json();
            if (!response.ok) {
                const errorDetail = result.detail || result.message || `HTTP error! status: ${response.status}`;
                throw new Error(errorDetail);
            }
            alert(result.message || "Email verification status updated.");
            fetchUsers(currentUsersPage, searchUserInput ? searchUserInput.value : '');
        } catch (error) {
            console.error("Error toggling email verified status:", error);
            alert("Error: " + error.message);
        }
    }
    
    if (searchUserButton && searchUserInput) {
        searchUserButton.addEventListener('click', () => fetchUsers(1, searchUserInput.value));
    }
    if (prevPageButtonUsers) {
        prevPageButtonUsers.addEventListener('click', () => {
            if (currentUsersPage > 1) fetchUsers(currentUsersPage - 1, searchUserInput ? searchUserInput.value : '');
        });
    }
    if (nextPageButtonUsers) {
        nextPageButtonUsers.addEventListener('click', () => {
            fetchUsers(currentUsersPage + 1, searchUserInput ? searchUserInput.value : '');
        });
    }

    if (authToken && isAdmin) {
        fetchUsers(); // Initial fetch
    } else {
        if (usersTableBody) usersTableBody.innerHTML = '<tr><td colspan="7" style="text-align:center;">Access Denied. Please login as admin.</td></tr>';
    }
});
