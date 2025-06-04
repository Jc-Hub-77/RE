// frontend/admin/js/admin_strategies.js
console.log("admin_strategies.js loaded");

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

    const strategiesTableBody = document.getElementById('strategiesTableBody');
    const addNewStrategyBtn = document.querySelector('header.page-header button'); 

    async function fetchAdminStrategies() {
        if (!strategiesTableBody) return;
        strategiesTableBody.innerHTML = '<tr><td colspan="7" style="text-align:center;">Loading strategies...</td></tr>'; // Updated colspan to 7

        try {
            const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/admin/strategies`, { 
                headers: { 'Authorization': `Bearer ${authToken}` } 
            });
            if (!response.ok) {
                if (response.status === 401 || response.status === 403) { window.location.href = 'login.html'; return; }
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json(); 
            
            if (data.status === "success" && data.strategies) {
                strategiesTableBody.innerHTML = '';
                if (data.strategies.length === 0) {
                    strategiesTableBody.innerHTML = '<tr><td colspan="7" style="text-align:center;">No strategies defined.</td></tr>';
                    return;
                }
                data.strategies.forEach(strategy => {
                    const row = strategiesTableBody.insertRow();
                    row.insertCell().textContent = strategy.id;
                    row.insertCell().textContent = strategy.name;
                    row.insertCell().textContent = strategy.category || 'N/A';
                    row.insertCell().textContent = strategy.risk_level || 'N/A';
                    row.insertCell().textContent = strategy.python_code_path || 'N/A';
                    row.insertCell().innerHTML = `<span class="status-${strategy.is_active ? 'active' : 'inactive'}">${strategy.is_active ? 'Active' : 'Inactive'}</span>`;
                    
                    const actionsCell = row.insertCell();
                    const editButton = document.createElement('button');
                    editButton.className = 'btn btn-sm btn-outline';
                    editButton.textContent = 'Edit';
                    editButton.onclick = () => handleEditStrategyModal(strategy); 
                    actionsCell.appendChild(editButton);

                    const toggleActiveButton = document.createElement('button');
                    toggleActiveButton.className = `btn btn-sm ${strategy.is_active ? 'btn-warning' : 'btn-success'}`;
                    toggleActiveButton.textContent = strategy.is_active ? 'Disable' : 'Enable';
                    toggleActiveButton.style.marginLeft = '5px';
                    toggleActiveButton.onclick = () => handleToggleStrategyStatus(strategy.id, strategy.is_active);
                    actionsCell.appendChild(toggleActiveButton);

                    const deleteButton = document.createElement('button');
                    deleteButton.className = 'btn btn-sm btn-danger';
                    deleteButton.textContent = 'Delete';
                    deleteButton.style.marginLeft = '5px';
                    deleteButton.onclick = () => handleDeleteStrategy(strategy.id, strategy.name);
                    actionsCell.appendChild(deleteButton);
                });
            } else {
                throw new Error(data.message || "Failed to parse strategies list.");
            }
        } catch (error) {
            console.error("Error fetching admin strategies:", error);
            strategiesTableBody.innerHTML = `<tr><td colspan="7" style="text-align:center;">Error loading strategies: ${error.message}</td></tr>`;
        }
    }

    function handleEditStrategyModal(strategyData) {
        // This function would populate and show a modal.
        // For now, we'll use prompts for a simplified edit.
        console.log("Editing strategy:", strategyData);
        
        const newName = prompt("Enter new name (or leave blank to keep current):", strategyData.name);
        const newDescription = prompt("Enter new description (or leave blank):", strategyData.description);
        const newPythonCodePath = prompt("Enter new Python Code Path (or leave blank):", strategyData.python_code_path);
        const newDefaultParamsStr = prompt("Enter new Default Parameters (JSON string, or leave blank):", strategyData.default_parameters); // Expects JSON string
        const newCategory = prompt("Enter new category (or leave blank):", strategyData.category);
        const newRiskLevel = prompt("Enter new risk level (or leave blank):", strategyData.risk_level);
        const isActiveStr = prompt("Set active? (true/false, or leave blank):", String(strategyData.is_active));

        const updates = {};
        if (newName !== null && newName !== strategyData.name) updates.name = newName;
        if (newDescription !== null && newDescription !== strategyData.description) updates.description = newDescription;
        if (newPythonCodePath !== null && newPythonCodePath !== strategyData.python_code_path) updates.python_code_path = newPythonCodePath;
        if (newDefaultParamsStr !== null && newDefaultParamsStr !== strategyData.default_parameters) {
            try { JSON.parse(newDefaultParamsStr); updates.default_parameters = newDefaultParamsStr; } // Validate JSON
            catch (e) { alert("Invalid JSON for default parameters. Not updating parameters."); }
        }
        if (newCategory !== null && newCategory !== strategyData.category) updates.category = newCategory;
        if (newRiskLevel !== null && newRiskLevel !== strategyData.risk_level) updates.risk_level = newRiskLevel;
        if (isActiveStr !== null && isActiveStr !== String(strategyData.is_active)) {
            updates.is_active = isActiveStr.toLowerCase() === 'true';
        }
        
        if (Object.keys(updates).length > 0) {
            handleUpdateStrategy(strategyData.id, updates);
        } else {
            alert("No changes made.");
        }
    }
    
    async function handleUpdateStrategy(strategyId, updatedData) {
        console.log(`Updating strategy ID ${strategyId} with:`, updatedData);
        try {
           const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/admin/strategies/${strategyId}`, {
               method: 'PUT',
               headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` },
               body: JSON.stringify(updatedData)
           });
           const result = await response.json();
           if (!response.ok) {
                const errorDetail = result.detail || result.message || `HTTP error! status: ${response.status}`;
                if (Array.isArray(errorDetail)) { throw new Error(errorDetail.map(e => `${e.loc.join('->')}: ${e.msg}`).join(', ')); }
                throw new Error(errorDetail);
           }
           
           alert(result.message || "Strategy updated successfully.");
           fetchAdminStrategies(); 
        } catch (error) { 
            console.error("Error updating strategy:", error);
            alert("Error updating strategy: " + error.message);
        }
    }

    if (addNewStrategyBtn) {
        addNewStrategyBtn.addEventListener('click', async () => { 
            // Simplified: use prompts for new strategy data
            const name = prompt("Enter Strategy Name:");
            if (!name) return;
            const description = prompt("Enter Description:") || "";
            const python_code_path = prompt("Enter Python Code Path (e.g., my_strategy.py):");
            if (!python_code_path) return;
            const default_parameters_str = prompt("Enter Default Parameters (JSON string, e.g., {\"period\": 20}):", "{}");
            const category = prompt("Enter Category (e.g., Trend, Oscillator):") || "N/A";
            const risk_level = prompt("Enter Risk Level (e.g., Low, Medium, High):") || "N/A";

            try {
                JSON.parse(default_parameters_str); // Validate JSON
            } catch (e) {
                alert("Invalid JSON for default parameters. Strategy not added.");
                return;
            }
            
            const newStrategyData = { 
                name, description, python_code_path, 
                default_parameters: default_parameters_str, // Send as string
                category, risk_level,
                is_active: true // New strategies are active by default
            };
            
            console.log("Adding new strategy:", newStrategyData);
            try {
               const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/admin/strategies`, {
                   method: 'POST',
                   headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` },
                   body: JSON.stringify(newStrategyData)
               });
               const result = await response.json();
               if (!response.ok) { // Check for non-2xx status codes
                    const errorDetail = result.detail || result.message || `HTTP error! status: ${response.status}`;
                    if (Array.isArray(errorDetail)) { throw new Error(errorDetail.map(e => `${e.loc.join('->')}: ${e.msg}`).join(', ')); }
                    throw new Error(errorDetail);
               }
               
               alert(result.message || "Strategy added successfully.");
               fetchAdminStrategies(); 
            } catch (error) { 
                console.error("Error adding new strategy:", error);
                alert("Error adding strategy: " + error.message);
            }
        });
    }
    
    if (authToken) { // Only fetch if authenticated
        fetchAdminStrategies();
    } else {
        if (strategiesTableBody) strategiesTableBody.innerHTML = '<tr><td colspan="7" style="text-align:center;">Please login as admin.</td></tr>';
    }

    // --- Modal and New/Edit Logic Description (as per subtask) ---
    // To replace prompt()-based editing/adding:
    // 1. HTML Modal:
    //    - A hidden modal element would be added to admin_strategies.html.
    //    - It would contain a form with input fields for: Name, Description, Python Code Path,
    //      Default Parameters (textarea), Category, Risk Level, Payment Options JSON (textarea), and an Is Active checkbox.
    //    - A hidden input for 'strategyId' would be used for editing.
    //    - "Save" and "Cancel" buttons.
    // 2. openStrategyModal(strategyDataOrNull):
    //    - If strategyDataOrNull is provided (editing), populate form fields with this data and set strategyId.
    //    - If null (adding), clear form fields and strategyId.
    //    - Display the modal.
    // 3. Modal "Save" Button Listener:
    //    - Collects data from form fields.
    //    - Performs basic frontend validation (e.g., required fields, JSON validity for params/payment_options).
    //    - If strategyId exists, calls handleUpdateStrategy(strategyId, collectedData).
    //    - Else, calls a new handleCreateStrategy(collectedData) function (which would adapt current addNewStrategyBtn logic).
    //    - Handles API response and closes modal or shows errors.
    // 4. Modal "Cancel" Button: Hides the modal.
    // The existing handleUpdateStrategy and the logic inside addNewStrategyBtn's listener would be adapted.

    async function handleToggleStrategyStatus(strategyId, currentIsActive) {
        const newStatus = !currentIsActive;
        if (!confirm(`Are you sure you want to ${newStatus ? 'enable' : 'disable'} strategy ID ${strategyId}?`)) return;

        try {
            const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/admin/strategies/${strategyId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` },
                body: JSON.stringify({ is_active: newStatus }) // Assumes backend endpoint handles partial update for is_active
            });
            const result = await response.json();
            if (!response.ok) {
                throw new Error(result.detail || result.message || "Failed to toggle strategy status.");
            }
            alert(result.message || `Strategy status updated to ${newStatus ? 'Active' : 'Inactive'}.`);
            fetchAdminStrategies(); // Refresh the list
        } catch (error) {
            console.error(`Error toggling strategy ${strategyId} status:`, error);
            alert(`Error: ${error.message}`);
        }
    }

    async function handleDeleteStrategy(strategyId, strategyName) {
        if (!confirm(`Are you sure you want to DELETE strategy ID ${strategyId} (${strategyName})? This action cannot be undone.`)) return;
        
        // Placeholder for backend API call, as DELETE endpoint might not exist yet.
        // console.warn(`Attempting to delete strategy ID ${strategyId}. Backend DELETE endpoint needs to be implemented.`);
        // alert(`Simulated delete for strategy ID ${strategyId}. Implement backend DELETE /api/v1/admin/strategies/${strategyId}`);
        
        // Actual call to the implemented backend endpoint:
        try {
            const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/admin/managed-strategies/${strategyId}`, {
                method: 'DELETE',
                headers: { 'Authorization': `Bearer ${authToken}` }
            });
            if (!response.ok) {
                const result = await response.json().catch(() => ({})); // Try to get error message
                throw new Error(result.detail || result.message || `Failed to delete strategy (HTTP ${response.status})`);
            }
            // If response is 204 No Content, there might not be a JSON body
            if (response.status === 204) {
                 alert(`Strategy ID ${strategyId} (${strategyName}) deleted successfully.`);
            } else {
                const result = await response.json();
                alert(result.message || `Strategy ID ${strategyId} (${strategyName}) deleted successfully.`);
            }
            fetchAdminStrategies(); // Refresh the list
        } catch (error) {
            console.error(`Error deleting strategy ${strategyId}:`, error);
            alert(`Error: ${error.message}`);
        }
    }
});
