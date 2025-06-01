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
                    editButton.onclick = () => handleEditStrategyModal(strategy); // Pass full strategy object
                    actionsCell.appendChild(editButton);
                    // TODO: Add delete/disable buttons if strategies are DB managed and API supports it
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
});
