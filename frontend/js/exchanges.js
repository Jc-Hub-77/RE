// frontend/js/exchanges.js
console.log("exchanges.js loaded");

// const BACKEND_API_BASE_URL = 'http://127.0.0.1:8000'; // Ensure this is correct - This will now be set globally via HTML script tag

document.addEventListener('DOMContentLoaded', () => {
    const authToken = localStorage.getItem('authToken');
    const userId = localStorage.getItem('userId'); // userId from localStorage might not be needed if backend uses token

    // Optional: Implement a more robust check or rely on individual fetch calls to handle 401
    if (!authToken) {
        console.warn("User not authenticated. Some functionality may be limited or redirect.");
        // window.location.href = 'login.html'; // Consider global auth check
        // return;
    }

    const addExchangeForm = document.getElementById('addExchangeForm');
    const exchangeNameSelect = document.getElementById('exchangeName');
    const connectedExchangesTableBody = document.getElementById('connectedExchangesTableBody');

    async function initializeExchangesPage() {
        await populateExchangeDropdown();
        await fetchAndDisplayConnectedExchanges();
    }

    async function populateExchangeDropdown() {
        if (!exchangeNameSelect) return;
        exchangeNameSelect.innerHTML = '<option value="">Loading exchanges...</option>';
        
        try {
            const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/exchange/supported-exchanges`, { // Corrected Endpoint
                headers: { 'Authorization': `Bearer ${authToken}` }
            });
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const supportedExchanges = await response.json(); // List of strings

            exchangeNameSelect.innerHTML = '<option value="">Select an Exchange</option>';
            supportedExchanges.sort().forEach(exchangeId => {
                const option = document.createElement('option');
                option.value = exchangeId.toLowerCase();
                option.textContent = exchangeId.charAt(0).toUpperCase() + exchangeId.slice(1).replace(/([A-Z])/g, ' $1').trim();
                exchangeNameSelect.appendChild(option);
            });
        } catch (error) {
            console.error("Failed to load supported exchanges:", error);
            exchangeNameSelect.innerHTML = '<option value="">Error loading exchanges</option>';
        }
    }

    async function fetchAndDisplayConnectedExchanges() {
        if (!connectedExchangesTableBody) return;
        connectedExchangesTableBody.innerHTML = '<tr><td colspan="5" style="text-align:center;">Fetching your connected exchanges...</td></tr>';

        try {
            const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/exchange/api-keys`, { // Corrected Endpoint
                headers: { 'Authorization': `Bearer ${authToken}` }
            });
            if (!response.ok) {
                if (response.status === 401) { window.location.href = 'login.html'; return; }
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json(); 
            
            if (data.status === "success" && data.keys) {
                connectedExchangesTableBody.innerHTML = ''; 
                if (data.keys.length === 0) {
                    connectedExchangesTableBody.innerHTML = '<tr><td colspan="5" style="text-align:center;">No API keys added yet.</td></tr>';
                    return;
                }
                data.keys.forEach(key => {
                    const row = connectedExchangesTableBody.insertRow();
                    row.insertCell().textContent = key.label;
                    row.insertCell().textContent = key.exchange_name; // Assuming backend sends it capitalized or we do it here
                    row.insertCell().textContent = key.api_key_preview;
                    
                    const statusCell = row.insertCell();
                    statusCell.textContent = (key.status || 'unknown').replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                    statusCell.className = `status-${(key.status || 'unknown').replace(/_/g, '-')}`;

                    const actionsCell = row.insertCell();
                    const testButton = document.createElement('button');
                    testButton.textContent = "Test";
                    testButton.className = "btn btn-sm btn-outline test-key-btn";
                    testButton.dataset.keyId = key.id;
                    testButton.disabled = key.status === "error_decryption" || key.status === "testing";

                    const removeButton = document.createElement('button');
                    removeButton.textContent = "Remove";
                    removeButton.className = "btn btn-sm btn-danger remove-key-btn";
                    removeButton.dataset.keyId = key.id;
                    removeButton.style.marginLeft = "5px";

                    actionsCell.appendChild(testButton);
                    actionsCell.appendChild(removeButton);
                });
            } else {
                throw new Error(data.message || "Failed to parse exchange keys.");
            }
        } catch (error) {
            console.error("Failed to fetch connected exchanges:", error);
            connectedExchangesTableBody.innerHTML = `<tr><td colspan="5" style="text-align:center;">Error loading exchanges: ${error.message}</td></tr>`;
        }
    }

    if (addExchangeForm) {
        addExchangeForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            const formData = new FormData(addExchangeForm);
            const exchangeData = {
                exchange_name: formData.get('exchangeName'),
                label: formData.get('apiKeyLabel'),
                api_key_public: formData.get('apiKey'),
                secret_key: formData.get('secretKey'),
                passphrase: formData.get('passphrase') || null 
            };

            console.log("Submitting new API key:", exchangeData);
            try {
                const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/exchange/api-keys`, { // Corrected Endpoint
                    method: 'POST',
                    headers: { 
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${authToken}` 
                    },
                    body: JSON.stringify(exchangeData)
                });
                const result = await response.json();
                if (response.status === 201 && result.status === "success") {
                    alert(result.message || "API Key added successfully. Please test it.");
                    addExchangeForm.reset();
                    fetchAndDisplayConnectedExchanges(); 
                } else {
                    const errorMessage = result.detail || result.message || `Failed to add API key (HTTP ${response.status})`;
                    if (Array.isArray(errorMessage)) { 
                        alert("Validation Error: " + errorMessage.map(err => `${err.loc.join('->')}: ${err.msg}`).join("; "));
                    } else {
                        alert("Failed to add API key: " + errorMessage);
                    }
                    if (!response.ok) throw new Error(errorMessage);
                }
            } catch (error) {
                console.error("Error adding API key:", error);
                alert("Error adding API key: " + error.message);
            }
        });
    }

    async function handleTestKey(apiKeyId, buttonTextContext = "Test") {
        const testButton = document.querySelector(`.test-key-btn[data-key-id="${apiKeyId}"]`);
        if(testButton) { testButton.disabled = true; testButton.textContent = "Testing...";}

        console.log(`Testing API Key ID: ${apiKeyId}`);
        try {
            const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/exchange/api-keys/${apiKeyId}/test-connectivity`, { // Corrected Endpoint
                method: 'POST',
                headers: { 'Authorization': `Bearer ${authToken}` }
            });
            const result = await response.json();
            
            alert(`Test for key ${apiKeyId}: ${result.message || 'Test completed.'} (Status: ${result.status || 'unknown'})`);
            
            if (result.status && result.status.startsWith("error_")) {
                 console.warn(`API Key Test Error for ${apiKeyId}: ${result.message} (Status: ${result.status})`);
            }
        } catch (error) {
            console.error(`Error testing API key ${apiKeyId}:`, error);
            alert(`Error testing API key: ${error.message || "Network or server error."}`);
        } finally {
            if(testButton) { testButton.disabled = false; testButton.textContent = buttonTextContext; }
            fetchAndDisplayConnectedExchanges(); 
        }
    }

    async function handleRemoveKey(apiKeyId) {
        if (!confirm(`Are you sure you want to remove API Key ID: ${apiKeyId}? This action cannot be undone.`)) {
            return;
        }
        console.log(`Removing API Key ID: ${apiKeyId}`);
        try {
            const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/exchange/api-keys/${apiKeyId}`, { // Corrected Endpoint
                method: 'DELETE',
                headers: { 'Authorization': `Bearer ${authToken}` }
            });
            const result = await response.json(); 

            if (response.ok && result.status === "success") {
                alert(result.message || "API key removed successfully.");
            } else {
                const errorMessage = result.detail || result.message || `Failed to remove API key (HTTP ${response.status})`;
                alert(errorMessage);
                if (!response.ok) throw new Error(errorMessage); // Throw to be caught by outer catch if needed
            }
        } catch (error) {
            console.error(`Error removing API key ${apiKeyId}:`, error);
            alert(`Error removing API key: ${error.message || "Network or server error."}`);
        } finally {
            fetchAndDisplayConnectedExchanges(); 
        }
    }

    if (connectedExchangesTableBody) {
        connectedExchangesTableBody.addEventListener('click', (event) => {
            const targetButton = event.target.closest('button');
            if (!targetButton) return;

            const keyId = targetButton.dataset.keyId;
            if (targetButton.classList.contains('test-key-btn')) {
                handleTestKey(keyId);
            } else if (targetButton.classList.contains('remove-key-btn')) {
                handleRemoveKey(keyId);
            }
        });
    }
    
    if (authToken && userId) { // Only initialize if seemingly authenticated
        initializeExchangesPage();
    } else {
        console.warn("User not authenticated. Exchanges page will not initialize fully.");
        if (connectedExchangesTableBody) connectedExchangesTableBody.innerHTML = '<tr><td colspan="5" style="text-align:center;">Please login to manage exchanges.</td></tr>';
        if (exchangeNameSelect) exchangeNameSelect.innerHTML = '<option value="">Please login</option>';
    }
});
