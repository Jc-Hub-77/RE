// frontend/admin/js/admin_settings.js
console.log("admin_settings.js loaded");

document.addEventListener('DOMContentLoaded', () => {
    const authToken = localStorage.getItem('authToken');
    // const isAdmin = localStorage.getItem('isAdmin') === 'true';
    // if (!isAdmin || !authToken) { /* Redirect */ }

    const siteSettingsForm = document.getElementById('siteSettingsForm');
    
    const coinbaseApiKeyStatusElem = document.getElementById('coinbaseApiKeyStatus');
    const coinbaseWebhookSecretStatusElem = document.getElementById('coinbaseWebhookSecretStatus');
    const emailSmtpHostValueElem = document.getElementById('emailSmtpHostValue'); // To display current value

    const coinbaseApiKeyInput = document.getElementById('coinbaseApiKey');
    const coinbaseWebhookSecretInput = document.getElementById('coinbaseWebhookSecret');
    const emailSmtpHostInput = document.getElementById('emailSmtpHost');
    // Add more elements if new settings are added to admin_settings.html

    async function fetchSiteSettings() {
        console.log("Fetching site settings...");
        if (!coinbaseApiKeyStatusElem) { console.log("Settings page elements not found"); return; } // Ensure elements exist

        try {
            // Conceptual API: GET /api/admin/settings
            const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/admin/site-settings`, { 
                headers: { 'Authorization': `Bearer ${authToken}` } 
            });
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json(); // Expects { status, settings: {...} }
            
            if (data.status === "success" && data.settings) {
                const settings = data.settings;
                if (coinbaseApiKeyStatusElem) {
                    coinbaseApiKeyStatusElem.textContent = settings.COINBASE_COMMERCE_API_KEY_SET ? 'Set' : 'Not Set';
                    coinbaseApiKeyStatusElem.style.color = settings.COINBASE_COMMERCE_API_KEY_SET ? 'var(--success-color)' : 'var(--danger-color)';
                }
                if (coinbaseWebhookSecretStatusElem) {
                    coinbaseWebhookSecretStatusElem.textContent = settings.COINBASE_COMMERCE_WEBHOOK_SECRET_SET ? 'Set' : 'Not Set';
                    coinbaseWebhookSecretStatusElem.style.color = settings.COINBASE_COMMERCE_WEBHOOK_SECRET_SET ? 'var(--success-color)' : 'var(--danger-color)';
                }
                if (emailSmtpHostValueElem && emailSmtpHostInput) {
                    emailSmtpHostValueElem.textContent = settings.EMAIL_SMTP_HOST || 'Not Set';
                    emailSmtpHostInput.value = settings.EMAIL_SMTP_HOST || '';
                }
                // Example for a boolean setting like Maintenance Mode
                // const maintenanceModeToggle = document.getElementById('maintenanceModeToggle'); // Assuming you add this
                // if (maintenanceModeToggle) maintenanceModeToggle.checked = settings.MAINTENANCE_MODE;

            } else {
                throw new Error(data.message || "Failed to parse site settings.");
            }
        } catch (error) {
            console.error("Failed to load site settings:", error);
            if (coinbaseApiKeyStatusElem) coinbaseApiKeyStatusElem.textContent = "Error loading";
            // Update other elements to show error state
        }
    }

    if (siteSettingsForm) {
        siteSettingsForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            const settingsToUpdate = []; // Array of {key, value} pairs
            
            // Only include settings if a new value is provided (especially for sensitive ones)
            if (coinbaseApiKeyInput.value) settingsToUpdate.push({key: "COINBASE_COMMERCE_API_KEY", value: coinbaseApiKeyInput.value});
            if (coinbaseWebhookSecretInput.value) settingsToUpdate.push({key: "COINBASE_COMMERCE_WEBHOOK_SECRET", value: coinbaseWebhookSecretInput.value});
            
            // For non-sensitive, always send current value or new value
            if (emailSmtpHostInput) settingsToUpdate.push({key: "EMAIL_SMTP_HOST", value: emailSmtpHostInput.value});
            
            // Example for a boolean toggle
            // const maintenanceModeToggle = document.getElementById('maintenanceModeToggle');
            // if (maintenanceModeToggle) settingsToUpdate.push({key: "MAINTENANCE_MODE", value: maintenanceModeToggle.checked});


            if (settingsToUpdate.length === 0) {
                alert("No changes to save.");
                return;
            }

            if (!confirm("Are you sure you want to update these site settings? Some changes may require a server restart or can impact site functionality if misconfigured.")) {
                return;
            }

            console.log("Updating site settings:", settingsToUpdate);
            let allUpdatesSuccessful = true;

            try {
                // Conceptual API: POST /api/admin/settings (can send multiple updates or one by one)
                // For simplicity, let's assume a single endpoint that takes a list of updates
                // For this conceptual example, we don't have a backend endpoint to update these settings directly via a generic API.
                // The instruction was to make BACKEND_API_BASE_URL configurable for *fetching*.
                // Actual update logic for these specific settings would need dedicated backend endpoints,
                // which are not part of this task. This part remains simulated.
                
                // const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/admin/site-settings`, { 
                //     method: 'POST', 
                //     headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` },
                //     body: JSON.stringify({settings: settingsToUpdate}) // Send as a list of key-value pairs
                // });
                // if (!response.ok) {
                //     const errData = await response.json().catch(() => ({message: "Unknown error updating settings"}));
                //     throw new Error(errData.message || `HTTP error! status: ${response.status}`);
                // }
                // const result = await response.json();

                await new Promise(resolve => setTimeout(resolve, 700)); // Simulate delay
                const result = {status: "success", message: "Site settings updated successfully (simulated). Some changes may require server restart."};

                if (result.status === "success") {
                    alert(result.message);
                } else {
                    allUpdatesSuccessful = false;
                    throw new Error(result.message || "Failed to update one or more settings.");
                }
            } catch (error) {
                allUpdatesSuccessful = false;
                console.error("Error updating site settings:", error);
                alert("Error updating settings: " + error.message);
            } finally {
                // Clear sensitive input fields after attempt
                if (coinbaseApiKeyInput) coinbaseApiKeyInput.value = ''; 
                if (coinbaseWebhookSecretInput) coinbaseWebhookSecretInput.value = '';
                if (allUpdatesSuccessful) fetchSiteSettings(); // Re-fetch to show updated status
            }
        });
    }

    fetchSiteSettings();
});
