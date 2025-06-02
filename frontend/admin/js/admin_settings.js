// frontend/admin/js/admin_settings.js
console.log("admin_settings.js loaded");

document.addEventListener('DOMContentLoaded', () => {
    const authToken = localStorage.getItem('authToken');
    // const isAdmin = localStorage.getItem('isAdmin') === 'true';
    // if (!isAdmin || !authToken) { /* Redirect */ }

    const systemSettingsForm = document.getElementById('systemSettingsForm'); // Assuming form ID is systemSettingsForm
    const settingsContainer = document.getElementById('systemSettingsContainer'); // Container to dynamically add inputs

    // Store initially fetched settings to compare on save
    let currentSystemSettings = {};

    async function fetchSystemSettings() {
        console.log("Fetching system settings from DB...");
        if (!settingsContainer) {
            console.error("System settings container not found.");
            return;
        }
        settingsContainer.innerHTML = '<p>Loading settings...</p>';

        try {
            const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/admin/system-settings`, {
                headers: { 'Authorization': `Bearer ${authToken}` }
            });
            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || `HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            
            if (data.status === "success" && data.system_settings) {
                settingsContainer.innerHTML = ''; // Clear loading
                currentSystemSettings = {}; // Reset
                data.system_settings.forEach(setting => {
                    currentSystemSettings[setting.key] = setting; // Store full setting object

                    const formGroup = document.createElement('div');
                    formGroup.className = 'form-group';

                    const label = document.createElement('label');
                    label.setAttribute('for', `settingInput_${setting.key}`);
                    label.textContent = `${setting.key} (${setting.description || 'No description'}):`;
                    
                    const input = document.createElement('input');
                    input.type = 'text'; // Keep as text for now, specific types can be handled on backend or with more complex FE
                    input.className = 'form-control'; // Assuming bootstrap or similar styling
                    input.id = `settingInput_${setting.key}`;
                    input.name = setting.key;
                    input.value = setting.value;
                    
                    formGroup.appendChild(label);
                    formGroup.appendChild(input);
                    settingsContainer.appendChild(formGroup);
                });
            } else {
                throw new Error(data.message || "Failed to parse system settings.");
            }
        } catch (error) {
            console.error("Failed to load system settings:", error);
            settingsContainer.innerHTML = `<p class="error-message">Error loading system settings: ${error.message}</p>`;
        }
    }

    if (systemSettingsForm) {
        systemSettingsForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            
            const settingsToUpdate = [];
            const inputs = settingsContainer.querySelectorAll('input[type="text"]');

            inputs.forEach(input => {
                const key = input.name;
                const newValue = input.value;
                // Check if value has changed from originally fetched value
                if (currentSystemSettings[key] && currentSystemSettings[key].value !== newValue) {
                    settingsToUpdate.push({ 
                        key: key, 
                        value: newValue,
                        // Description could also be made editable if desired
                        description: currentSystemSettings[key].description 
                    });
                }
            });

            if (settingsToUpdate.length === 0) {
                alert("No changes detected in system settings.");
                return;
            }

            if (!confirm(`Are you sure you want to update ${settingsToUpdate.length} system setting(s)?`)) {
                return;
            }

            let updateSuccessCount = 0;
            let updateErrorCount = 0;

            for (const setting of settingsToUpdate) {
                console.log(`Updating system setting: ${setting.key} to ${setting.value}`);
                try {
                    const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/admin/system-settings/${setting.key}`, {
                        method: 'PUT',
                        headers: { 
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${authToken}` 
                        },
                        body: JSON.stringify({ value: setting.value, description: setting.description })
                    });
                    const result = await response.json();
                    if (!response.ok || result.status !== "success") {
                        updateErrorCount++;
                        console.error(`Failed to update setting ${setting.key}:`, result.message || result.detail || `HTTP ${response.status}`);
                        alert(`Failed to update setting ${setting.key}: ${result.message || result.detail || 'Unknown error'}`);
                    } else {
                        updateSuccessCount++;
                    }
                } catch (error) {
                    updateErrorCount++;
                    console.error(`Error updating setting ${setting.key}:`, error);
                    alert(`Error updating setting ${setting.key}: ${error.message}`);
                }
            }

            alert(`Settings update process finished. Successful: ${updateSuccessCount}, Failed: ${updateErrorCount}.`);
            fetchSystemSettings(); // Re-fetch to show updated values and clear changed state
        });
    }

    // Initial fetch of settings
    fetchSystemSettings();
});
