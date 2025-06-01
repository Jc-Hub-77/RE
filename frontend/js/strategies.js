// frontend/js/strategies.js
console.log("strategies.js loaded");

// const BACKEND_API_BASE_URL = 'http://127.0.0.1:8000'; // This will now be set globally via HTML script tag

document.addEventListener('DOMContentLoaded', () => {
    const authToken = localStorage.getItem('authToken');
    const userId = localStorage.getItem('userId');

    if (!authToken || !userId) {
        console.warn("User not authenticated. Redirecting to login.");
        window.location.href = 'login.html';
        return; // Stop execution if not logged in
    }

    const strategyListContainer = document.getElementById('strategyListContainer');
    const modal = document.getElementById('strategyDetailsModal');
    const closeModalBtn = document.getElementById('closeModalBtn');
    const modalStrategyName = document.getElementById('modalStrategyName');
    const modalStrategyDescription = document.getElementById('modalStrategyDescription');
    const modalStrategyCategory = document.getElementById('modalStrategyCategory');
    const modalStrategyRisk = document.getElementById('modalStrategyRisk');
    const modalStrategyParamsContainer = document.getElementById('modalStrategyParamsContainer');
    const strategyCustomizationForm = document.getElementById('strategyCustomizationForm');
    const modalApiKeySelect = document.getElementById('modalApiKeySelect');
    const subscribeButton = document.getElementById('subscribeButton'); // Assuming a button with this ID exists in the modal form

    let currentStrategyData = null; // Store details of the currently viewed strategy

    async function fetchAvailableStrategies() {
        if (!strategyListContainer) return;
        strategyListContainer.innerHTML = '<p>Loading strategies...</p>';

        try {
            // Corrected API: GET /api/v1/strategies/ (Public endpoint)
            const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/strategies/`, { headers: { 'Authorization': `Bearer ${authToken}` } }); // Include auth token even for public? Backend might filter based on auth.
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json(); // Expects { status, strategies: [...] } or just [...] based on backend schema

            // Assuming the backend returns a list directly or { strategies: [...] }
            const strategies = data.strategies || data; // Adjust based on actual backend response structure

            if (strategies && strategies.length > 0) {
                strategyListContainer.innerHTML = '';
                strategies.forEach(strategy => {
                    const card = document.createElement('div');
                    card.className = 'card strategy-card'; // Use .card for global styling
                    card.innerHTML = `
                        <div class="card-header"><h3>${strategy.name || 'Unnamed Strategy'}</h3></div>
                        <p>${strategy.description || 'No description available.'}</p>
                        <p><strong>Category:</strong> ${strategy.category || 'N/A'} | <strong>Risk:</strong> ${strategy.risk_level || 'N/A'}</p>
                        <p><em>${strategy.historical_performance_summary || "No performance summary available."}</em></p>
                        <button class="btn details-btn" data-strategy-id="${strategy.id}">View Details & Customize</button>
                    `;
                    strategyListContainer.appendChild(card);
                });
            } else {
                strategyListContainer.innerHTML = '<p>No strategies available at the moment.</p>';
            }
        } catch (error) {
            console.error("Failed to load strategies:", error);
            strategyListContainer.innerHTML = `<p class="error-message">Error loading strategies: ${error.message}</p>`;
        }
    }

    async function fetchUserApiKeys() {
        if (!modalApiKeySelect) return;
        modalApiKeySelect.innerHTML = '<option value="">Loading API Keys...</option>';
        try {
            // Corrected API: GET /api/v1/exchanges/api-keys (Lists all user keys)
            const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/exchanges/api-keys`, { headers: { 'Authorization': `Bearer ${authToken}` } });
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json(); // Expects { status: "success", keys: [...] }

            if (data.status === "success" && data.keys) {
                const activeKeys = data.keys.filter(key => key.status === 'active'); // Filter for active keys on frontend
                modalApiKeySelect.innerHTML = '<option value="">Select API Key for this strategy</option>';
                if (activeKeys.length > 0) {
                    activeKeys.forEach(key => {
                        const option = document.createElement('option');
                        option.value = key.id;
                        option.textContent = `${key.label || 'Unnamed Key'} (${key.exchange_name || 'N/A'})`;
                        modalApiKeySelect.appendChild(option);
                    });
                    modalApiKeySelect.disabled = false;
                    if (subscribeButton) subscribeButton.disabled = false; // Enable subscribe button if keys are available
                } else {
                     modalApiKeySelect.innerHTML = '<option value="">No active API keys found. Please add one on the dashboard.</option>';
                     modalApiKeySelect.disabled = true;
                     if (subscribeButton) subscribeButton.disabled = true; // Disable subscribe button
                }
            } else {
                throw new Error(data.message || "Failed to parse API keys.");
            }
        } catch (error) {
            console.error("Failed to load user API keys:", error);
            modalApiKeySelect.innerHTML = `<option value="">Error loading API keys</option>`;
            modalApiKeySelect.disabled = true;
            if (subscribeButton) subscribeButton.disabled = true; // Disable subscribe button
        }
    }

    async function openStrategyModal(strategyId) {
        currentStrategyData = null; // Reset
        modal.style.display = 'block';
        modalStrategyParamsContainer.innerHTML = '<p><em>Loading parameters...</em></p>';
        if (subscribeButton) subscribeButton.disabled = true; // Disable subscribe button while loading

        try {
            // Corrected API: GET /api/v1/strategies/{strategy_db_id} (Public endpoint)
            const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/strategies/${strategyId}`, { headers: { 'Authorization': `Bearer ${authToken}` } }); // Include auth token
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json(); // Expects { status: "success", details: {...} }

            if (data.status === "success" && data.details) {
                currentStrategyData = data.details; // Store for submission
                modalStrategyName.textContent = currentStrategyData.name || 'Unnamed Strategy';
                modalStrategyDescription.textContent = currentStrategyData.description || 'No description available.';
                modalStrategyCategory.textContent = currentStrategyData.category || 'N/A';
                modalStrategyRisk.textContent = currentStrategyData.risk_level || 'N/A';

                modalStrategyParamsContainer.innerHTML = '';
                if (currentStrategyData.parameters_definition && Object.keys(currentStrategyData.parameters_definition).length > 0) {
                    for (const paramName in currentStrategyData.parameters_definition) {
                        const paramDef = currentStrategyData.parameters_definition[paramName];
                        const group = document.createElement('div');
                        group.className = 'form-group'; // Use global form-group for styling

                        const label = document.createElement('label');
                        label.setAttribute('for', `param_${paramName}`);
                        label.textContent = `${paramDef.label || paramName}:`;

                        const input = document.createElement('input');
                        input.type = paramDef.type === "int" || paramDef.type === "float" ? "number" : "text";
                        input.id = `param_${paramName}`;
                        input.name = paramName; // Will be collected as custom_parameters
                        input.value = paramDef.default;
                        if (paramDef.min !== undefined) input.min = paramDef.min;
                        if (paramDef.max !== undefined) input.max = paramDef.max;
                        if (paramDef.type === "float") input.step = paramDef.step || "any";
                        input.required = paramDef.required || false; // Add required attribute

                        group.appendChild(label);
                        group.appendChild(input);
                        modalStrategyParamsContainer.appendChild(group);
                    }
                } else {
                     modalStrategyParamsContainer.innerHTML = '<p><em>This strategy has no customizable parameters.</em></p>';
                }

                // --- Payment Options ---
                let paymentOptionsContainer = document.getElementById('modalPaymentOptionsContainer');
                if (!paymentOptionsContainer && strategyCustomizationForm) { // Ensure form exists
                    paymentOptionsContainer = document.createElement('div');
                    paymentOptionsContainer.id = 'modalPaymentOptionsContainer';
                    paymentOptionsContainer.className = 'form-group'; // Standard styling
                    // Insert before API key selection or at a sensible place
                    const apiKeySelectDiv = modalApiKeySelect ? modalApiKeySelect.closest('.form-group') : null;
                    if (apiKeySelectDiv) {
                        strategyCustomizationForm.insertBefore(paymentOptionsContainer, apiKeySelectDiv);
                    } else {
                         // Fallback: append to params container or form if API key select not found
                        modalStrategyParamsContainer.parentNode.insertBefore(paymentOptionsContainer, modalStrategyParamsContainer.nextSibling);
                    }
                }

                if (paymentOptionsContainer) { // Check if container exists or was created
                    paymentOptionsContainer.innerHTML = ''; // Clear previous options
                    if (currentStrategyData.payment_options && currentStrategyData.payment_options.length > 0) {
                        const heading = document.createElement('label'); // Use label as it's common for form sections
                        heading.textContent = 'Subscription Options:';
                        paymentOptionsContainer.appendChild(heading);

                        currentStrategyData.payment_options.forEach((option, index) => {
                            const optionDiv = document.createElement('div');
                            optionDiv.className = 'payment-option-item'; // For styling individual options

                            const radioInput = document.createElement('input');
                            radioInput.type = 'radio';
                            radioInput.name = 'payment_option';
                            radioInput.id = `payment_option_${index}`;
                            radioInput.value = JSON.stringify(option);
                            if (index === 0) {
                                radioInput.checked = true; // Select first option by default
                            }

                            const radioLabel = document.createElement('label');
                            radioLabel.setAttribute('for', `payment_option_${index}`);
                            radioLabel.textContent = ` ${option.description || option.months + ' Month(s)'} - $${option.price_usd.toFixed(2)}`;

                            optionDiv.appendChild(radioInput);
                            optionDiv.appendChild(radioLabel);
                            paymentOptionsContainer.appendChild(optionDiv);
                        });
                    } else {
                        paymentOptionsContainer.innerHTML = '<p><em>No subscription options currently available for this strategy.</em></p>';
                        if (subscribeButton) subscribeButton.disabled = true; // Disable subscribe if no payment options
                    }
                }
                // --- End Payment Options ---

                fetchUserApiKeys(); // Fetch API keys after strategy details are loaded
            } else {
                throw new Error(data.message || "Could not load strategy details.");
            }
        } catch (error) {
            console.error(`Error opening strategy modal for ${strategyId}:`, error);
            modalStrategyName.textContent = "Error";
            modalStrategyDescription.textContent = error.message;
            modalStrategyCategory.textContent = "N/A";
            modalStrategyRisk.textContent = "N/A";
            modalStrategyParamsContainer.innerHTML = `<p class="error-message">Error loading parameters: ${error.message}</p>`;
            if (subscribeButton) subscribeButton.disabled = true; // Keep subscribe button disabled on error
        }
    }

    if (strategyListContainer) {
        strategyListContainer.addEventListener('click', (event) => {
            const button = event.target.closest('.details-btn');
            if (button) {
                const strategyId = button.dataset.strategyId;
                openStrategyModal(strategyId);
            }
        });
    }

    if (closeModalBtn) closeModalBtn.onclick = () => { modal.style.display = 'none'; currentStrategyData = null; };
    window.onclick = (event) => { if (event.target == modal) { modal.style.display = 'none'; currentStrategyData = null; }};

    if (strategyCustomizationForm) {
        strategyCustomizationForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            if (!currentStrategyData) {
                alert("Error: Strategy data not loaded."); return;
            }

            const formData = new FormData(strategyCustomizationForm);
            const customParameters = {};
            // Collect parameters based on the definition loaded
            for (const paramName in currentStrategyData.parameters_definition) {
                 const paramValue = formData.get(paramName);
                 // Convert to number if type is int or float
                 if (currentStrategyData.parameters_definition[paramName].type === "int") {
                     customParameters[paramName] = parseInt(paramValue, 10);
                 } else if (currentStrategyData.parameters_definition[paramName].type === "float") {
                     customParameters[paramName] = parseFloat(paramValue);
                 } else {
                     customParameters[paramName] = paramValue;
                 }
            }

            const apiKeyId = formData.get('apiKeyId'); // Assuming the select has name="apiKeyId"
            if (!apiKeyId) {
                alert("Please select an API Key for this strategy subscription."); return;
            }

            // --- Get selected payment option ---
            const selectedPaymentOptionRadio = document.querySelector('input[name="payment_option"]:checked');
            if (!selectedPaymentOptionRadio) {
                alert("Please select a subscription option.");
                if (subscribeButton) subscribeButton.disabled = false; // Re-enable button
                return;
            }
            const selectedPaymentOption = JSON.parse(selectedPaymentOptionRadio.value);
            const amountUsd = selectedPaymentOption.price_usd;
            const subscriptionMonths = selectedPaymentOption.months;
            const paymentDescription = selectedPaymentOption.description || `${subscriptionMonths} Month(s) Access for ${currentStrategyData.name}`;
            // --- End Get selected payment option ---

            console.log("Attempting to initiate payment for strategy:", currentStrategyData.id, "with params:", customParameters, "on API Key:", apiKeyId, "Option:", selectedPaymentOption);

            // --- Initiate Payment ---
            try {
                const createChargeResponse = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/payments/charges`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` },
                    body: JSON.stringify({
                        item_id: currentStrategyData.id, // Strategy DB ID
                        item_type: "new_strategy_subscription",
                        item_name: currentStrategyData.name,
                        item_description: paymentDescription, // Use dynamic description
                        amount_usd: amountUsd, // Use dynamic amount
                        subscription_months: subscriptionMonths, // Use dynamic months
                        // Include necessary metadata for the webhook to activate the subscription
                        metadata: {
                            user_id: userId, // Pass user_id in metadata
                            strategy_db_id: currentStrategyData.id,
                            api_key_id: parseInt(apiKeyId, 10), // Ensure it's an integer
                            custom_parameters_json: JSON.stringify(customParameters) // Pass custom params as JSON string
                        },
                        // Optional: Provide redirect/cancel URLs
                        // redirect_url: `${window.location.origin}/payment/success`,
                        // cancel_url: `${window.location.origin}/payment/cancel`
                    })
                });

                const chargeResult = await createChargeResponse.json();

                if (createChargeResponse.ok) {
                    // Redirect user to the payment page provided by Coinbase Commerce
                    if (chargeResult.hosted_url) {
                        alert(`Redirecting to payment page for ${currentStrategyData.name}...`);
                        window.location.href = chargeResult.hosted_url;
                    } else if (chargeResult.status === "success_simulated") {
                         // Handle simulated success if backend is configured for it
                         alert("Payment simulation successful. Subscription should be active shortly.");
                         modal.style.display = 'none';
                         window.location.href = 'dashboard.html#subscriptions'; // Redirect to dashboard
                    }
                    else {
                        throw new Error("Payment initiation failed: No hosted_url received.");
                    }
                } else {
                    // Handle backend errors during charge creation
                    alert('Payment initiation failed: ' + (chargeResult.detail || chargeResult.message || "Unknown error."));
                }
            } catch (error) {
                console.error("Error initiating payment:", error);
                alert("An error occurred while initiating payment: " + error.message);
            }

            // --- Removed Simulated Subscription Activation ---
            // The actual subscription activation happens on the backend via the Coinbase Commerce webhook
            // after the user successfully pays. The frontend should not directly activate the subscription here.
            // It should redirect to the payment page and potentially show a "pending payment" status
            // until the webhook confirms the payment and the backend activates the subscription.
        });
    }

    initializeStrategiesPage(); // Renamed from fetchAvailableStrategies for clarity
    function initializeStrategiesPage() { // Wrapper for clarity
        fetchAvailableStrategies();
    }
});
