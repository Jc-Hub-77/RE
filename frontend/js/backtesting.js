// frontend/js/backtesting.js
console.log("backtesting.js loaded");

// Helper function to get CSS variable values
function getCssVariable(varName) {
    return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
}

// const BACKEND_API_BASE_URL = 'http://127.0.0.1:8000'; // Ensure this is correct - This will now be set globally via HTML script tag

// Constants for backtest result polling
const BACKTEST_POLL_INTERVAL_MS = 5000; // Interval in milliseconds for polling backtest results
const BACKTEST_MAX_POLL_ATTEMPTS = 360;  // Maximum number of polling attempts (e.g., 360 * 5s = 30 minutes)

document.addEventListener('DOMContentLoaded', () => {
    const authToken = localStorage.getItem('authToken');
    const userId = localStorage.getItem('userId');

    if (!authToken || !userId) {
        console.warn("User not authenticated. Redirecting to login.");
        window.location.href = 'login.html';
        return;
    }

    const backtestSetupForm = document.getElementById('backtestSetupForm');
    const strategySelect = document.getElementById('backtestStrategySelect');
    const paramsContainer = document.getElementById('backtestStrategyParamsContainer');
    const resultsSection = document.getElementById('backtestResultsSection');
    const resultsLoading = document.getElementById('resultsLoading');
    const resultsContent = document.getElementById('resultsContent');
    const metricsSummaryContainer = document.getElementById('metricsSummaryContainer');
    const priceChartContainer = document.getElementById('priceChartContainer');
    const equityChartContainer = document.getElementById('equityChartContainer');
    const tradesLogTableBody = document.getElementById('tradesLogTableBody');

    let priceChart = null;
    let equityChart = null;
    let candlestickSeries = null;
    let equitySeries = null;

    // Function to define common chart options
    const commonChartOptions = (isDarkMode) => ({
        width: priceChartContainer.clientWidth, // This will be specific to each chart container
        height: 400,
        layout: {
            textColor: isDarkMode ? getCssVariable('--dm-text-color') : getCssVariable('--text-color'),
            background: { 
                type: 'solid', 
                color: isDarkMode ? getCssVariable('--dm-surface-color') : getCssVariable('--card-background')
            },
            fontSize: 12,
        },
        grid: {
            vertLines: {
                color: isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)',
            },
            horzLines: {
                color: isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)',
            }
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
            vertLine: {
                color: isDarkMode ? getCssVariable('--dm-text-color-muted') : getCssVariable('--dark-gray'),
                style: LightweightCharts.LineStyle.Dashed,
            },
            horzLine: {
                color: isDarkMode ? getCssVariable('--dm-text-color-muted') : getCssVariable('--dark-gray'),
                style: LightweightCharts.LineStyle.Dashed,
            }
        },
        timeScale: {
            borderColor: isDarkMode ? getCssVariable('--dm-border-color') : getCssVariable('--border-color'),
        },
        rightPriceScale: {
            borderColor: isDarkMode ? getCssVariable('--dm-border-color') : getCssVariable('--border-color'),
        }
    });


    async function initializeBacktestPage() {
        await populateStrategySelect();
        setDefaultDates();
        if (strategySelect) {
            strategySelect.addEventListener('change', loadStrategyParameters);
        }
    }

    function setDefaultDates() {
        const endDateInput = document.getElementById('backtestEndDate');
        const startDateInput = document.getElementById('backtestStartDate');
        if (!endDateInput || !startDateInput) return;

        const now = new Date();
        const yesterday = new Date(now);
        yesterday.setDate(now.getDate() - 1);
        const defaultEndDate = new Date(yesterday.getFullYear(), yesterday.getMonth(), yesterday.getDate(), 23, 59);
        
        const defaultStartDate = new Date(defaultEndDate);
        defaultStartDate.setMonth(defaultEndDate.getMonth() - 1);

        endDateInput.value = defaultEndDate.toISOString().slice(0,16);
        startDateInput.value = defaultStartDate.toISOString().slice(0,16);
    }

    async function populateStrategySelect() {
        if (!strategySelect) return;
        strategySelect.innerHTML = '<option value="">Loading strategies...</option>';
        try {
            const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/strategies/`, {
                headers: { 'Authorization': `Bearer ${authToken}` }
            });
            if (!response.ok) { if (response.status === 401) { window.location.href = 'login.html'; return; } throw new Error(`HTTP error! status: ${response.status}`);}
            const data = await response.json(); 

            const strategies = data.strategies || data; 

            if (strategies && strategies.length > 0) {
                strategySelect.innerHTML = '<option value="">Select Strategy</option>';
                strategies.forEach(strategy => {
                    const option = document.createElement('option');
                    option.value = strategy.id;
                    option.textContent = strategy.name;
                    strategySelect.appendChild(option);
                });
            } else { throw new Error(data.message || "No strategies available or failed to parse."); }
        } catch (error) {
            console.error("Failed to load strategies for backtest:", error);
            strategySelect.innerHTML = '<option value="">Error loading strategies</option>';
        }
    }

    async function loadStrategyParameters() {
        const strategyId = strategySelect.value;
        if (!paramsContainer) return;
        if (!strategyId) {
            paramsContainer.innerHTML = '<p><em>Select a strategy to see its parameters.</em></p>';
            return;
        }
        paramsContainer.innerHTML = '<p><em>Loading parameters...</em></p>';
        try {
            const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/strategies/${strategyId}`, {
                headers: { 'Authorization': `Bearer ${authToken}` }
            });
            if (!response.ok) { if (response.status === 401) { window.location.href = 'login.html'; return; } throw new Error(`HTTP error! status: ${response.status}`);}
            const data = await response.json();

            if (data.status === "success" && data.details && data.details.parameters_definition) {
                paramsContainer.innerHTML = '<h3>Strategy Parameters:</h3>';
                const paramsGrid = document.createElement('div');
                paramsGrid.className = 'form-grid';
                for (const paramName in data.details.parameters_definition) {
                    const paramDef = data.details.parameters_definition[paramName];
                    const paramGroup = document.createElement('div'); 
                    paramGroup.className = 'form-group';
                    const label = document.createElement('label');
                    label.setAttribute('for', `param_${paramName}`);
                    label.textContent = paramDef.label || paramName;
                    
                    let input;
                    if (paramDef.type === "select" && paramDef.options) {
                        input = document.createElement('select');
                        input.id = `param_${paramName}`;
                        input.name = `custom_params_${paramName}`;
                        paramDef.options.forEach(opt => {
                            const optionEl = document.createElement('option');
                            optionEl.value = opt;
                            optionEl.textContent = opt;
                            if (opt === paramDef.default) optionEl.selected = true;
                            input.appendChild(optionEl);
                        });
                    } else {
                        input = document.createElement('input');
                        input.type = paramDef.type === "int" || paramDef.type === "float" ? "number" : (paramDef.type === "bool" ? "checkbox" : "text");
                        input.id = `param_${paramName}`;
                        input.name = `custom_params_${paramName}`;
                        if (paramDef.type === "bool") {
                            input.checked = paramDef.default === true || String(paramDef.default).toLowerCase() === "true";
                        } else {
                            input.value = paramDef.default;
                        }
                        if (input.type === "number") {
                             if (paramDef.min !== undefined) input.min = paramDef.min;
                             if (paramDef.max !== undefined) input.max = paramDef.max;
                             input.step = paramDef.step || (paramDef.type === "float" ? "any" : "1");
                        }
                    }
                    input.required = paramDef.required || false;
                    
                    paramGroup.appendChild(label); 
                    if (paramDef.type === "bool") { 
                        const wrapper = document.createElement('div');
                        wrapper.style.display = 'flex';
                        wrapper.style.alignItems = 'center';
                        input.style.marginRight = '10px';
                        wrapper.appendChild(input);
                        wrapper.appendChild(label); 
                        paramGroup.appendChild(wrapper);
                    } else {
                        paramGroup.appendChild(input);
                    }
                    paramsGrid.appendChild(paramGroup);
                }
                paramsContainer.appendChild(paramsGrid);
            } else { throw new Error(data.message || "Could not load parameters."); }
        } catch (error) {
            console.error("Error loading strategy parameters:", error);
            paramsContainer.innerHTML = `<p class="error-message">Error loading parameters: ${error.message}</p>`;
        }
    }

    if (backtestSetupForm) {
        backtestSetupForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            resultsSection.style.display = 'block';
            resultsContent.style.display = 'none';
            metricsSummaryContainer.innerHTML = '';
            if(priceChart) priceChart.remove(); priceChart = null;
            if(equityChart) equityChart.remove(); equityChart = null;
            if(tradesLogTableBody) tradesLogTableBody.innerHTML = '';
            resultsLoading.style.display = 'block';
            resultsLoading.textContent = 'Submitting backtest job...';

            const formData = new FormData(backtestSetupForm);
            const customParameters = {};
            const strategyId = parseInt(formData.get('strategyId'), 10);
            const strategyDetailsResponse = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/strategies/${strategyId}`, { headers: { 'Authorization': `Bearer ${authToken}` } });
            const strategyDetailsData = await strategyDetailsResponse.json();
            const paramDefs = strategyDetailsData.details?.parameters_definition || {};

            for (const [key, value] of formData.entries()) {
                if (key.startsWith('custom_params_')) {
                    const paramName = key.replace('custom_params_', '');
                    const paramDef = paramDefs[paramName];
                    if (paramDef) {
                        if (paramDef.type === "int") customParameters[paramName] = parseInt(value, 10);
                        else if (paramDef.type === "float") customParameters[paramName] = parseFloat(value);
                        else if (paramDef.type === "bool") customParameters[paramName] = document.getElementById(`param_${paramName}`).checked;
                        else customParameters[paramName] = value;
                    } else {
                         customParameters[paramName] = value;
                    }
                }
            }
            
            const payload = {
                backtest_params: {
                    strategy_db_id: strategyId,
                    custom_parameters: customParameters 
                },
                exchange_id: formData.get('exchangeId'),
                symbol: formData.get('symbol'),
                timeframe: formData.get('timeframe'),
                start_date: new Date(formData.get('startDate')).toISOString(),
                end_date: new Date(formData.get('endDate')).toISOString(),
                initial_capital: parseFloat(formData.get('initialCapital'))
            };

            console.log("Running backtest with payload:", JSON.stringify(payload, null, 2));
            try {
                const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/backtests`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` },
                    body: JSON.stringify(payload)
                });
                
                const result = await response.json();

                if (!response.ok || result.status === "error") {
                    const errData = result;
                    throw new Error(errData.detail || errData.message || `HTTP error! status: ${response.status}`);
                }

                if (result.backtest_id) {
                    resultsLoading.textContent = `Backtest job queued (ID: ${result.backtest_id}). Polling for results...`;
                    pollForBacktestResults(result.backtest_id);
                } else {
                    throw new Error("Backtest job submission did not return a backtest ID.");
                }
            } catch (error) {
                console.error("Backtest submission error:", error);
                resultsLoading.style.display = 'none';
                metricsSummaryContainer.innerHTML = `<p class="error-message">Backtest submission failed: ${error.message}</p>`;
            }
        });
    }

    async function pollForBacktestResults(backtestId) {
        let attempts = 0;

        const intervalId = setInterval(async () => {
            attempts++;
            if (attempts > BACKTEST_MAX_POLL_ATTEMPTS) {
                clearInterval(intervalId);
                resultsLoading.textContent = 'Backtest timed out waiting for results.';
                console.error(`Timeout polling for backtest ID ${backtestId}`);
                return;
            }
            resultsLoading.textContent = `Polling for results... (Attempt ${attempts}/${BACKTEST_MAX_POLL_ATTEMPTS})`;
            try {
                const response = await fetch(`${window.BACKEND_API_BASE_URL}/api/v1/backtests/${backtestId}`, {
                    headers: { 'Authorization': `Bearer ${authToken}` }
                });
                if (!response.ok) {
                    if (response.status === 404) {
                        clearInterval(intervalId);
                        resultsLoading.style.display = 'none';
                        metricsSummaryContainer.innerHTML = `<p class="error-message">Backtest ID ${backtestId} not found. Please try again.</p>`;
                        return;
                    }
                    console.warn(`Polling error for backtest ID ${backtestId}: HTTP ${response.status}`);
                    return; 
                }
                
                const results = await response.json();

                if (results.status === "completed" || results.status === "failed" || results.status === "no_data") {
                    clearInterval(intervalId);
                    resultsLoading.style.display = 'none';
                    resultsContent.style.display = 'block';
                    if (results.status === "completed") {
                        displayBacktestResults(results);
                    } else {
                        metricsSummaryContainer.innerHTML = `<p class="error-message">Backtest ${results.status}: ${results.status_message || "No further details."}</p>`;
                    }
                } else if (results.status === "running" || results.status === "queued") {
                    resultsLoading.textContent = `Backtest is ${results.status}. Polling... (Attempt ${attempts}/${BACKTEST_MAX_POLL_ATTEMPTS})`;
                } else { 
                    clearInterval(intervalId);
                    resultsLoading.style.display = 'none';
                    metricsSummaryContainer.innerHTML = `<p class="error-message">Unknown backtest status: ${results.status}. Message: ${results.status_message}</p>`;
                }
            } catch (error) {
                console.error(`Error polling for backtest ID ${backtestId}:`, error);
            }
        }, BACKTEST_POLL_INTERVAL_MS);
    }

    function displayBacktestResults(results) {
        const details = results.details || {};
        const isDarkMode = document.body.classList.contains('dark-mode');
        
        metricsSummaryContainer.innerHTML = `
            <div class="metric-item"><strong>Strategy:</strong> ${details.strategy_name_used || results.strategy_name_used || 'N/A'}</div>
            <div class="metric-item"><strong>Symbol:</strong> ${details.symbol || results.symbol || 'N/A'} (${details.timeframe || results.timeframe || 'N/A'})</div>
            <div class="metric-item"><strong>Period:</strong> ${new Date(details.start_date || results.start_date).toLocaleDateString()} - ${new Date(details.end_date || results.end_date).toLocaleDateString()}</div>
            <div class="metric-item"><strong>Initial Capital:</strong> $${(details.initial_capital !== undefined ? details.initial_capital : (results.initial_capital !== undefined ? results.initial_capital : 0)).toFixed(2)}</div>
            <div class="metric-item"><strong>Final Equity:</strong> $${(details.final_equity !== undefined ? details.final_equity : (results.final_equity !== undefined ? results.final_equity : 0)).toFixed(2)}</div>
            <div class="metric-item"><strong>PnL:</strong> $${(details.pnl !== undefined ? details.pnl : (results.pnl !== undefined ? results.pnl : 0)).toFixed(2)} (${(details.pnl_percentage !== undefined ? details.pnl_percentage : (results.pnl_percentage !== undefined ? results.pnl_percentage : 0)).toFixed(2)}%)</div>
            <div class="metric-item"><strong>Sharpe Ratio:</strong> ${(details.sharpe_ratio !== undefined ? details.sharpe_ratio : (results.sharpe_ratio !== undefined ? results.sharpe_ratio : 0)).toFixed(2)}</div>
            <div class="metric-item"><strong>Max Drawdown:</strong> ${(details.max_drawdown !== undefined ? details.max_drawdown : (results.max_drawdown !== undefined ? results.max_drawdown : 0)).toFixed(2)}%</div>
            <div class="metric-item"><strong>Total Trades:</strong> ${details.total_trades !== undefined ? details.total_trades : (results.total_trades !== undefined ? results.total_trades : 0)}</div>
            <div class="metric-item"><strong>Win Rate:</strong> ${(details.win_rate !== undefined ? details.win_rate : (results.win_rate !== undefined ? results.win_rate : 0)).toFixed(2)}%</div>`;

        const ohlcvDataForChart = details.ohlcv_data ? details.ohlcv_data.map(d => ({ time: d.time, open: d.open, high: d.high, low: d.low, close: d.close })) : [];
        const equityCurveForChart = details.equity_curve ? details.equity_curve.map(d => ({ time: d.time, value: d.value })) : [];
        
        const tradesLogForChart = details.trades_log ? details.trades_log.map(t => ({
            time: t.entry_time,
            position: t.type === 'long' ? 'belowBar' : 'aboveBar',
            color: t.type === 'long' ? getCssVariable('--success-color') : getCssVariable('--danger-color'),
            shape: t.type === 'long' ? 'arrowUp' : 'arrowDown',
            text: `${t.type.toUpperCase()} @ ${t.entry_price.toFixed(2)}` + (t.exit_time ? ` -> ${t.exit_price.toFixed(2)} (P: ${t.pnl.toFixed(2)})` : '')
        })) : [];
        
        if(details.trades_log) {
            details.trades_log.forEach(trade => {
                if (trade.exit_time) {
                    tradesLogForChart.push({
                        time: trade.exit_time,
                        position: trade.type === 'long' ? 'aboveBar' : 'belowBar',
                        color: getCssVariable('--dark-gray'), 
                        shape: trade.type === 'long' ? 'arrowDown' : 'arrowUp',
                        text: `Exit ${trade.type.toUpperCase()} @ ${trade.exit_price.toFixed(2)} (P: ${trade.pnl.toFixed(2)})`
                    });
                }
            });
        }

        if (priceChart) priceChart.remove();
        const priceChartSpecificOptions = { ...commonChartOptions(isDarkMode), width: priceChartContainer.clientWidth };
        priceChart = LightweightCharts.createChart(priceChartContainer, priceChartSpecificOptions);
        candlestickSeries = priceChart.addCandlestickSeries({
            upColor: getCssVariable('--success-color'),
            downColor: getCssVariable('--danger-color'),
            borderVisible: false,
            wickUpColor: getCssVariable('--success-color'),
            wickDownColor: getCssVariable('--danger-color'),
        });
        if (ohlcvDataForChart.length > 0) candlestickSeries.setData(ohlcvDataForChart);
        else console.warn("OHLCV data not provided or empty for price chart.");
        if(candlestickSeries && tradesLogForChart.length > 0) candlestickSeries.setMarkers(tradesLogForChart);
        priceChart.timeScale().fitContent();

        if (equityChart) equityChart.remove();
        const equityChartSpecificOptions = { ...commonChartOptions(isDarkMode), width: equityChartContainer.clientWidth };
        equityChart = LightweightCharts.createChart(equityChartContainer, equityChartSpecificOptions);
        equitySeries = equityChart.addLineSeries({ 
            color: isDarkMode ? getCssVariable('--dm-primary-color') : getCssVariable('--primary-color'),
            lineWidth: 2 
        });
        if (equityCurveForChart.length > 0) equitySeries.setData(equityCurveForChart);
        equityChart.timeScale().fitContent();

        tradesLogTableBody.innerHTML = '';
        if(details.trades_log) {
            details.trades_log.forEach(trade => {
                const row = tradesLogTableBody.insertRow();
                row.insertCell().textContent = trade.type || 'N/A';
                row.insertCell().textContent = trade.entry_time ? new Date(trade.entry_time * 1000).toLocaleString() : 'N/A';
                row.insertCell().textContent = trade.entry_price ? trade.entry_price.toFixed(2) : 'N/A';
                row.insertCell().textContent = trade.exit_time ? new Date(trade.exit_time * 1000).toLocaleString() : 'N/A';
                row.insertCell().textContent = trade.exit_price ? trade.exit_price.toFixed(2) : 'N/A';
                row.insertCell().textContent = trade.size || 'N/A';
                row.insertCell().textContent = trade.take_profit ? trade.take_profit.toFixed(2) : 'N/A';
                row.insertCell().textContent = trade.stop_loss ? trade.stop_loss.toFixed(2) : 'N/A';
                
                const pnlCell = row.insertCell();
                const pnlValue = trade.pnl !== undefined ? parseFloat(trade.pnl) : NaN;
                pnlCell.textContent = !isNaN(pnlValue) ? pnlValue.toFixed(2) : 'N/A';
                if (!isNaN(pnlValue)) {
                    pnlCell.className = pnlValue >= 0 ? 'pnl-positive' : 'pnl-negative';
                }
                row.insertCell().textContent = trade.reason || 'N/A';
            });
        }
    }
    
    const themeToggle = document.getElementById('themeSwitch');
    if (themeToggle) {
        themeToggle.addEventListener('change', function() {
            const isDarkMode = document.body.classList.contains('dark-mode');
            const newPriceChartOptions = { ...commonChartOptions(isDarkMode), width: priceChartContainer.clientWidth };
            const newEquityChartOptions = { ...commonChartOptions(isDarkMode), width: equityChartContainer.clientWidth };

            if (priceChart) priceChart.applyOptions(newPriceChartOptions);
            if (candlestickSeries) candlestickSeries.applyOptions({
                upColor: getCssVariable('--success-color'),
                downColor: getCssVariable('--danger-color'),
                wickUpColor: getCssVariable('--success-color'),
                wickDownColor: getCssVariable('--danger-color'),
            });
            if (equityChart) equityChart.applyOptions(newEquityChartOptions);
            if (equitySeries) equitySeries.applyOptions({ 
                color: isDarkMode ? getCssVariable('--dm-primary-color') : getCssVariable('--primary-color') 
            });
        });
    }

    if (authToken && userId) {
        initializeBacktestPage();
    } else {
        console.warn("User not authenticated. Backtesting page will not initialize fully.");
        if(strategySelect) strategySelect.innerHTML = '<option value="">Please login</option>';
        if(paramsContainer) paramsContainer.innerHTML = '<p>Please login to load parameters.</p>';
    }
});
