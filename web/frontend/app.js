// SMC Trading Bot - Web Interface JavaScript

class SMCWebInterface {
    constructor() {
        this.ws = null;
        this.isConnected = false;
        this.signals = [];
        this.config = {
            symbols: ['BTCUSDT'],
            min_risk_reward: 3.0,
            fractal_left: 2,
            fractal_right: 2,
            telegram_token: '7834170834:AAG1OxqOxCjxFP38oUW-TAPidA7CkfV2c3c',
            telegram_chat_id: '333744879',
            status_check_interval: 45
        };
        
        this.initializeElements();
        this.bindEvents();
        this.connectWebSocket();
        this.loadConfig();
        this.loadSettings();
    }
    
    initializeElements() {
        // Control elements
        this.newSymbolInput = document.getElementById('new-symbol');
        this.addSymbolBtn = document.getElementById('add-symbol-btn');
        this.minRrInput = document.getElementById('min-rr');
        this.startBtn = document.getElementById('start-btn');
        this.stopBtn = document.getElementById('stop-btn');
        this.clearBtn = document.getElementById('clear-btn');
        this.testAlertBtn = document.getElementById('test-alert-btn');
        
        // Backtest elements
        this.autoBacktestCheckbox = document.getElementById('auto-backtest');
        this.backtestDaysInput = document.getElementById('backtest-days');
        this.runBacktestBtn = document.getElementById('run-backtest-btn');
        
        // Status elements
        this.statusText = document.getElementById('status-text');
        this.statusDot = document.getElementById('status-dot');
        this.enginesCount = document.getElementById('engines-count');
        this.wsStatus = document.getElementById('ws-status');
        this.symbolsGrid = document.getElementById('symbols-grid');
        this.pairsList = document.getElementById('pairs-list');
        
        // Statistics elements
        this.totalSignals = document.getElementById('total-signals');
        this.validSignals = document.getElementById('valid-signals');
        this.triggeredSignals = document.getElementById('triggered-signals');
        this.tpSignals = document.getElementById('tp-signals');
        this.slSignals = document.getElementById('sl-signals');
        this.winRate = document.getElementById('win-rate');
        
        // Table elements
        this.signalsTable = document.getElementById('signals-tbody');
        
        // Telegram elements
        this.telegramToken = document.getElementById('telegram-token');
        this.telegramChatId = document.getElementById('telegram-chat-id');
        this.saveTelegramBtn = document.getElementById('save-telegram-btn');
    }
    
    bindEvents() {
        // Control buttons
        this.startBtn.addEventListener('click', () => this.startBot());
        this.stopBtn.addEventListener('click', () => this.stopBot());
        this.clearBtn.addEventListener('click', () => this.clearSignals());
        this.testAlertBtn.addEventListener('click', () => this.sendTestAlert());
        
        // Backtest events
        this.runBacktestBtn.addEventListener('click', () => this.runBacktest());
        this.autoBacktestCheckbox.addEventListener('change', () => this.saveSettings());
        
        // Symbol management
        this.addSymbolBtn.addEventListener('click', () => this.addSymbol());
        this.newSymbolInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.addSymbol();
        });
        
        // Config inputs
        this.minRrInput.addEventListener('change', () => this.updateConfig());
        
        // Telegram config
        this.saveTelegramBtn.addEventListener('click', () => this.saveTelegramConfig());
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch(e.key) {
                    case 's':
                        e.preventDefault();
                        this.startBot();
                        break;
                    case 'x':
                        e.preventDefault();
                        this.stopBot();
                        break;
                    case 'd':
                        e.preventDefault();
                        this.clearSignals();
                        break;
                }
            }
        });
    }
    
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            this.isConnected = true;
            this.updateConnectionStatus(true);
            this.showToast('Connected to SMC Bot', 'success');
            
            // Start heartbeat
            this.startHeartbeat();
        };
        
        this.ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            this.handleWebSocketMessage(message);
        };
        
        this.ws.onclose = () => {
            this.isConnected = false;
            this.updateConnectionStatus(false);
            this.stopHeartbeat();
            this.showToast('Disconnected from SMC Bot', 'error');
            
            // Attempt to reconnect after 5 seconds with backoff
            setTimeout(() => {
                if (!this.isConnected) {
                    console.log('Attempting to reconnect...');
                    this.connectWebSocket();
                }
            }, 5000);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.showToast('WebSocket connection error', 'error');
        };
    }
    
    handleWebSocketMessage(message) {
        switch (message.type) {
            case 'initial_data':
                this.handleInitialData(message.data);
                break;
                
            case 'new_signal':
                this.handleNewSignal(message.data);
                break;
                
            case 'status_update':
                this.handleStatusUpdate(message.data);
                break;
                
            case 'signals_cleared':
                this.signals = [];
                this.updateSignalsTable();
                this.updateStatistics();
                break;
                
            case 'heartbeat':
            case 'pong':
                // Keep alive - connection is good
                break;
                
            default:
                console.log('Unknown message type:', message.type);
        }
    }
    
    handleInitialData(data) {
        this.signals = data.signals || [];
        this.config = { ...this.config, ...data.config };
        
        // Update UI with config
        this.minRrInput.value = this.config.min_risk_reward;
        this.telegramToken.value = this.config.telegram_token || '';
        this.telegramChatId.value = this.config.telegram_chat_id || '';
        
        // Update pairs list
        this.updatePairsList();
        
        // Update status
        this.updateBotStatus(data.status.is_running);
        this.updateMarketInfo(data.status);
        
        // Update tables and stats
        this.updateSignalsTable();
        this.updateStatistics();
    }
    
    handleNewSignal(signal) {
        this.signals.push(signal);
        
        // Keep only last 50 signals
        if (this.signals.length > 50) {
            this.signals = this.signals.slice(-50);
        }
        
        this.updateSignalsTable();
        this.updateStatistics();
        
        // Show notification
        this.showToast(`New ${signal.direction} signal for ${signal.symbol}`, 'success');
        
        // Browser notification (if permitted)
        this.showBrowserNotification(signal);
    }
    
    handleStatusUpdate(data) {
        this.updateBotStatus(data.is_running);
        this.updateMarketInfo(data);
    }
    
    updateConnectionStatus(connected) {
        this.statusText.textContent = connected ? 'Connected' : 'Disconnected';
        this.statusDot.className = `status-dot ${connected ? 'online' : 'offline'}`;
        this.wsStatus.textContent = connected ? 'Connected' : 'Disconnected';
        this.wsStatus.className = connected ? 'status-online' : 'status-offline';
    }
    
    updateBotStatus(isRunning) {
        this.startBtn.disabled = isRunning;
        this.stopBtn.disabled = !isRunning;
        
        if (isRunning) {
            this.statusText.textContent = 'Bot Running';
            this.statusDot.className = 'status-dot online';
        } else if (this.isConnected) {
            this.statusText.textContent = 'Bot Stopped';
            this.statusDot.className = 'status-dot offline';
        }
    }
    
    updateMarketInfo(data) {
        // Update basic info
        this.enginesCount.textContent = data.engines_count || 0;
        
        // Update pairs list
        this.updatePairsList();
        
        // Update symbol cards
        this.updateSymbolCards(data.market_data || {});
    }
    
    updateSymbolCards(marketData) {
        this.symbolsGrid.innerHTML = '';
        
        Object.entries(marketData).forEach(([symbol, data]) => {
            const card = document.createElement('div');
            card.className = `symbol-card ${data.status || 'connecting'}`;
            
            card.innerHTML = `
                <div class="symbol-header">
                    <div class="symbol-name">${symbol}</div>
                    <div class="symbol-status ${data.status || 'connecting'}">${(data.status || 'connecting').toUpperCase()}</div>
                </div>
                <div class="symbol-details">
                    <div class="symbol-detail">
                        <div class="symbol-detail-label">Price</div>
                        <div class="symbol-detail-value">$${this.formatPrice(data.current_price || 0)}</div>
                    </div>
                    <div class="symbol-detail">
                        <div class="symbol-detail-label">HTF Bias</div>
                        <div class="symbol-detail-value bias-${(data.htf_bias || 'neutral').toLowerCase()}">
                            ${(data.htf_bias || 'NEUTRAL').toUpperCase()}
                        </div>
                    </div>
                </div>
            `;
            
            this.symbolsGrid.appendChild(card);
        });
    }
    
    updateSignalsTable() {
        if (this.signals.length === 0) {
            this.signalsTable.innerHTML = `
                <tr class="no-signals">
                    <td colspan="9">
                        <i class="fas fa-hourglass-half"></i> Waiting for signals...
                    </td>
                </tr>
            `;
            return;
        }
        
        // Show last 20 signals (most recent first)
        const recentSignals = this.signals.slice(-20).reverse();
        
        this.signalsTable.innerHTML = recentSignals.map(signal => {
            const time = new Date(signal.timestamp_str).toLocaleTimeString('en-US', {
                hour12: false,
                hour: '2-digit',
                minute: '2-digit'
            });
            
            const directionClass = signal.direction === 'LONG' ? 'signal-long' : 'signal-short';
            const statusBadge = this.getStatusBadge(signal.status);
            
            return `
                <tr>
                    <td>${time}</td>
                    <td><span class="${directionClass}">${signal.direction}</span></td>
                    <td>$${this.formatPrice(signal.entry)}</td>
                    <td>$${this.formatPrice(signal.sl)}</td>
                    <td>$${this.formatPrice(signal.tp)}</td>
                    <td>${signal.rr.toFixed(1)}</td>
                    <td>${signal.confidence || 'med'}</td>
                    <td>${statusBadge}</td>
                    <td>
                        <button class="action-btn copy-btn" onclick="copyToClipboard('${signal.entry}')" title="Copy Entry">
                            <i class="fas fa-copy"></i>
                        </button>
                        <button class="action-btn copy-btn" onclick="copyToClipboard('${signal.sl}')" title="Copy SL">
                            SL
                        </button>
                        <button class="action-btn copy-btn" onclick="copyToClipboard('${signal.tp}')" title="Copy TP">
                            TP
                        </button>
                    </td>
                </tr>
            `;
        }).join('');
    }
    
    updateStatistics() {
        const total = this.signals.length;
        let valid = 0, triggered = 0, hitTp = 0, hitSl = 0;
        
        this.signals.forEach(signal => {
            const status = signal.status || 'new';
            switch (status) {
                case 'valid': valid++; break;
                case 'triggered': triggered++; break;
                case 'hit_tp': hitTp++; break;
                case 'hit_sl': hitSl++; break;
            }
        });
        
        const totalClosed = hitTp + hitSl;
        const winRate = totalClosed > 0 ? ((hitTp / totalClosed) * 100).toFixed(1) : 0;
        
        this.totalSignals.textContent = total;
        this.validSignals.textContent = valid;
        this.triggeredSignals.textContent = triggered;
        this.tpSignals.textContent = hitTp;
        this.slSignals.textContent = hitSl;
        this.winRate.textContent = `${winRate}%`;
    }
    
    getStatusBadge(status) {
        if (!status) return '<span class="status-badge status-new">NEW</span>';
        
        const statusMap = {
            'new': { text: 'NEW', class: 'status-new' },
            'valid': { text: 'VALID', class: 'status-valid' },
            'triggered': { text: 'TRIGGERED', class: 'status-triggered' },
            'hit_tp': { text: 'HIT TP', class: 'status-hit-tp' },
            'hit_sl': { text: 'HIT SL', class: 'status-hit-sl' },
            'missed': { text: 'MISSED', class: 'status-missed' },
            'expired': { text: 'EXPIRED', class: 'status-expired' }
        };
        
        const statusInfo = statusMap[status] || { text: status.toUpperCase(), class: 'status-new' };
        return `<span class="status-badge ${statusInfo.class}">${statusInfo.text}</span>`;
    }
    
    formatPrice(price) {
        if (price >= 100) return price.toFixed(2);
        if (price >= 1) return price.toFixed(4);
        if (price >= 0.01) return price.toFixed(6);
        return price.toFixed(8);
    }
    
    async startBot() {
        try {
            await this.updateConfig();
            
            // Check if auto-backtest is enabled
            const autoBacktest = this.autoBacktestCheckbox.checked;
            
            if (autoBacktest) {
                this.showToast('Running backtest before starting...', 'info');
                
                const backtestResponse = await fetch('/api/start-with-backtest', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
                const backtestData = await backtestResponse.json();
                
                if (backtestResponse.ok) {
                    this.showToast('Bot started with backtest successfully!', 'success');
                    
                    // Show backtest results
                    if (backtestData.backtest_results) {
                        this.displayBacktestResults(backtestData.backtest_results);
                    }
                } else {
                    throw new Error(backtestData.detail || 'Failed to start bot with backtest');
                }
            } else {
                const response = await fetch('/api/start', { method: 'POST' });
                const data = await response.json();
                
                if (response.ok) {
                    this.showToast('Bot started successfully', 'success');
                } else {
                    throw new Error(data.detail || 'Failed to start bot');
                }
            }
        } catch (error) {
            this.showToast(`Error starting bot: ${error.message}`, 'error');
        }
    }
    
    async stopBot() {
        try {
            const response = await fetch('/api/stop', { method: 'POST' });
            const data = await response.json();
            
            if (response.ok) {
                this.showToast('Bot stopped successfully', 'success');
            } else {
                throw new Error(data.detail || 'Failed to stop bot');
            }
        } catch (error) {
            this.showToast(`Error stopping bot: ${error.message}`, 'error');
        }
    }
    
    async clearSignals() {
        try {
            const response = await fetch('/api/signals', { method: 'DELETE' });
            const data = await response.json();
            
            if (response.ok) {
                this.showToast('Signals cleared', 'success');
            } else {
                throw new Error(data.detail || 'Failed to clear signals');
            }
        } catch (error) {
            this.showToast(`Error clearing signals: ${error.message}`, 'error');
        }
    }
    
    addSymbol() {
        const newSymbol = this.newSymbolInput.value.toUpperCase().trim();
        
        if (!newSymbol) {
            this.showToast('Please enter a symbol', 'warning');
            return;
        }
        
        if (this.config.symbols.includes(newSymbol)) {
            this.showToast(`${newSymbol} is already added`, 'warning');
            return;
        }
        
        this.config.symbols.push(newSymbol);
        this.newSymbolInput.value = '';
        
        this.updateConfig();
        this.updatePairsList();
        this.showToast(`Added ${newSymbol}`, 'success');
    }
    
    removeSymbol(symbol) {
        const index = this.config.symbols.indexOf(symbol);
        if (index > -1) {
            this.config.symbols.splice(index, 1);
            
            this.updateConfig();
            this.updatePairsList();
            this.showToast(`Removed ${symbol}`, 'success');
        }
    }
    
    updatePairsList() {
        if (this.config.symbols.length === 0) {
            this.pairsList.innerHTML = '<div class="no-pairs">No pairs added yet. Add a symbol above.</div>';
            return;
        }
        
        this.pairsList.innerHTML = this.config.symbols.map(symbol => {
            const status = this.getSymbolStatus(symbol);
            return `
                <div class="pair-tag ${status}">
                    ${symbol}
                    <button class="pair-remove" onclick="window.smcApp.removeSymbol('${symbol}')" title="Remove ${symbol}">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            `;
        }).join('');
    }
    
    getSymbolStatus(symbol) {
        // This will be updated based on market data
        return 'active'; // Default status
    }
    
    quickAddSymbol(symbol) {
        this.newSymbolInput.value = symbol;
        this.addSymbol();
    }
    
    async sendTestAlert() {
        try {
            // Create a test signal
            const testSignal = {
                timestamp: new Date().toISOString(),
                symbol: 'TESTUSDT',
                direction: 'LONG',
                entry: 50000.00,
                sl: 49000.00,
                tp: 53000.00,
                rr: 3.0,
                htf_bias: 'bull',
                fvg_confluence: true,
                confidence: 'high'
            };
            
            const response = await fetch('/api/test-alert', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(testSignal)
            });
            
            if (response.ok) {
                this.showToast('🔔 Test alert sent to Telegram!', 'success');
            } else {
                throw new Error('Failed to send test alert');
            }
        } catch (error) {
            this.showToast(`❌ Test alert failed: ${error.message}`, 'error');
        }
    }

    async updateConfig() {
        this.config.min_risk_reward = parseFloat(this.minRrInput.value);
        
        try {
            const response = await fetch('/api/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(this.config)
            });
            
            if (!response.ok) {
                throw new Error('Failed to update config');
            }
        } catch (error) {
            console.error('Error updating config:', error);
        }
    }
    
    async saveTelegramConfig() {
        this.config.telegram_token = this.telegramToken.value;
        this.config.telegram_chat_id = this.telegramChatId.value;
        
        try {
            await this.updateConfig();
            this.showToast('Telegram configuration saved', 'success');
        } catch (error) {
            this.showToast('Error saving Telegram config', 'error');
        }
    }
    
    loadConfig() {
        const saved = localStorage.getItem('smc-config');
        if (saved) {
            const config = JSON.parse(saved);
            this.config.symbols = config.symbols || ['ETHUSDT'];
            this.minRrInput.value = config.min_risk_reward || 3.0;
            this.telegramToken.value = config.telegram_token || '';
            this.telegramChatId.value = config.telegram_chat_id || '';
            this.updatePairsList();
        }
    }
    
    saveConfig() {
        localStorage.setItem('smc-config', JSON.stringify(this.config));
    }
    
    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        
        const container = document.getElementById('toast-container');
        container.appendChild(toast);
        
        // Auto remove after 3 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 3000);
    }
    
    showBrowserNotification(signal) {
        if ('Notification' in window && Notification.permission === 'granted') {
            new Notification(`SMC Signal - ${signal.symbol}`, {
                body: `${signal.direction} @ $${this.formatPrice(signal.entry)}`,
                icon: '/static/favicon.ico'
            });
        }
    }
    
    startHeartbeat() {
        // Clear existing heartbeat
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
        }
        
        // Send ping every 25 seconds
        this.heartbeatInterval = setInterval(() => {
            if (this.isConnected && this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 25000);
    }
    
    stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
    }
    
    // Backtest methods
    async runBacktest() {
        try {
            const days = parseInt(this.backtestDaysInput.value) || 30;
            this.showToast(`Running backtest for ${days} days...`, 'info');
            
            // Run backtest for all symbols
            const promises = this.config.symbols.map(async (symbol) => {
                const response = await fetch(`/api/backtest/${symbol}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
                
                if (response.ok) {
                    return { symbol, success: true, data: await response.json() };
                } else {
                    return { symbol, success: false, error: await response.text() };
                }
            });
            
            const results = await Promise.all(promises);
            
            // Display results
            this.displayBacktestResults(results);
            this.showToast('Backtest completed!', 'success');
            
        } catch (error) {
            this.showToast(`Error running backtest: ${error.message}`, 'error');
        }
    }
    
    displayBacktestResults(results) {
        // Create a simple popup/modal to show backtest results
        const modal = document.createElement('div');
        modal.style.cssText = `
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.8); z-index: 1000;
            display: flex; align-items: center; justify-content: center;
        `;
        
        const content = document.createElement('div');
        content.style.cssText = `
            background: white; padding: 20px; border-radius: 10px;
            max-width: 600px; max-height: 70vh; overflow-y: auto;
        `;
        
        let html = '<h3>📊 Backtest Results</h3>';
        
        if (Array.isArray(results)) {
            results.forEach(result => {
                if (result.success) {
                    const data = result.data;
                    html += `
                        <div style="margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
                            <h4>${result.symbol}</h4>
                            <p><strong>Total Trades:</strong> ${data.total_trades || 'N/A'}</p>
                            <p><strong>Win Rate:</strong> ${data.win_rate || 'N/A'}%</p>
                            <p><strong>Profit Factor:</strong> ${data.profit_factor || 'N/A'}</p>
                            <p><strong>Total P&L:</strong> ${data.total_pnl || 'N/A'}</p>
                        </div>
                    `;
                } else {
                    html += `
                        <div style="margin: 10px 0; padding: 10px; border: 1px solid #f00; border-radius: 5px;">
                            <h4>${result.symbol} - Error</h4>
                            <p style="color: red;">${result.error}</p>
                        </div>
                    `;
                }
            });
        } else {
            // Single result object
            Object.keys(results).forEach(symbol => {
                const data = results[symbol];
                html += `
                    <div style="margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
                        <h4>${symbol}</h4>
                        <p><strong>Total Trades:</strong> ${data.total_trades || 'N/A'}</p>
                        <p><strong>Win Rate:</strong> ${data.win_rate || 'N/A'}%</p>
                        <p><strong>Profit Factor:</strong> ${data.profit_factor || 'N/A'}</p>
                        <p><strong>Total P&L:</strong> ${data.total_pnl || 'N/A'}</p>
                    </div>
                `;
            });
        }
        
        html += '<button onclick="this.parentElement.parentElement.remove()" style="margin-top: 15px; padding: 10px 20px; background: #3498db; color: white; border: none; border-radius: 5px; cursor: pointer;">Close</button>';
        
        content.innerHTML = html;
        modal.appendChild(content);
        document.body.appendChild(modal);
        
        // Close on backdrop click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.remove();
            }
        });
    }
    
    saveSettings() {
        // Save settings to localStorage
        const settings = {
            autoBacktest: this.autoBacktestCheckbox.checked,
            backtestDays: parseInt(this.backtestDaysInput.value) || 30
        };
        localStorage.setItem('smcBotSettings', JSON.stringify(settings));
    }
    
    loadSettings() {
        // Load settings from localStorage
        const settings = JSON.parse(localStorage.getItem('smcBotSettings') || '{}');
        
        if (settings.autoBacktest !== undefined) {
            this.autoBacktestCheckbox.checked = settings.autoBacktest;
        }
        
        if (settings.backtestDays) {
            this.backtestDaysInput.value = settings.backtestDays;
        }
    }
}

// Global functions
function toggleTelegramConfig() {
    const form = document.getElementById('telegram-form');
    const chevron = document.getElementById('telegram-chevron');
    
    form.classList.toggle('hidden');
    chevron.classList.toggle('fa-chevron-down');
    chevron.classList.toggle('fa-chevron-up');
}

function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        // Show temporary feedback
        const toast = document.createElement('div');
        toast.className = 'toast success';
        toast.textContent = `Copied: ${text}`;
        document.getElementById('toast-container').appendChild(toast);
        
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 1500);
    });
}

// Request notification permission on load
if ('Notification' in window && Notification.permission === 'default') {
    Notification.requestPermission();
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    window.smcApp = new SMCWebInterface();
});
