class TradingManager {
    constructor() {
        this.isTrading = false;
        this.portfolio = {
            cash: 10000,
            positions: {},
            totalValue: 10000
        };
        this.performanceHistory = [];
        this.updateInterval = null;
    }

    async startTrading(demoMode = true) {
        try {
            const response = await fetch('/api/trading/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ demo_mode: demoMode })
            });

            const result = await response.json();
            
            if (result.status === 'success') {
                this.isTrading = true;
                this.startPerformanceUpdates();
                this.showNotification('Trading started successfully!', 'success');
                return true;
            } else {
                throw new Error(result.message);
            }
        } catch (error) {
            console.error('Error starting trading:', error);
            this.showNotification('Failed to start trading: ' + error.message, 'error');
            return false;
        }
    }

    async stopTrading() {
        try {
            const response = await fetch('/api/trading/stop', {
                method: 'POST'
            });

            const result = await response.json();
            
            if (result.status === 'success') {
                this.isTrading = false;
                this.stopPerformanceUpdates();
                this.showNotification('Trading stopped successfully!', 'success');
                return true;
            } else {
                throw new Error(result.message);
            }
        } catch (error) {
            console.error('Error stopping trading:', error);
            this.showNotification('Failed to stop trading: ' + error.message, 'error');
            return false;
        }
    }

    async getTradingStatus() {
        try {
            const response = await fetch('/api/trading/status');
            return await response.json();
        } catch (error) {
            console.error('Error getting trading status:', error);
            return { is_running: false };
        }
    }

    async placeOrder(symbol, action, quantity, price) {
        try {
            const response = await fetch('/api/trading/order', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    symbol: symbol,
                    action: action,
                    quantity: quantity,
                    price: price
                })
            });

            return await response.json();
        } catch (error) {
            console.error('Error placing order:', error);
            return { status: 'error', message: error.message };
        }
    }

    async getPortfolio() {
        try {
            const response = await fetch('/api/trading/portfolio');
            this.portfolio = await response.json();
            this.updatePortfolioDisplay();
            return this.portfolio;
        } catch (error) {
            console.error('Error getting portfolio:', error);
            return this.portfolio;
        }
    }

    startPerformanceUpdates() {
        this.updateInterval = setInterval(async () => {
            await this.updateAllData();
        }, 5000); // Update every 5 seconds
    }

    stopPerformanceUpdates() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }

    async updateAllData() {
        await this.getPortfolio();
        await this.updatePredictions();
        await this.updatePerformance();
        await this.updateMemoryInsights();
        await this.updateRiskMetrics();
    }

    async updatePredictions() {
        try {
            const response = await fetch('/api/predictions');
            const predictions = await response.json();
            
            if (predictions && predictions.length > 0) {
                this.updatePredictionsTable(predictions);
                if (window.chartManager) {
                    window.chartManager.updateAccuracyChart(predictions);
                    window.chartManager.updatePriceChart(predictions);
                }
            }
        } catch (error) {
            console.error('Error updating predictions:', error);
        }
    }

    async updatePerformance() {
        try {
            const response = await fetch('/api/performance');
            const performance = await response.json();
            
            this.updatePerformanceMetrics(performance);
            
            // Add to performance history
            this.performanceHistory.push({
                timestamp: new Date().toISOString(),
                portfolio_value: this.portfolio.totalValue,
                benchmark_value: performance.benchmark || this.portfolio.totalValue
            });
            
            // Keep only last 100 records
            if (this.performanceHistory.length > 100) {
                this.performanceHistory = this.performanceHistory.slice(-100);
            }
            
            if (window.chartManager) {
                window.chartManager.updatePerformanceChart(this.performanceHistory);
            }
        } catch (error) {
            console.error('Error updating performance:', error);
        }
    }

    async updateMemoryInsights() {
        try {
            const response = await fetch('/api/memory/insights');
            const insights = await response.json();
            
            this.updateInsightsDisplay(insights);
        } catch (error) {
            console.error('Error updating memory insights:', error);
        }
    }

    async updateRiskMetrics() {
        try {
            const response = await fetch('/api/risk/metrics');
            const riskMetrics = await response.json();
            
            if (window.chartManager) {
                window.chartManager.updateRiskChart(riskMetrics);
            }
        } catch (error) {
            console.error('Error updating risk metrics:', error);
        }
    }

    updatePredictionsTable(predictions) {
        const tbody = document.getElementById('predictionsBody');
        if (!tbody) return;

        // Clear existing rows
        tbody.innerHTML = '';

        // Show last 10 predictions
        const recentPredictions = predictions.slice(-10).reverse();

        recentPredictions.forEach(prediction => {
            const row = document.createElement('tr');
            
            // Determine row styling based on correctness
            if (prediction.is_correct !== null) {
                if (prediction.is_correct) {
                    row.className = 'prediction-correct';
                } else {
                    row.className = 'prediction-wrong';
                }
            }

            const time = new Date(prediction.timestamp).toLocaleTimeString();
            const signalClass = prediction.signal.toLowerCase() === 'buy' ? 'signal-buy' : 'signal-sell';
            const confidenceColor = this.getConfidenceColor(prediction.confidence);

            row.innerHTML = `
                <td>${time}</td>
                <td>${prediction.symbol}</td>
                <td class="${signalClass}">${prediction.signal}</td>
                <td>$${prediction.predicted_price?.toFixed(2) || 'N/A'}</td>
                <td>$${prediction.current_price?.toFixed(2) || 'N/A'}</td>
                <td>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${prediction.confidence * 100}%; background: ${confidenceColor};"></div>
                        <span>${(prediction.confidence * 100).toFixed(1)}%</span>
                    </div>
                </td>
                <td>${this.getResultDisplay(prediction)}</td>
            `;

            tbody.appendChild(row);
        });
    }

    getConfidenceColor(confidence) {
        if (confidence >= 0.8) return '#00ff88';
        if (confidence >= 0.6) return '#ffcc00';
        return '#ff4444';
    }

    getResultDisplay(prediction) {
        if (prediction.is_correct === null) {
            return '<span class="result-pending">Pending</span>';
        } else if (prediction.is_correct) {
            return '<span class="result-correct">✓ Correct</span>';
        } else {
            return '<span class="result-wrong">✗ Wrong</span>';
        }
    }

    updatePerformanceMetrics(performance) {
        const elements = {
            totalPredictions: document.getElementById('totalPredictions'),
            accuracy: document.getElementById('accuracy'),
            recentAccuracy: document.getElementById('recentAccuracy'),
            livePnl: document.getElementById('livePnl')
        };

        if (elements.totalPredictions) {
            elements.totalPredictions.textContent = performance.total_predictions || 0;
        }
        
        if (elements.accuracy) {
            const accuracy = (performance.accuracy || 0) * 100;
            elements.accuracy.textContent = `${accuracy.toFixed(1)}%`;
            elements.accuracy.style.color = this.getAccuracyColor(accuracy);
        }
        
        if (elements.recentAccuracy) {
            const recentAccuracy = (performance.recent_accuracy || 0) * 100;
            elements.recentAccuracy.textContent = `${recentAccuracy.toFixed(1)}%`;
            elements.recentAccuracy.style.color = this.getAccuracyColor(recentAccuracy);
        }
        
        if (elements.livePnl) {
            const pnl = performance.live_pnl || 0;
            elements.livePnl.textContent = `$${pnl.toFixed(2)}`;
            elements.livePnl.style.color = pnl >= 0 ? '#00ff88' : '#ff4444';
        }
    }

    getAccuracyColor(accuracy) {
        if (accuracy >= 70) return '#00ff88';
        if (accuracy >= 60) return '#ffcc00';
        return '#ff4444';
    }

    updatePortfolioDisplay() {
        const portfolioElement = document.getElementById('portfolioValue');
        if (portfolioElement) {
            portfolioElement.textContent = `$${this.portfolio.totalValue.toFixed(2)}`;
            portfolioElement.style.color = this.portfolio.totalValue >= 10000 ? '#00ff88' : '#ff4444';
        }
    }

    updateInsightsDisplay(insights) {
        this.updateElementContent('bestSymbols', this.formatBestSymbols(insights.best_performing_symbols));
        this.updateElementContent('recommendations', this.formatRecommendations(insights.recommendations));
        this.updateElementContent('patternRates', this.formatPatternRates(insights.pattern_success_rates));
    }

    updateElementContent(elementId, content) {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = content;
        }
    }

    formatBestSymbols(symbols) {
        if (!symbols || symbols.length === 0) {
            return '<div class="no-data">No symbol data available</div>';
        }

        return symbols.map(symbol => `
            <div class="symbol-item">
                <span class="symbol-name">${symbol.symbol}</span>
                <span class="symbol-accuracy" style="color: ${this.getAccuracyColor(symbol.accuracy * 100)}">
                    ${(symbol.accuracy * 100).toFixed(1)}%
                </span>
                <small class="symbol-trades">(${symbol.total_trades} trades)</small>
            </div>
        `).join('');
    }

    formatRecommendations(recommendations) {
        if (!recommendations || recommendations.length === 0) {
            return '<div class="no-data">No recommendations available</div>';
        }

        return recommendations.map(rec => `
            <div class="recommendation-item">
                <span class="recommendation-bullet">•</span>
                <span class="recommendation-text">${rec}</span>
            </div>
        `).join('');
    }

    formatPatternRates(patternRates) {
        if (!patternRates) {
            return '<div class="no-data">No pattern data available</div>';
        }

        return Object.entries(patternRates)
            .slice(0, 5)
            .map(([pattern, rate]) => `
                <div class="pattern-item">
                    <span class="pattern-name">${pattern}</span>
                    <span class="pattern-rate" style="color: ${this.getAccuracyColor(rate * 100)}">
                        ${(rate * 100).toFixed(1)}%
                    </span>
                </div>
            `).join('');
    }

    showNotification(message, type = 'info') {
        // Remove existing notifications
        const existingNotifications = document.querySelectorAll('.notification');
        existingNotifications.forEach(notification => notification.remove());

        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <span class="notification-message">${message}</span>
                <button class="notification-close" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;

        // Add styles if not already added
        if (!document.querySelector('#notification-styles')) {
            const styles = document.createElement('style');
            styles.id = 'notification-styles';
            styles.textContent = `
                .notification {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    z-index: 1000;
                    animation: slideIn 0.3s ease-out;
                }
                .notification-content {
                    padding: 15px 20px;
                    border-radius: 8px;
                    color: white;
                    font-weight: bold;
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    min-width: 300px;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                }
                .notification-success {
                    background: linear-gradient(45deg, #00ff88, #00ccff);
                }
                .notification-error {
                    background: linear-gradient(45deg, #ff4444, #ff6b6b);
                }
                .notification-info {
                    background: linear-gradient(45deg, #00ccff, #0099cc);
                }
                .notification-close {
                    background: none;
                    border: none;
                    color: white;
                    font-size: 18px;
                    cursor: pointer;
                    margin-left: 10px;
                }
                @keyframes slideIn {
                    from { transform: translateX(100%); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }
            `;
            document.head.appendChild(styles);
        }

        document.body.appendChild(notification);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 5000);
    }

    // Method to run backtest
    async runBacktest() {
        try {
            const symbol = document.getElementById('backtestSymbol')?.value || 'AAPL';
            const startDate = document.getElementById('backtestStartDate')?.value || '2023-01-01';
            const endDate = document.getElementById('backtestEndDate')?.value || '2023-12-31';

            const response = await fetch('/api/backtest/run', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    symbol: symbol,
                    start_date: startDate,
                    end_date: endDate,
                    initial_capital: 10000
                })
            });

            const result = await response.json();
            this.displayBacktestResults(result);
            
        } catch (error) {
            console.error('Error running backtest:', error);
            this.showNotification('Failed to run backtest: ' + error.message, 'error');
        }
    }

    displayBacktestResults(results) {
        // Create or update backtest results modal
        let modal = document.getElementById('backtestResultsModal');
        if (!modal) {
            modal = document.createElement('div');
            modal.id = 'backtestResultsModal';
            modal.className = 'modal';
            modal.innerHTML = `
                <div class="modal-content">
                    <div class="modal-header">
                        <h3>Backtest Results</h3>
                        <button class="modal-close" onclick="this.parentElement.parentElement.parentElement.remove()">×</button>
                    </div>
                    <div class="modal-body" id="backtestResultsContent">
                        <!-- Results will be populated here -->
                    </div>
                </div>
            `;
            document.body.appendChild(modal);
        }

        const content = document.getElementById('backtestResultsContent');
        if (content) {
            content.innerHTML = this.formatBacktestResults(results);
        }

        // Show modal
        modal.style.display = 'block';
    }

    formatBacktestResults(results) {
        if (!results.performance_metrics) {
            return '<div class="error">No results available</div>';
        }

        const metrics = results.performance_metrics;
        
        return `
            <div class="backtest-results">
                <div class="result-grid">
                    <div class="result-item">
                        <label>Total Return</label>
                        <span class="result-value ${metrics.total_return >= 0 ? 'positive' : 'negative'}">
                            ${(metrics.total_return * 100).toFixed(2)}%
                        </span>
                    </div>
                    <div class="result-item">
                        <label>Sharpe Ratio</label>
                        <span class="result-value ${metrics.sharpe_ratio >= 1 ? 'positive' : 'negative'}">
                            ${metrics.sharpe_ratio.toFixed(2)}
                        </span>
                    </div>
                    <div class="result-item">
                        <label>Max Drawdown</label>
                        <span class="result-value negative">
                            ${(metrics.max_drawdown * 100).toFixed(2)}%
                        </span>
                    </div>
                    <div class="result-item">
                        <label>Win Rate</label>
                        <span class="result-value ${metrics.win_rate >= 0.5 ? 'positive' : 'negative'}">
                            ${(metrics.win_rate * 100).toFixed(2)}%
                        </span>
                    </div>
                    <div class="result-item">
                        <label>Total Trades</label>
                        <span class="result-value">${metrics.total_trades}</span>
                    </div>
                    <div class="result-item">
                        <label>Profit Factor</label>
                        <span class="result-value ${metrics.profit_factor >= 1 ? 'positive' : 'negative'}">
                            ${metrics.profit_factor.toFixed(2)}
                        </span>
                    </div>
                </div>
            </div>
        `;
    }
}

// Initialize trading manager
window.tradingManager = new TradingManager();
