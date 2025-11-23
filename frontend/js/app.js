class TradingDashboard {
    constructor() {
        this.apiBase = 'http://localhost:5000/api';
        this.isTrading = false;
        this.predictions = [];
        this.charts = {};
        
        this.initializeEventListeners();
        this.startDataPolling();
    }
    
    initializeEventListeners() {
        document.getElementById('toggleTrading').addEventListener('click', () => {
            this.toggleTrading();
        });
    }
    
    async toggleTrading() {
        try {
            const response = await fetch(`${this.apiBase}/start_trading`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ demo_mode: true })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                this.isTrading = true;
                this.updateStatus(true);
                this.showNotification('Trading started successfully!', 'success');
            }
        } catch (error) {
            console.error('Error starting trading:', error);
            this.showNotification('Error starting trading', 'error');
        }
    }
    
    updateStatus(isOnline) {
        const statusElement = document.getElementById('status');
        const buttonElement = document.getElementById('toggleTrading');
        
        if (isOnline) {
            statusElement.textContent = 'Live Trading';
            statusElement.className = 'status-online';
            buttonElement.textContent = 'Stop Trading';
            buttonElement.style.background = 'linear-gradient(45deg, #ff4444, #ff6b6b)';
        } else {
            statusElement.textContent = 'Offline';
            statusElement.className = 'status-offline';
            buttonElement.textContent = 'Start Trading';
            buttonElement.style.background = 'linear-gradient(45deg, #00ff88, #00ccff)';
        }
    }
    
    async startDataPolling() {
        setInterval(() => {
            this.fetchPredictions();
            this.fetchPerformance();
            this.fetchMemoryInsights();
        }, 2000); // Poll every 2 seconds
    }
    
    async fetchPredictions() {
        try {
            const response = await fetch(`${this.apiBase}/predictions`);
            const predictions = await response.json();
            
            if (predictions && predictions.length > 0) {
                this.predictions = predictions;
                this.updatePredictionsTable();
                this.updateCharts();
            }
        } catch (error) {
            console.error('Error fetching predictions:', error);
        }
    }
    
    async fetchPerformance() {
        try {
            const response = await fetch(`${this.apiBase}/performance`);
            const performance = await response.json();
            
            this.updatePerformanceMetrics(performance);
        } catch (error) {
            console.error('Error fetching performance:', error);
        }
    }
    
    async fetchMemoryInsights() {
        try {
            const response = await fetch(`${this.apiBase}/memory_insights`);
            const insights = await response.json();
            
            this.updateMemoryInsights(insights);
        } catch (error) {
            console.error('Error fetching insights:', error);
        }
    }
    
    updatePerformanceMetrics(performance) {
        document.getElementById('totalPredictions').textContent = performance.total_predictions;
        document.getElementById('accuracy').textContent = `${(performance.accuracy * 100).toFixed(1)}%`;
        document.getElementById('recentAccuracy').textContent = `${(performance.recent_accuracy * 100).toFixed(1)}%`;
        
        // Calculate simulated P&L
        const pnl = performance.correct_predictions * 10 - (performance.total_predictions - performance.correct_predictions) * 5;
        document.getElementById('livePnl').textContent = `$${pnl.toFixed(2)}`;
    }
    
    updatePredictionsTable() {
        const tbody = document.getElementById('predictionsBody');
        tbody.innerHTML = '';
        
        // Show last 10 predictions
        const recentPredictions = this.predictions.slice(-10).reverse();
        
        recentPredictions.forEach(prediction => {
            const row = document.createElement('tr');
            
            // Determine row class based on correctness
            if (prediction.is_correct !== null) {
                row.className = prediction.is_correct ? 'prediction-correct' : 'prediction-wrong';
            }
            
            const time = new Date(prediction.timestamp).toLocaleTimeString();
            const signalClass = prediction.signal.toLowerCase() === 'buy' ? 'signal-buy' : 'signal-sell';
            
            row.innerHTML = `
                <td>${time}</td>
                <td>${prediction.symbol}</td>
                <td class="${signalClass}">${prediction.signal}</td>
                <td>$${prediction.predicted_price?.toFixed(2) || 'N/A'}</td>
                <td>$${prediction.current_price?.toFixed(2) || 'N/A'}</td>
                <td>${(prediction.confidence * 100).toFixed(1)}%</td>
                <td>${this.getResultDisplay(prediction)}</td>
            `;
            
            tbody.appendChild(row);
        });
    }
    
    getResultDisplay(prediction) {
        if (prediction.is_correct === null) {
            return '<span style="color: #888;">Pending</span>';
        } else if (prediction.is_correct) {
            return '<span style="color: #00ff88;">✓ Correct</span>';
        } else {
            return '<span style="color: #ff4444;">✗ Wrong</span>';
        }
    }
    
    updateMemoryInsights(insights) {
        // Update best symbols
        const bestSymbolsElement = document.getElementById('bestSymbols');
        if (insights.best_performing_symbols && insights.best_performing_symbols.length > 0) {
            bestSymbolsElement.innerHTML = insights.best_performing_symbols
                .map(symbol => `
                    <div style="margin-bottom: 8px;">
                        <strong>${symbol.symbol}</strong>: ${(symbol.accuracy * 100).toFixed(1)}% 
                        <small>(${symbol.total_trades} trades)</small>
                    </div>
                `).join('');
        }
        
        // Update recommendations
        const recommendationsElement = document.getElementById('recommendations');
        if (insights.recommendations && insights.recommendations.length > 0) {
            recommendationsElement.innerHTML = insights.recommendations
                .map(rec => `<div style="margin-bottom: 8px;">• ${rec}</div>`)
                .join('');
        }
        
        // Update pattern rates
        const patternRatesElement = document.getElementById('patternRates');
        if (insights.pattern_success_rates) {
            patternRatesElement.innerHTML = Object.entries(insights.pattern_success_rates)
                .slice(0, 5)
                .map(([pattern, rate]) => `
                    <div style="margin-bottom: 6px;">
                        ${pattern}: <strong>${(rate * 100).toFixed(1)}%</strong>
                    </div>
                `).join('');
        }
    }
    
    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 8px;
            color: white;
            font-weight: bold;
            z-index: 1000;
            transition: all 0.3s ease;
            ${type === 'success' ? 'background: linear-gradient(45deg, #00ff88, #00ccff);' : ''}
            ${type === 'error' ? 'background: linear-gradient(45deg, #ff4444, #ff6b6b);' : ''}
            ${type === 'info' ? 'background: linear-gradient(45deg, #00ccff, #0099cc);' : ''}
        `;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        // Remove notification after 3 seconds
        setTimeout(() => {
            notification.style.opacity = '0';
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 3000);
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.tradingDashboard = new TradingDashboard();
});
