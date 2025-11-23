class ChartManager {
    constructor() {
        this.charts = {};
        this.initializeCharts();
    }

    initializeCharts() {
        this.initializeAccuracyChart();
        this.initializePricePredictionChart();
        this.initializePerformanceChart();
        this.initializeRiskChart();
    }

    initializeAccuracyChart() {
        const ctx = document.getElementById('accuracyChart').getContext('2d');
        this.charts.accuracy = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Prediction Accuracy (%)',
                    data: [],
                    borderColor: '#00ff88',
                    backgroundColor: 'rgba(0, 255, 136, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                }, {
                    label: 'Moving Average (7)',
                    data: [],
                    borderColor: '#00ccff',
                    backgroundColor: 'rgba(0, 204, 255, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    borderDash: [5, 5],
                    fill: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Prediction Accuracy Over Time',
                        color: '#ffffff',
                        font: {
                            size: 16
                        }
                    },
                    legend: {
                        labels: {
                            color: '#ffffff',
                            font: {
                                size: 12
                            }
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#00ff88',
                        bodyColor: '#ffffff'
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#ffffff'
                        }
                    },
                    y: {
                        min: 0,
                        max: 100,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#ffffff',
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                },
                interaction: {
                    intersect: false,
                    mode: 'nearest'
                }
            }
        });
    }

    initializePricePredictionChart() {
        const ctx = document.getElementById('priceChart').getContext('2d');
        this.charts.price = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Actual Price',
                    data: [],
                    borderColor: '#00ccff',
                    backgroundColor: 'rgba(0, 204, 255, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                }, {
                    label: 'Predicted Price',
                    data: [],
                    borderColor: '#ff4444',
                    backgroundColor: 'rgba(255, 68, 68, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true,
                    pointBackgroundColor: '#ff4444',
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Price vs Predictions',
                        color: '#ffffff',
                        font: {
                            size: 16
                        }
                    },
                    legend: {
                        labels: {
                            color: '#ffffff',
                            font: {
                                size: 12
                            }
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#00ccff',
                        bodyColor: '#ffffff',
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    label += new Intl.NumberFormat('en-US', {
                                        style: 'currency',
                                        currency: 'USD'
                                    }).format(context.parsed.y);
                                }
                                return label;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#ffffff'
                        }
                    },
                    y: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#ffffff',
                            callback: function(value) {
                                return '$' + value.toFixed(2);
                            }
                        }
                    }
                }
            }
        });
    }

    initializePerformanceChart() {
        const ctx = document.getElementById('performanceChart').getContext('2d');
        this.charts.performance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Portfolio Value',
                    data: [],
                    borderColor: '#00ff88',
                    backgroundColor: 'rgba(0, 255, 136, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                }, {
                    label: 'Benchmark (Buy & Hold)',
                    data: [],
                    borderColor: '#8884d8',
                    backgroundColor: 'rgba(136, 132, 216, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: false,
                    borderDash: [5, 5]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Portfolio Performance',
                        color: '#ffffff',
                        font: {
                            size: 16
                        }
                    },
                    legend: {
                        labels: {
                            color: '#ffffff',
                            font: {
                                size: 12
                            }
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#00ff88',
                        bodyColor: '#ffffff',
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    label += new Intl.NumberFormat('en-US', {
                                        style: 'currency',
                                        currency: 'USD'
                                    }).format(context.parsed.y);
                                }
                                return label;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#ffffff'
                        }
                    },
                    y: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#ffffff',
                            callback: function(value) {
                                return '$' + value.toLocaleString();
                            }
                        }
                    }
                }
            }
        });
    }

    initializeRiskChart() {
        const ctx = document.getElementById('riskChart').getContext('2d');
        this.charts.risk = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Max Drawdown', 'Volatility', 'Sharpe Ratio', 'Win Rate', 'Profit Factor'],
                datasets: [{
                    label: 'Current Values',
                    data: [0, 0, 0, 0, 0],
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.8)',
                        'rgba(54, 162, 235, 0.8)',
                        'rgba(255, 206, 86, 0.8)',
                        'rgba(75, 192, 192, 0.8)',
                        'rgba(153, 102, 255, 0.8)'
                    ],
                    borderColor: [
                        'rgba(255, 99, 132, 1)',
                        'rgba(54, 162, 235, 1)',
                        'rgba(255, 206, 86, 1)',
                        'rgba(75, 192, 192, 1)',
                        'rgba(153, 102, 255, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Risk Metrics',
                        color: '#ffffff',
                        font: {
                            size: 16
                        }
                    },
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                const value = context.parsed.y;
                                switch(context.label) {
                                    case 'Max Drawdown':
                                        label += (value * 100).toFixed(2) + '%';
                                        break;
                                    case 'Volatility':
                                        label += (value * 100).toFixed(2) + '%';
                                        break;
                                    case 'Sharpe Ratio':
                                        label += value.toFixed(2);
                                        break;
                                    case 'Win Rate':
                                        label += (value * 100).toFixed(2) + '%';
                                        break;
                                    case 'Profit Factor':
                                        label += value.toFixed(2);
                                        break;
                                    default:
                                        label += value.toFixed(2);
                                }
                                return label;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#ffffff'
                        }
                    },
                    y: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#ffffff'
                        }
                    }
                }
            }
        });
    }

    updateAccuracyChart(predictions) {
        if (!this.charts.accuracy) return;

        const recentPredictions = predictions.slice(-30);
        const labels = recentPredictions.map(p => 
            new Date(p.timestamp).toLocaleTimeString()
        );
        
        const accuracyData = recentPredictions.map(p => {
            if (p.is_correct !== null) {
                return p.is_correct ? 100 : 0;
            }
            return null;
        }).filter(val => val !== null);

        // Calculate moving average
        const movingAvg = this.calculateMovingAverage(accuracyData, 7);

        this.charts.accuracy.data.labels = labels.slice(-movingAvg.length);
        this.charts.accuracy.data.datasets[0].data = accuracyData.slice(-movingAvg.length);
        this.charts.accuracy.data.datasets[1].data = movingAvg;
        this.charts.accuracy.update();
    }

    updatePriceChart(predictions) {
        if (!this.charts.price) return;

        const recentData = predictions.slice(-20);
        const labels = recentData.map(p => 
            new Date(p.timestamp).toLocaleTimeString()
        );
        
        const actualPrices = recentData.map(p => p.current_price);
        const predictedPrices = recentData.map(p => p.predicted_price);

        this.charts.price.data.labels = labels;
        this.charts.price.data.datasets[0].data = actualPrices;
        this.charts.price.data.datasets[1].data = predictedPrices;
        this.charts.price.update();
    }

    updatePerformanceChart(performanceHistory) {
        if (!this.charts.performance) return;

        const labels = performanceHistory.map(p => 
            new Date(p.timestamp).toLocaleDateString()
        );
        const portfolioValues = performanceHistory.map(p => p.portfolio_value);
        const benchmarkValues = performanceHistory.map(p => p.benchmark_value);

        this.charts.performance.data.labels = labels;
        this.charts.performance.data.datasets[0].data = portfolioValues;
        this.charts.performance.data.datasets[1].data = benchmarkValues;
        this.charts.performance.update();
    }

    updateRiskChart(riskMetrics) {
        if (!this.charts.risk) return;

        this.charts.risk.data.datasets[0].data = [
            riskMetrics.max_drawdown || 0,
            riskMetrics.volatility || 0,
            riskMetrics.sharpe_ratio || 0,
            riskMetrics.win_rate || 0,
            riskMetrics.profit_factor || 0
        ];
        this.charts.risk.update();
    }

    calculateMovingAverage(data, window) {
        const movingAvg = [];
        for (let i = 0; i <= data.length - window; i++) {
            const windowSlice = data.slice(i, i + window);
            const avg = windowSlice.reduce((a, b) => a + b, 0) / window;
            movingAvg.push(avg);
        }
        return movingAvg;
    }

    destroyCharts() {
        Object.values(this.charts).forEach(chart => {
            if (chart) {
                chart.destroy();
            }
        });
        this.charts = {};
    }
}

// Initialize chart manager
window.chartManager = new ChartManager();
