document.addEventListener('DOMContentLoaded', () => {
    const stateSelect = document.getElementById('stateSelect');
    const generateBtn = document.getElementById('generateBtn');
    const btnText = document.querySelector('.btn-text');
    const loader = document.querySelector('.loader');
    
    const bestModelValue = document.getElementById('bestModelValue');
    const modelVersionValue = document.getElementById('modelVersionValue');
    const chartBadge = document.getElementById('chartBadge');
    
    const week1Val = document.getElementById('week1Val');
    const week8Val = document.getElementById('week8Val');
    const cumulativeVal = document.getElementById('cumulativeVal');

    let forecastChart = null;

    // Format currency
    const formatCurrency = (val) => {
        if (val >= 1e9) {
            return '$' + (val / 1e9).toFixed(2) + 'B';
        }
        if (val >= 1e6) {
            return '$' + (val / 1e6).toFixed(2) + 'M';
        }
        if (val >= 1e3) {
            return '$' + (val / 1e3).toFixed(2) + 'K';
        }
        return '$' + val.toFixed(2);
    };

    // Initialize Chart
    const initChart = () => {
        const ctx = document.getElementById('forecastChart').getContext('2d');
        
        // Gradient for line
        const gradient = ctx.createLinearGradient(0, 0, 0, 400);
        gradient.addColorStop(0, 'rgba(99, 102, 241, 0.8)');
        gradient.addColorStop(1, 'rgba(99, 102, 241, 0.0)');

        Chart.defaults.color = '#8a8a9e';
        Chart.defaults.font.family = "'Outfit', sans-serif";

        forecastChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6', 'Week 7', 'Week 8'],
                datasets: [{
                    label: 'Projected Sales',
                    data: [0, 0, 0, 0, 0, 0, 0, 0],
                    borderColor: '#a855f7',
                    backgroundColor: gradient,
                    borderWidth: 3,
                    pointBackgroundColor: '#fff',
                    pointBorderColor: '#a855f7',
                    pointBorderWidth: 2,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(20, 20, 25, 0.9)',
                        titleColor: '#fff',
                        bodyColor: '#fff',
                        borderColor: 'rgba(255,255,255,0.1)',
                        borderWidth: 1,
                        padding: 12,
                        displayColors: false,
                        callbacks: {
                            label: function(context) {
                                return 'Sales: ' + formatCurrency(context.raw);
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)',
                            drawBorder: false
                        },
                        ticks: {
                            callback: function(value) {
                                return formatCurrency(value);
                            }
                        }
                    },
                    x: {
                        grid: {
                            display: false,
                            drawBorder: false
                        }
                    }
                },
                animation: {
                    duration: 2000,
                    easing: 'easeOutQuart'
                }
            }
        });
    };

    // Load states
    const loadStates = async () => {
        try {
            const response = await fetch('/states');
            if (!response.ok) throw new Error('Failed to load states');
            const data = await response.json();
            
            stateSelect.innerHTML = '';
            data.states.forEach(state => {
                const option = document.createElement('option');
                option.value = state;
                option.textContent = state;
                stateSelect.appendChild(option);
            });
            
            stateSelect.disabled = false;
            generateBtn.disabled = false;
        } catch (err) {
            console.error(err);
            stateSelect.innerHTML = '<option>Error loading states</option>';
        }
    };

    // Fetch forecast
    const fetchForecast = async () => {
        const state = stateSelect.value;
        if (!state) return;

        // UI Loading state
        generateBtn.disabled = true;
        btnText.classList.add('hidden');
        loader.classList.remove('hidden');
        stateSelect.disabled = true;

        try {
            const response = await fetch(`/predict?state=${encodeURIComponent(state)}`);
            if (!response.ok) throw new Error('Failed to fetch forecast');
            const data = await response.json();

            // Update UI Stats
            bestModelValue.textContent = data.best_model;
            modelVersionValue.textContent = data.model_version;
            chartBadge.textContent = `${data.state} Forecast`;
            chartBadge.style.background = 'rgba(168, 85, 247, 0.2)';
            chartBadge.style.color = '#c084fc';

            const values = data.forecast;
            week1Val.textContent = formatCurrency(values[0]);
            week8Val.textContent = formatCurrency(values[7]);
            
            const total = values.reduce((a, b) => a + b, 0);
            cumulativeVal.textContent = formatCurrency(total);

            // Update Chart
            forecastChart.data.datasets[0].data = values;
            forecastChart.update();

        } catch (err) {
            console.error(err);
            alert('Error generating forecast. Please try again.');
        } finally {
            // Restore UI
            generateBtn.disabled = false;
            btnText.classList.remove('hidden');
            loader.classList.add('hidden');
            stateSelect.disabled = false;
        }
    };

    // Event Listeners
    generateBtn.addEventListener('click', fetchForecast);
    stateSelect.addEventListener('change', fetchForecast);

    // Init
    initChart();
    loadStates();
});
