document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predict-form');
    if (!form) {
        console.error('Form element not found');
        return;
    }

    form.addEventListener('submit', async (event) => {
        event.preventDefault();

        const formData = new FormData(event.target);
        const data = Object.fromEntries(formData);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (response.ok) {
                const resultDiv = document.getElementById('result');
                const predictionRf = document.getElementById('prediction-rf');
                const predictionLr = document.getElementById('prediction-lr');
                const predictionEnsemble = document.getElementById('prediction-ensemble');
                const metrics = document.getElementById('metrics');

                if (resultDiv && predictionRf && predictionLr && predictionEnsemble && metrics) {
                    predictionRf.textContent = `Random Forest: ${result.prediction_rf}`;
                    predictionLr.textContent = `Linear Regression: ${result.prediction_lr}`;
                    predictionEnsemble.textContent = `Ensemble Prediction: ${result.prediction_ensemble}`;
                    metrics.textContent = `Test Metrics - RF: MAE ${result.mae_rf}, RMSE ${result.rmse_rf}; LR: MAE ${result.mae_lr}, RMSE ${result.rmse_lr}\nCV Metrics - RF: MAE ${result.mae_rf_cv}, LR: MAE ${result.mae_lr_cv}`;
                    resultDiv.classList.remove('hidden');
                } else {
                    console.error('One or more result elements not found:', { resultDiv, predictionRf, predictionLr, predictionEnsemble, metrics });
                }

                // Feature Importance/Coefficients Chart
                const featureCtx = document.getElementById('featureChart')?.getContext('2d');
                if (featureCtx) {
                    if (window.featureChart instanceof Chart) {
                        window.featureChart.destroy();
                        window.featureChart = null; // Reset after destruction
                    }
                    window.featureChart = new Chart(featureCtx, {
                        type: 'bar',
                        data: {
                            labels: Object.entries(result.rf_feature_importances)
                                .sort((a, b) => b[1] - a[1])
                                .slice(0, 10)
                                .map(x => x[0]),
                            datasets: [
                                { 
                                    label: 'RF Importance', 
                                    data: Object.entries(result.rf_feature_importances)
                                        .sort((a, b) => b[1] - a[1])
                                        .slice(0, 10)
                                        .map(x => x[1]), 
                                    backgroundColor: '#1f77b4', 
                                    yAxisID: 'y1' 
                                },
                                { 
                                    label: 'LR Coefficient', 
                                    data: Object.entries(result.lr_coefficients)
                                        .filter(([key]) => Object.entries(result.rf_feature_importances)
                                            .sort((a, b) => b[1] - a[1])
                                            .slice(0, 10)
                                            .map(x => x[0])
                                            .includes(key))
                                        .sort((a, b) => result.rf_feature_importances[b[0]] - result.rf_feature_importances[a[0]])
                                        .map(x => x[1]), 
                                    backgroundColor: '#ff7f0e', 
                                    yAxisID: 'y2' 
                                }
                            ]
                        },
                        options: {
                            responsive: true,
                            plugins: { tooltip: { enabled: true }, legend: { position: 'top' } },
                            scales: {
                                y1: { position: 'left', title: { display: true, text: 'RF Importance', font: { size: 14 } }, beginAtZero: true },
                                y2: { position: 'right', title: { display: true, text: 'LR Coefficient', font: { size: 14 } }, grid: { display: false } }
                            }
                        }
                    });
                } else {
                    console.error('Feature chart context not found');
                }

                // Actual vs. Predicted Scatter Plot
                const actualVsPredictedCtx = document.getElementById('actualVsPredictedChart')?.getContext('2d');
                if (actualVsPredictedCtx) {
                    if (window.actualVsPredictedChart instanceof Chart) {
                        window.actualVsPredictedChart.destroy();
                        window.actualVsPredictedChart = null;
                    }
                    window.actualVsPredictedChart = new Chart(actualVsPredictedCtx, {
                        type: 'scatter',
                        data: {
                            datasets: [
                                {
                                    label: 'RF Predictions',
                                    data: result.actual_vs_predicted_rf.map(([a, p]) => ({ x: a, y: p })),
                                    backgroundColor: 'rgba(31, 119, 180, 0.6)',
                                    pointRadius: 5
                                },
                                {
                                    label: 'LR Predictions',
                                    data: result.actual_vs_predicted_lr.map(([a, p]) => ({ x: a, y: p })),
                                    backgroundColor: 'rgba(255, 127, 14, 0.6)',
                                    pointRadius: 5
                                },
                                {
                                    label: 'New RF Prediction',
                                    data: [{ x: result.prediction_rf, y: result.prediction_rf }],
                                    backgroundColor: 'blue',
                                    pointRadius: 8,
                                    pointStyle: 'star'
                                },
                                {
                                    label: 'New LR Prediction',
                                    data: [{ x: result.prediction_lr, y: result.prediction_lr }],
                                    backgroundColor: 'orange',
                                    pointRadius: 8,
                                    pointStyle: 'star'
                                },
                                {
                                    label: 'Perfect Prediction',
                                    data: [{ x: 0, y: 0 }, { x: 20, y: 20 }],
                                    type: 'line',
                                    borderColor: 'red',
                                    borderWidth: 2,
                                    fill: false,
                                    pointRadius: 0
                                }
                            ]
                        },
                        options: {
                            responsive: true,
                            plugins: { tooltip: { enabled: true }, legend: { position: 'top' } },
                            scales: {
                                x: { title: { display: true, text: 'Actual G3', font: { size: 14 } }, min: 0, max: 20 },
                                y: { title: { display: true, text: 'Predicted G3', font: { size: 14 } }, min: 0, max: 20 }
                            }
                        }
                    });
                }

                // Error Distribution Histogram with Normal Curve
                const errorDistributionCtx = document.getElementById('errorDistributionChart')?.getContext('2d');
                if (errorDistributionCtx) {
                    if (window.errorDistributionChart instanceof Chart) {
                        window.errorDistributionChart.destroy();
                        window.errorDistributionChart = null;
                    }
                    const errors_rf = result.errors_rf;
                    const errors_lr = result.errors_lr;
                    const bins = 20;
                    const minError = Math.min(...errors_rf, ...errors_lr);
                    const maxError = Math.max(...errors_rf, ...errors_lr);
                    const binWidth = (maxError - minError) / bins;
                    const binEdges = Array.from({ length: bins + 1 }, (_, i) => minError + i * binWidth);
                    const histDataRf = Array(bins).fill(0);
                    const histDataLr = Array(bins).fill(0);
                    errors_rf.forEach(error => {
                        const binIndex = Math.min(Math.floor((error - minError) / binWidth), bins - 1);
                        histDataRf[binIndex]++;
                    });
                    errors_lr.forEach(error => {
                        const binIndex = Math.min(Math.floor((error - minError) / binWidth), bins - 1);
                        histDataLr[binIndex]++;
                    });

                    const meanRf = errors_rf.reduce((a, b) => a + b, 0) / errors_rf.length;
                    const stdRf = Math.sqrt(errors_rf.reduce((a, b) => a + Math.pow(b - meanRf, 2), 0) / errors_rf.length);
                    const normalCurveRf = binEdges.slice(0, -1).map(x => (1 / (stdRf * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - meanRf) / stdRf, 2)) * errors_rf.length * binWidth);

                    const meanLr = errors_lr.reduce((a, b) => a + b, 0) / errors_lr.length;
                    const stdLr = Math.sqrt(errors_lr.reduce((a, b) => a + Math.pow(b - meanLr, 2), 0) / errors_lr.length);
                    const normalCurveLr = binEdges.slice(0, -1).map(x => (1 / (stdLr * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - meanLr) / stdLr, 2)) * errors_lr.length * binWidth);

                    window.errorDistributionChart = new Chart(errorDistributionCtx, {
                        type: 'bar',
                        data: {
                            labels: binEdges.slice(0, -1).map((edge, i) => `${edge.toFixed(1)} to ${(edge + binWidth).toFixed(1)}`),
                            datasets: [
                                { label: 'RF Errors', data: histDataRf, backgroundColor: 'rgba(31, 119, 180, 0.6)', barPercentage: 0.4 },
                                { label: 'LR Errors', data: histDataLr, backgroundColor: 'rgba(255, 127, 14, 0.6)', barPercentage: 0.4 },
                                { label: 'RF Normal Fit', data: normalCurveRf, type: 'line', borderColor: '#1f77b4', fill: false, tension: 0.1 },
                                { label: 'LR Normal Fit', data: normalCurveLr, type: 'line', borderColor: '#ff7f0e', fill: false, tension: 0.1 }
                            ]
                        },
                        options: {
                            responsive: true,
                            plugins: { tooltip: { enabled: true }, legend: { position: 'top' } },
                            scales: {
                                x: { title: { display: true, text: 'Error (Actual - Predicted)', font: { size: 14 } } },
                                y: { title: { display: true, text: 'Frequency/Density', font: { size: 14 } }, beginAtZero: true }
                            }
                        }
                    });
                }
            } else {
                alert(`Error: ${result.error}`);
            }
        } catch (error) {
            console.error('Fetch error:', error);
            alert(`Error: ${error.message}`);
        }
    });
});
