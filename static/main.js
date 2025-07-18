document.addEventListener('DOMContentLoaded', () => {
    // Initialize form and result elements
    const form = document.getElementById('prediction-form');
    const resultDiv = document.getElementById('prediction-result');
    const errorDiv = document.getElementById('error-message');
    const metricsChartCanvas = document.getElementById('metrics-chart');
    const featureChartCanvas = document.getElementById('feature-importance-chart');
    const clusterChartCanvas = document.getElementById('cluster-chart');
    const trendChartCanvas = document.getElementById('trend-chart');
    const clusterPieChartCanvas = document.getElementById('cluster-pie-chart');
    const clusterTableBody = document.getElementById('cluster-table-body');

    // Handle form submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        errorDiv.textContent = '';

        const formData = new FormData(form);
        const data = {};
        formData.forEach((value, key) => {
            data[key] = value;
        });

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            const result = await response.json();

            if (response.ok) {
                // Display predictions and cluster assignment
                resultDiv.innerHTML = `
                    <h3>Prediction Results</h3>
                    <p>Random Forest Prediction: ${result.prediction_rf.toFixed(2)}</p>
                    <p>Linear Regression Prediction: ${result.prediction_lr.toFixed(2)}</p>
                    <p>Ensemble Prediction: ${result.prediction_ensemble.toFixed(2)}</p>
                    <p>Cluster Assignment: Cluster ${result.cluster_assignment}</p>
                `;

                // Update cluster characteristics table
                clusterTableBody.innerHTML = '';
                result.cluster_characteristics.forEach((cluster, index) => {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td>Cluster ${index}</td>
                        <td>${cluster.age.toFixed(2)}</td>
                        <td>${cluster.studytime.toFixed(2)}</td>
                        <td>${cluster.failures.toFixed(2)}</td>
                        <td>${cluster.absences.toFixed(2)}</td>
                        <td>${cluster.G1.toFixed(2)}</td>
                        <td>${cluster.G2.toFixed(2)}</td>
                        <td>${cluster.average_grade.toFixed(2)}</td>
                    `;
                    clusterTableBody.appendChild(row);
                });

                // Plot model performance metrics
                new Chart(metricsChartCanvas, {
                    type: 'bar',
                    data: {
                        labels: ['Random Forest', 'Linear Regression'],
                        datasets: [
                            {
                                label: 'MAE (Test)',
                                data: [result.mae_rf, result.mae_lr],
                                backgroundColor: 'rgba(75, 192, 192, 0.5)',
                                borderColor: 'rgba(75, 192, 192, 1)',
                                borderWidth: 1
                            },
                            {
                                label: 'RMSE (Test)',
                                data: [result.rmse_rf, result.rmse_lr],
                                backgroundColor: 'rgba(255, 99, 132, 0.5)',
                                borderColor: 'rgba(255, 99, 132, 1)',
                                borderWidth: 1
                            }
                        ]
                    },
                    options: {
                        scales: {
                            y: { beginAtZero: true, title: { display: true, text: 'Error Value' } },
                            x: { title: { display: true, text: 'Model' } }
                        },
                        plugins: { title: { display: true, text: 'Model Performance Metrics' } }
                    }
                });

                // Plot feature importances
                const rfFeatures = Object.entries(result.rf_feature_importances)
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 10);
                new Chart(featureChartCanvas, {
                    type: 'bar',
                    data: {
                        labels: rfFeatures.map(f => f[0]),
                        datasets: [{
                            label: 'Feature Importance',
                            data: rfFeatures.map(f => f[1]),
                            backgroundColor: 'rgba(54, 162, 235, 0.5)',
                            borderColor: 'rgba(54, 162, 235, 1)',
                            borderWidth: 1
                        }]
                    },
                    options: {
                        indexAxis: 'y',
                        scales: {
                            x: { beginAtZero: true, title: { display: true, text: 'Importance' } },
                            y: { title: { display: true, text: 'Feature' } }
                        },
                        plugins: { title: { display: true, text: 'Top 10 Feature Importances (Random Forest)' } }
                    }
                });

                // Plot cluster scatter plot
                const clusterData = result.cluster_data;
                const scatterData = {
                    datasets: Array.from({ length: 4 }, (_, i) => {
                        const clusterPoints = clusterData.features
                            .map((point, idx) => clusterData.labels[idx] === i ? { x: point[0], y: point[1] } : null)
                            .filter(d => d);
                        // Calculate centroid for the cluster
                        const centroid = clusterPoints.length ? {
                            x: clusterPoints.reduce((sum, p) => sum + p.x, 0) / clusterPoints.length,
                            y: clusterPoints.reduce((sum, p) => sum + p.y, 0) / clusterPoints.length
                        } : { x: 0, y: 0 };
                        return {
                            label: `Cluster ${i}`,
                            data: clusterPoints,
                            backgroundColor: `rgba(${[75, 192, 192, 255, 99, 132, 54, 162, 235, 255, 206, 86][i * 3]}, ${[75, 192, 192, 255, 99, 132, 54, 162, 235, 255, 206, 86][i * 3 + 1]}, ${[75, 192, 192, 255, 99, 132, 54, 162, 235, 255, 206, 86][i * 3 + 2]}, 0.6)`,
                            pointRadius: 5,
                            pointHoverRadius: 7,
                            borderWidth: 1
                        };
                    }).concat(
                        Array.from({ length: 4 }, (_, i) => ({
                            label: `Centroid Cluster ${i}`,
                            data: [{ x: clusterData.features.filter((_, idx) => clusterData.labels[idx] === i).reduce((sum, p) => sum + p[0], 0) / (clusterData.labels.filter(l => l === i).length || 1), y: clusterData.features.filter((_, idx) => clusterData.labels[idx] === i).reduce((sum, p) => sum + p[1], 0) / (clusterData.labels.filter(l => l === i).length || 1) }],
                            backgroundColor: `rgba(${[75, 192, 192, 255, 99, 132, 54, 162, 235, 255, 206, 86][i * 3]}, ${[75, 192, 192, 255, 99, 132, 54, 162, 235, 255, 206, 86][i * 3 + 1]}, ${[75, 192, 192, 255, 99, 132, 54, 162, 235, 255, 206, 86][i * 3 + 2]}, 1)`,
                            pointRadius: 8,
                            pointStyle: 'star',
                            showLine: false
                        }))
                    )
                };
                new Chart(clusterChartCanvas, {
                    type: 'scatter',
                    data: scatterData,
                    options: {
                        scales: {
                            x: { 
                                title: { display: true, text: 'First Term Grade (G1)' },
                                grid: { color: 'rgba(0, 0, 0, 0.1)' }
                            },
                            y: { 
                                title: { display: true, text: 'Study Time' },
                                grid: { color: 'rgba(0, 0, 0, 0.1)' }
                            }
                        },
                        plugins: { 
                            title: { display: true, text: 'Student Clusters (G1 vs Study Time)' },
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        return `G1: ${context.raw.x.toFixed(2)}, Study Time: ${context.raw.y.toFixed(2)}`;
                                    }
                                }
                            }
                        },
                        maintainAspectRatio: false,
                        responsive: true
                    }
                });

                // Plot actual vs predicted grades (scatter plot)
                const predData = result.test_grades;
                const trendData = {
                    datasets: [
                        {
                            label: 'Actual vs RF Prediction',
                            data: predData.G3_actual.map((actual, i) => ({ x: actual, y: predData.G3_pred_rf[i] })),
                            backgroundColor: 'rgba(0, 128, 0, 0.6)', // Green for RF
                            pointRadius: 5,
                            pointHoverRadius: 7
                        },
                        {
                            label: 'Actual vs LR Prediction',
                            data: predData.G3_actual.map((actual, i) => ({ x: actual, y: predData.G3_pred_lr[i] })),
                            backgroundColor: 'rgba(153, 102, 255, 0.6)', // Purple for LR
                            pointRadius: 5,
                            pointHoverRadius: 7
                        },
                        {
                            label: 'Ideal Prediction (y=x)',
                            data: predData.G3_actual.map(actual => ({ x: actual, y: actual })),
                            backgroundColor: 'rgba(255, 99, 132, 0.3)', // Light red for ideal line
                            pointRadius: 0,
                            borderColor: 'rgba(255, 99, 132, 1)',
                            borderWidth: 2,
                            showLine: true,
                            fill: false
                        }
                    ]
                };
                new Chart(trendChartCanvas, {
                    type: 'scatter',
                    data: trendData,
                    options: {
                        scales: {
                            x: { 
                                title: { display: true, text: 'Actual Grade (G3)' },
                                min: 0,
                                max: 20,
                                grid: { color: 'rgba(0, 0, 0, 0.1)' }
                            },
                            y: { 
                                title: { display: true, text: 'Predicted Grade' },
                                min: 0,
                                max: 20,
                                grid: { color: 'rgba(0, 0, 0, 0.1)' }
                            }
                        },
                        plugins: { 
                            title: { display: true, text: 'Actual vs Predicted Grades (Scatter)' },
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        return `${context.dataset.label}: Actual = ${context.raw.x.toFixed(2)}, Predicted = ${context.raw.y.toFixed(2)}`;
                                    }
                                }
                            }
                        },
                        maintainAspectRatio: false,
                        responsive: true
                    }
                });

                // Plot cluster distribution pie chart
                const clusterCounts = Array(4).fill(0);
                result.cluster_data.labels.forEach(label => clusterCounts[label]++);
                new Chart(clusterPieChartCanvas, {
                    type: 'pie',
                    data: {
                        labels: ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3'],
                        datasets: [{
                            data: clusterCounts,
                            backgroundColor: ['rgba(75, 192, 192, 0.6)', 'rgba(255, 99, 132, 0.6)', 'rgba(54, 162, 235, 0.6)', 'rgba(255, 206, 86, 0.6)']
                        }]
                    },
                    options: {
                        plugins: { title: { display: true, text: 'Cluster Distribution' } }
                    }
                });
            } else {
                errorDiv.textContent = result.error;
            }
        } catch (error) {
            errorDiv.textContent = 'Error: Could not connect to the server';
        }
    });
});