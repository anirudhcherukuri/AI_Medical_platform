/**
 * Advanced AI Medical Intelligence Platform - Frontend Client Application
 */

document.addEventListener('DOMContentLoaded', () => {
    // -------------------------------------------------------------------------
    // 1. TAB NAVIGATION
    // -------------------------------------------------------------------------
    const navButtons = document.querySelectorAll('.nav-btn[data-target]');
    const tabContents = document.querySelectorAll('.tab-content');

    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetId = btn.getAttribute('data-target');
            
            navButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(t => t.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(targetId).classList.add('active');

            if (targetId === 'history-tab') {
                loadHistoryRecords();
            } else if (targetId === 'analytics-tab') {
                loadAnalyticsData();
            }
        });
    });

    // -------------------------------------------------------------------------
    // 2. DIAGNOSTIC STUDIO - UPLOAD & FORM MANAGEMENT
    // -------------------------------------------------------------------------
    const dropZone = document.getElementById('drop-zone');
    const imageInput = document.getElementById('image-input');
    const dropPrompt = document.getElementById('drop-zone-prompt');
    const previewWrapper = document.getElementById('preview-wrapper');
    const imagePreview = document.getElementById('image-preview');
    const btnRemoveImage = document.getElementById('btn-remove-image');
    const form = document.getElementById('diagnostic-form');

    let selectedFile = null;

    // Drag & Drop handlers
    dropZone.addEventListener('click', () => imageInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileSelection(e.dataTransfer.files[0]);
        }
    });

    imageInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelection(e.target.files[0]);
        }
    });

    function handleFileSelection(file) {
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropPrompt.classList.add('hidden');
            previewWrapper.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }

    btnRemoveImage.addEventListener('click', (e) => {
        e.stopPropagation();
        selectedFile = null;
        imageInput.value = '';
        imagePreview.src = '';
        previewWrapper.classList.add('hidden');
        dropPrompt.classList.remove('hidden');
    });

    // Preset Sample Image Handlers
    const sampleButtons = document.querySelectorAll('.btn-sample');
    sampleButtons.forEach(btn => {
        btn.addEventListener('click', async () => {
            const sampleType = btn.getAttribute('data-sample');
            try {
                const response = await fetch(`/storage/samples/sample_${sampleType}.jpg`);
                if (!response.ok) {
                    alert(`Sample image for ${sampleType} not initialized yet. Run train_and_setup.py first.`);
                    return;
                }
                const blob = await response.blob();
                const file = new File([blob], `sample_${sampleType}.jpg`, { type: "image/jpeg" });
                handleFileSelection(file);

                // Auto populate metadata
                if (sampleType === 'normal') {
                    document.getElementById('patient-name').value = "John Doe";
                    document.getElementById('patient-id').value = "P-NOR-102";
                } else if (sampleType === 'pneumonia') {
                    document.getElementById('patient-name').value = "Robert Chen";
                    document.getElementById('patient-id').value = "P-PNE-409";
                } else {
                    document.getElementById('patient-name').value = "Maria Garcia";
                    document.getElementById('patient-id').value = "P-COV-991";
                }
            } catch (err) {
                console.error("Error loading sample image:", err);
            }
        });
    });

    // Opacity Slider for Grad-CAM Fused Overlay
    const opacitySlider = document.getElementById('heatmap-opacity');
    const opacityValText = document.getElementById('opacity-val');
    const resOverlayImg = document.getElementById('res-overlay-img');

    opacitySlider.addEventListener('input', (e) => {
        const val = e.target.value;
        opacityValText.textContent = `${Math.round(val * 100)}%`;
        if (resOverlayImg) {
            resOverlayImg.style.opacity = val;
        }
    });

    // -------------------------------------------------------------------------
    // GLASSMORPHISM WARNING MODAL LOGIC
    // -------------------------------------------------------------------------
    const warningModal = document.getElementById('unsupported-image-modal');
    const btnCloseModal = document.getElementById('btn-close-modal');
    const btnModalDismiss = document.getElementById('btn-modal-dismiss');
    const modalReasonText = document.getElementById('modal-reason-text');

    function showUnsupportedModal(reasonDetail) {
        if (modalReasonText) {
            modalReasonText.textContent = reasonDetail ? `Reason: ${reasonDetail}` : "Prediction has been cancelled.";
        }
        if (warningModal) {
            warningModal.classList.remove('hidden');
        }
    }

    function hideUnsupportedModal() {
        if (warningModal) {
            warningModal.classList.add('hidden');
        }
    }

    if (btnCloseModal) btnCloseModal.addEventListener('click', hideUnsupportedModal);
    if (btnModalDismiss) btnModalDismiss.addEventListener('click', hideUnsupportedModal);
    if (warningModal) {
        warningModal.addEventListener('click', (e) => {
            if (e.target === warningModal) hideUnsupportedModal();
        });
    }

    // Form Submission & API Trigger
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (!selectedFile) {
            showUnsupportedModal("Please select or drop a Chest X-Ray image first.");
            return;
        }

        const emptyState = document.getElementById('results-empty-state');
        const loader = document.getElementById('results-loader');
        const resultsContent = document.getElementById('results-content');

        // Show Loader
        emptyState.classList.add('hidden');
        resultsContent.classList.add('hidden');
        loader.classList.remove('hidden');

        const formData = new FormData();
        formData.append("file", selectedFile);
        formData.append("patient_id", document.getElementById('patient-id').value);
        formData.append("patient_name", document.getElementById('patient-name').value);
        formData.append("age", document.getElementById('patient-age').value);
        formData.append("gender", document.getElementById('patient-gender').value);
        formData.append("scan_type", "Chest X-Ray PA View");

        try {
            const res = await fetch('/api/v1/predict', {
                method: 'POST',
                body: formData
            });

            if (!res.ok) {
                const errData = await res.json();
                const detailMsg = errData.detail || "Diagnostic analysis failed";
                showUnsupportedModal(detailMsg);
                loader.classList.add('hidden');
                emptyState.classList.remove('hidden');
                return;
            }

            const data = await res.json();
            renderDiagnosticResults(data);

        } catch (err) {
            showUnsupportedModal(err.message || "Prediction has been cancelled.");
            loader.classList.add('hidden');
            emptyState.classList.remove('hidden');
        }
    });

    function renderDiagnosticResults(data) {
        const loader = document.getElementById('results-loader');
        const resultsContent = document.getElementById('results-content');

        loader.classList.add('hidden');
        resultsContent.classList.remove('hidden');

        // Predicted Class & Risk Badge
        document.getElementById('res-predicted-class').textContent = data.predicted_class;
        document.getElementById('res-confidence').textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
        
        const riskBadge = document.getElementById('res-risk-level');
        riskBadge.textContent = data.risk_level;
        if (data.risk_level === 'High Risk') {
            riskBadge.style.borderColor = '#ef4444';
            riskBadge.style.color = '#ef4444';
            riskBadge.style.backgroundColor = 'rgba(239, 68, 68, 0.2)';
        } else {
            riskBadge.style.borderColor = '#10b981';
            riskBadge.style.color = '#10b981';
            riskBadge.style.backgroundColor = 'rgba(16, 185, 129, 0.2)';
        }

        // PDF Link
        const pdfLink = document.getElementById('res-pdf-link');
        if (data.pdf_report_url) {
            pdfLink.href = data.pdf_report_url;
            pdfLink.style.display = 'inline-flex';
        } else {
            pdfLink.style.display = 'none';
        }

        // Probabilities Progress Bars
        const probContainer = document.getElementById('prob-bars-container');
        probContainer.innerHTML = '';

        for (const [cName, pVal] of Object.entries(data.probabilities)) {
            const pct = (pVal * 100).toFixed(1);
            const classKey = cName.toLowerCase().replace(/[^a-z0-9]/g, '');

            const barItem = document.createElement('div');
            barItem.className = 'prob-bar-item';
            barItem.innerHTML = `
                <span>${cName}</span>
                <div class="bar-track">
                    <div class="bar-fill ${classKey}" style="width: ${pct}%;"></div>
                </div>
                <span class="bar-pct">${pct}%</span>
            `;
            probContainer.appendChild(barItem);
        }

        // Images Viewport
        document.getElementById('res-orig-img').src = data.original_image_url;
        document.getElementById('res-base-img').src = data.original_image_url;
        
        const overlayImg = document.getElementById('res-overlay-img');
        overlayImg.src = data.overlay_image_url;
        overlayImg.style.opacity = opacitySlider.value;

        // Spatial Metrics
        const sm = data.spatial_metrics || {};
        document.getElementById('metric-coverage').textContent = `${sm.activation_coverage_pct || 0}%`;
        
        const loc = sm.peak_location_normalized ? `(${sm.peak_location_normalized[0]}, ${sm.peak_location_normalized[1]})` : '(0.0, 0.0)';
        document.getElementById('metric-location').textContent = loc;
        document.getElementById('metric-severity').textContent = sm.severity_score || 0.0;

        // Report Text & Source
        document.getElementById('res-report-source').textContent = data.report_source;
        document.getElementById('res-report-text').textContent = data.report_text;
    }

    // -------------------------------------------------------------------------
    // 3. PATIENT RECORDS & HISTORY AUDIT TABLE
    // -------------------------------------------------------------------------
    const historySearch = document.getElementById('history-search');
    const historyFilterClass = document.getElementById('history-filter-class');
    const btnRefreshHistory = document.getElementById('btn-refresh-history');

    btnRefreshHistory.addEventListener('click', loadHistoryRecords);
    historySearch.addEventListener('input', debounce(loadHistoryRecords, 300));
    historyFilterClass.addEventListener('change', loadHistoryRecords);

    async function loadHistoryRecords() {
        const tbody = document.getElementById('history-tbody');
        tbody.innerHTML = '<tr><td colspan="8" style="text-align:center; padding:20px; color:#94a3b8;">Loading audit history...</td></tr>';

        const search = historySearch.value;
        const filterClass = historyFilterClass.value;

        try {
            const url = `/api/v1/history?search=${encodeURIComponent(search)}&filter_class=${encodeURIComponent(filterClass)}`;
            const res = await fetch(url);
            const data = await res.json();

            if (!data.scans || data.scans.length === 0) {
                tbody.innerHTML = '<tr><td colspan="8" style="text-align:center; padding:20px; color:#94a3b8;">No patient records found.</td></tr>';
                return;
            }

            tbody.innerHTML = '';
            data.scans.forEach(item => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td style="font-family:var(--font-mono); font-size:11px;">${item.scan_id}</td>
                    <td><b>${item.patient_name}</b><br/><span style="font-size:10px; color:#64748b;">${item.patient_id} (${item.age} yrs)</span></td>
                    <td style="font-size:11px;">${item.scan_type}</td>
                    <td><span style="font-weight:600;">${item.predicted_class}</span></td>
                    <td style="font-family:var(--font-mono);">${(item.confidence * 100).toFixed(1)}%</td>
                    <td><span class="badge-risk" style="${item.risk_level === 'High Risk' ? 'border-color:#ef4444; color:#ef4444;' : 'border-color:#10b981; color:#10b981;'}">${item.risk_level}</span></td>
                    <td style="font-size:11px; color:#94a3b8;">${item.created_at}</td>
                    <td>
                        ${item.pdf_report_url ? `<a href="${item.pdf_report_url}" target="_blank" style="color:var(--status-normal); font-size:14px;"><i class="fa-solid fa-file-pdf"></i></a>` : '-'}
                    </td>
                `;
                tbody.appendChild(tr);
            });

        } catch (err) {
            tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; padding:20px; color:#ef4444;">Failed to load records: ${err.message}</td></tr>`;
        }
    }

    // -------------------------------------------------------------------------
    // 4. PLATFORM ANALYTICS DASHBOARD
    // -------------------------------------------------------------------------
    async function loadAnalyticsData() {
        try {
            const [analyticsRes, modelRes] = await Promise.all([
                fetch('/api/v1/analytics'),
                fetch('/api/v1/model/info')
            ]);
            const data = await analyticsRes.json();
            const modelInfo = await modelRes.json();

            document.getElementById('stat-total-scans').textContent = data.total_scans;
            document.getElementById('stat-avg-conf').textContent = `${(data.average_confidence * 100).toFixed(1)}%`;
            document.getElementById('stat-high-risk').textContent = data.high_risk_cases;

            const barsContainer = document.getElementById('analytics-class-bars');
            barsContainer.innerHTML = '';

            const dist = data.class_distribution || {};
            const total = data.total_scans || 1;

            for (const [cName, count] of Object.entries(dist)) {
                const pct = ((count / total) * 100).toFixed(1);
                const classKey = cName.toLowerCase().replace(/[^a-z0-9]/g, '');

                const item = document.createElement('div');
                item.className = 'prob-bar-item';
                item.style.gridTemplateColumns = '120px 1fr 100px';
                item.innerHTML = `
                    <span><b>${cName}</b></span>
                    <div class="bar-track" style="height:12px;">
                        <div class="bar-fill ${classKey}" style="width: ${pct}%;"></div>
                    </div>
                    <span style="font-family:var(--font-mono); font-size:12px;">${count} (${pct}%)</span>
                `;
                barsContainer.appendChild(item);
            }

            renderModelTrainingPanel(modelInfo);

        } catch (err) {
            console.error("Failed to load analytics:", err);
        }
    }

    function renderModelTrainingPanel(modelInfo) {
        const subtitle = document.getElementById('model-training-subtitle');
        const stats = document.getElementById('model-training-stats');
        const charts = document.getElementById('model-training-charts');
        const gradcamGrid = document.getElementById('model-gradcam-grid');

        const valAcc = modelInfo.validation_accuracy != null
            ? `${(modelInfo.validation_accuracy * 100).toFixed(1)}%`
            : 'N/A';
        const epoch = modelInfo.training_epoch != null ? modelInfo.training_epoch : 'N/A';
        const classes = (modelInfo.class_names || []).join(', ');

        subtitle.textContent = `${modelInfo.architecture} | ${modelInfo.dataset}`;

        stats.innerHTML = `
            <div class="model-stat-pill"><span>Validation Accuracy</span><strong>${valAcc}</strong></div>
            <div class="model-stat-pill"><span>Best Epoch</span><strong>${epoch}</strong></div>
            <div class="model-stat-pill"><span>Model Status</span><strong>${modelInfo.model_loaded ? 'Loaded' : 'Not Loaded'}</strong></div>
            <div class="model-stat-pill"><span>Classes</span><strong>${classes}</strong></div>
        `;

        const chartItems = [
            { title: 'Training Curves', url: modelInfo.training_curves_url },
            { title: 'Confusion Matrix', url: modelInfo.confusion_matrix_url },
            { title: 'ROC Curve', url: modelInfo.roc_curve_url }
        ].filter(item => item.url);

        charts.innerHTML = chartItems.map(item => `
            <div class="training-chart-card">
                <h3>${item.title}</h3>
                <img src="${item.url}" alt="${item.title}">
            </div>
        `).join('');

        gradcamGrid.innerHTML = (modelInfo.gradcam_samples || []).map(url => {
            const label = decodeURIComponent(url.split('/').pop().replace('.png', '').replace(/_/g, ' '));
            return `
                <div class="gradcam-sample-card">
                    <img src="${url}" alt="${label}">
                    <span>${label}</span>
                </div>
            `;
        }).join('');
    }

    // Helper Utility
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
});
