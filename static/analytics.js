// /static/analytics.js
// Analytics page only.
// Navbar/auth/logout is handled by /static/script.js

(() => {
  'use strict';

  const el = {};

  let allAnalyses = [];
  let studyAreas = [];
  let currentShowcase = null;
  let currentViewerMode = 'before';

  // ==============================
  // Fixed Precomputed Langkawi Samples
  // ==============================
  const precomputedAnalyses = [
    {
      id: 'precomputed-langkawi-1',
      type: 'precomputed',
      title: 'Langkawi Analysis 1',
      location: 'Langkawi, Kedah, Malaysia',
      originalImagePath: '/static/precomputed/before_mapping/Langkawi1before.png',
      resultImagePath: '/static/precomputed/after_mapping/Langkawi1.png',
      mangroveCoverage: 92.5,
      totalAreaHectares: 147.6005,
      totalAreaM2: 1476005.46,
      carbonStock: 22140.08,
      co2Equivalent: 81254.1,
      model: 'DeepLabV3+',
      pixelSizeM: 10,
      createdAt: '2026-04-28T10:00:00',
      notes: 'This sample is a fixed showcase result from the Langkawi study area for presentation purposes only.',
      methodology: 'Satellite Image → DeepLabV3+ → Mangrove Mapping → Area & Carbon Estimation'
    },
    {
      id: 'precomputed-langkawi-2',
      type: 'precomputed',
      title: 'Langkawi Analysis 2',
      location: 'Langkawi, Kedah, Malaysia',
      originalImagePath: '/static/precomputed/before_mapping/Langkawi2before.png',
      resultImagePath: '/static/precomputed/after_mapping/Langkawi2.png',
      mangroveCoverage: 55.97,
      totalAreaHectares: 177.7519,
      totalAreaM2: 1777518.73,
      carbonStock: 26662.78,
      co2Equivalent: 97852.41,
      model: 'DeepLabV3+',
      pixelSizeM: 10,
      createdAt: '2026-04-28T10:00:00',
      notes: 'This sample is a fixed showcase result from the Langkawi study area for presentation purposes only.',
      methodology: 'Satellite Image → DeepLabV3+ → Mangrove Mapping → Area & Carbon Estimation'
    },
    {
      id: 'precomputed-langkawi-3',
      type: 'precomputed',
      title: 'Langkawi Analysis 3',
      location: 'Langkawi, Kedah, Malaysia',
      originalImagePath: '/static/precomputed/before_mapping/Langkawi3before.png',
      resultImagePath: '/static/precomputed/after_mapping/Langkawi3.png',
      mangroveCoverage: 41.05,
      totalAreaHectares: 130.8202,
      totalAreaM2: 1308201.9,
      carbonStock: 19623.03,
      co2Equivalent: 72016.51,
      model: 'DeepLabV3+',
      pixelSizeM: 10,
      createdAt: '2026-04-28T10:00:00',
      notes: 'This sample is a fixed showcase result from the Langkawi study area for presentation purposes only.',
      methodology: 'Satellite Image → DeepLabV3+ → Mangrove Mapping → Area & Carbon Estimation'
    },
    {
      id: 'precomputed-langkawi-4',
      type: 'precomputed',
      title: 'Langkawi Analysis 4',
      location: 'Langkawi, Kedah, Malaysia',
      originalImagePath: '/static/precomputed/before_mapping/Langkawi4before.png',
      resultImagePath: '/static/precomputed/after_mapping/Langkawi4.png',
      mangroveCoverage: 48.42,
      totalAreaHectares: 307.2365,
      totalAreaM2: 3072364.97,
      carbonStock: 46085.47,
      co2Equivalent: 169133.69,
      model: 'DeepLabV3+',
      pixelSizeM: 10,
      createdAt: '2026-04-28T10:00:00',
      notes: 'This sample is a fixed showcase result from the Langkawi study area for presentation purposes only.',
      methodology: 'Satellite Image → DeepLabV3+ → Mangrove Mapping → Area & Carbon Estimation'
    }
  ];

  // ==============================
  // Init
  // ==============================
  function init() {
    cacheElements();
    bindEvents();

    currentShowcase = precomputedAnalyses[0] || null;

    switchTab('uploaded');
    loadAnalyses();
    loadStudyAreas();

    initializeShowcaseControls();
    updateShowcaseOverview();
    renderPrecomputedAnalyses();
  }

  function cacheElements() {
    el.tabUploaded = document.getElementById('tabUploaded');
    el.tabStudyAreas = document.getElementById('tabStudyAreas');

    el.uploadedContent = document.getElementById('uploadedContent');
    el.studyAreasContent = document.getElementById('studyAreasContent');

    el.uploadedList = document.getElementById('uploadedList');
    el.uploadedEmpty = document.getElementById('uploadedEmpty');
    el.sortBy = document.getElementById('sortBy');

    el.studyAreasList = document.getElementById('studyAreasList');
    el.langkawiList = document.getElementById('langkawiList');
    el.langkawiEmpty = document.getElementById('langkawiEmpty');

    el.showcaseStatusBadge = document.getElementById('showcaseStatusBadge');
    el.showcaseSampleSelect = document.getElementById('showcaseSampleSelect');

    el.overviewArea = document.getElementById('overviewArea');
    el.overviewModel = document.getElementById('overviewModel');
    el.overviewSamples = document.getElementById('overviewSamples');
    el.overviewStatus = document.getElementById('overviewStatus');
    el.overviewUpdated = document.getElementById('overviewUpdated');

    el.detailModal = document.getElementById('detailModal');
    el.closeModal = document.getElementById('closeModal');
    el.modalTitle = document.getElementById('modalTitle');
    el.modalSubtitle = document.getElementById('modalSubtitle');

    el.beforeImageBlock = document.getElementById('beforeImageBlock');
    el.modalOriginalImage = document.getElementById('modalOriginalImage');
    el.modalResultImage = document.getElementById('modalResultImage');

    el.statCoverage = document.getElementById('statCoverage');
    el.statAreaHa = document.getElementById('statAreaHa');
    el.statAreaM2 = document.getElementById('statAreaM2');
    el.statCarbon = document.getElementById('statCarbon');
    el.statCO2 = document.getElementById('statCO2');
    el.statModel = document.getElementById('statModel');
    el.statPixelSize = document.getElementById('statPixelSize');
    el.statDate = document.getElementById('statDate');
  }

  function bindEvents() {
    el.tabUploaded?.addEventListener('click', () => switchTab('uploaded'));
    el.tabStudyAreas?.addEventListener('click', () => switchTab('studyareas'));

    el.sortBy?.addEventListener('change', renderAnalyses);

    el.closeModal?.addEventListener('click', closeDetailModal);

    el.detailModal?.addEventListener('click', (event) => {
      if (event.target === el.detailModal) {
        closeDetailModal();
      }
    });
  }

  // ==============================
  // Tab Switching
  // ==============================
  function switchTab(tab) {
    const uploadedActive = tab === 'uploaded';

    if (uploadedActive) {
      el.tabUploaded?.classList.add('border-blue-600', 'text-gray-800');
      el.tabUploaded?.classList.remove('border-transparent', 'text-gray-500');

      el.tabStudyAreas?.classList.remove('border-blue-600', 'text-gray-800');
      el.tabStudyAreas?.classList.add('border-transparent', 'text-gray-500');

      el.uploadedContent?.classList.remove('hidden');
      el.studyAreasContent?.classList.add('hidden');
    } else {
      el.tabStudyAreas?.classList.add('border-blue-600', 'text-gray-800');
      el.tabStudyAreas?.classList.remove('border-transparent', 'text-gray-500');

      el.tabUploaded?.classList.remove('border-blue-600', 'text-gray-800');
      el.tabUploaded?.classList.add('border-transparent', 'text-gray-500');

      el.uploadedContent?.classList.add('hidden');
      el.studyAreasContent?.classList.remove('hidden');

      renderPrecomputedAnalyses();
    }
  }

  // ==============================
  // Uploaded Analysis History
  // ==============================
  async function loadAnalyses() {
    if (!el.uploadedList) return;

    el.uploadedList.innerHTML = `
      <div class="col-span-full text-center py-8 text-gray-500">
        Loading uploaded analyses...
      </div>
    `;

    try {
      const response = await fetch('/api/analyses/type/uploaded', {
        credentials: 'include'
      });

      const result = await response.json();

      if (!response.ok || !result.success) {
        throw new Error(result.error || 'Failed to load analyses');
      }

      allAnalyses = Array.isArray(result.data) ? result.data : [];
      renderAnalyses();

    } catch (error) {
      console.error('Error loading analyses:', error);
      allAnalyses = [];
      renderAnalyses();
      notify('Error loading uploaded analyses', 'error');
    }
  }

  function renderAnalyses() {
    if (!el.uploadedList || !el.uploadedEmpty) return;

    let analyses = [...allAnalyses].filter(item => item.type === 'uploaded');

    analyses.sort((a, b) => {
      const dateA = new Date(a.createdAt || 0).getTime();
      const dateB = new Date(b.createdAt || 0).getTime();

      if (el.sortBy?.value === 'oldest') {
        return dateA - dateB;
      }

      return dateB - dateA;
    });

    el.uploadedList.innerHTML = '';

    if (analyses.length === 0) {
      el.uploadedEmpty.classList.remove('hidden');
      return;
    }

    el.uploadedEmpty.classList.add('hidden');

    analyses.forEach((analysis) => {
      const card = document.createElement('div');
      card.className = 'bg-gradient-to-br from-white to-gray-50 rounded-xl border border-gray-200 overflow-hidden shadow-md hover:shadow-xl transition-all duration-300 cursor-pointer group relative';
      card.dataset.analysisId = analysis.id;

      card.innerHTML = `
        <button
          type="button"
          class="delete-analysis-btn absolute top-3 right-3 z-10 bg-white/90 hover:bg-red-50 text-red-600 border border-gray-200 hover:border-red-200 rounded-lg px-3 py-1.5 text-xs font-semibold shadow-sm transition"
          data-analysis-id="${escapeAttr(analysis.id)}"
          data-analysis-title="${escapeAttr(analysis.title || 'Uploaded Analysis')}"
        >
          Delete
        </button>

        <div class="h-48 bg-gray-200 overflow-hidden">
          <img
            src="${escapeAttr(analysis.resultImagePath || analysis.maskPath || '')}"
            alt="${escapeAttr(analysis.title || 'Analysis')}"
            class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
          >
        </div>

        <div class="p-4">
          <h3 class="font-semibold text-gray-800 truncate mb-2">
            ${escapeHtml(analysis.title || 'Uploaded Analysis')}
          </h3>

          <div class="space-y-2 text-sm mb-4">
            <div class="flex justify-between">
              <span class="text-gray-600">Coverage:</span>
              <span class="font-semibold text-gray-800">${formatPercent(analysis.mangroveCoverage)}</span>
            </div>

            <div class="flex justify-between">
              <span class="text-gray-600">Area:</span>
              <span class="font-semibold text-gray-800">${formatHectares(analysis.totalAreaHectares)}</span>
            </div>

            <div class="flex justify-between">
              <span class="text-gray-600">Carbon:</span>
              <span class="font-semibold text-gray-800">${formatTons(analysis.carbonStock)}</span>
            </div>
          </div>

          <div class="border-t border-gray-200 pt-3 flex justify-between items-center">
            <span class="text-xs text-gray-500">${formatDate(analysis.createdAt)}</span>
            <span class="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded font-medium">
              ${escapeHtml(analysis.model || 'Unknown')}
            </span>
          </div>
        </div>
      `;

      card.addEventListener('click', () => {
        viewAnalysisDetail(analysis.id);
      });

      const deleteBtn = card.querySelector('.delete-analysis-btn');

      deleteBtn?.addEventListener('click', (event) => {
        event.stopPropagation();
        deleteUploadedAnalysis(analysis.id, analysis.title || 'Uploaded Analysis');
      });

      el.uploadedList.appendChild(card);
    });
  }

  async function deleteUploadedAnalysis(analysisId, analysisTitle) {
    const confirmed = confirm(`Delete "${analysisTitle}" from your analysis history?`);
    if (!confirmed) return;

    try {
      const response = await fetch(`/api/analyses/${encodeURIComponent(analysisId)}`, {
        method: 'DELETE',
        credentials: 'include'
      });

      const result = await response.json();

      if (!response.ok || !result.success) {
        throw new Error(result.error || result.message || 'Failed to delete analysis');
      }

      allAnalyses = allAnalyses.filter(item => String(item.id) !== String(analysisId));
      renderAnalyses();
      notify('Analysis deleted successfully', 'success');

    } catch (error) {
      console.error('Delete analysis error:', error);
      notify(error.message || 'Error deleting analysis', 'error');
    }
  }

  // ==============================
  // Study Areas
  // ==============================
  async function loadStudyAreas() {
    try {
      const response = await fetch('/api/study-areas', {
        credentials: 'include'
      });

      const result = await response.json();

      if (!response.ok || !result.success) {
        studyAreas = [];
      } else {
        studyAreas = Array.isArray(result.data) ? result.data : [];
      }

      renderStudyAreas();

    } catch (error) {
      console.error('Error loading study areas:', error);
      studyAreas = [];
      renderStudyAreas();
    }
  }

  function renderStudyAreas() {
    if (!el.studyAreasList) return;

    const fallbackStudyArea = [
      {
        name: 'Langkawi',
        location: 'Kedah, Malaysia',
        description: 'Precomputed mangrove study area in Langkawi.'
      }
    ];

    const areasToRender = studyAreas.length > 0 ? studyAreas : fallbackStudyArea;

    el.studyAreasList.innerHTML = areasToRender.map((area) => {
      const name = area.name || area.title || 'Study Area';
      const location = area.location || 'Kedah, Malaysia';
      const description = area.description || 'Precomputed mangrove study area.';
      const isLangkawi = String(name).toLowerCase().includes('langkawi');
      const count = isLangkawi ? precomputedAnalyses.length : (area.images?.length || 0);

      return `
        <div class="bg-gradient-to-br from-white to-gray-50 rounded-xl border border-gray-200 p-6 shadow-md hover:shadow-lg transition-all duration-300">
          <div class="flex items-start gap-4">
            <div class="text-4xl">📍</div>

            <div class="flex-1">
              <h3 class="text-xl font-semibold text-gray-800 mb-1">${escapeHtml(name)}</h3>
              <p class="text-sm text-gray-600 mb-2">${escapeHtml(location)}</p>
              <p class="text-sm text-gray-500 mb-4">${escapeHtml(description)}</p>

              <div class="bg-gray-100 rounded-lg p-3 text-sm">
                <p class="text-gray-700">
                  <strong>${count}</strong> precomputed image${count === 1 ? '' : 's'} available
                </p>
              </div>
            </div>
          </div>
        </div>
      `;
    }).join('');
  }

  // ==============================
  // Showcase Controls
  // ==============================
  function initializeShowcaseControls() {
    if (el.showcaseStatusBadge) {
      el.showcaseStatusBadge.textContent = 'Showcase Only';
    }

    if (!el.showcaseSampleSelect) return;

    el.showcaseSampleSelect.innerHTML = precomputedAnalyses.map(item => `
      <option value="${escapeAttr(item.id)}">${escapeHtml(item.title)}</option>
    `).join('');

    if (currentShowcase) {
      el.showcaseSampleSelect.value = currentShowcase.id;
    }

    el.showcaseSampleSelect.addEventListener('change', (event) => {
      const selectedId = event.target.value;
      const selected = precomputedAnalyses.find(item => item.id === selectedId);

      if (!selected) return;

      currentShowcase = selected;
      currentViewerMode = 'before';

      updateShowcaseOverview();
      renderPrecomputedAnalyses();
    });
  }

  function updateShowcaseOverview() {
    if (!currentShowcase) return;

    if (el.overviewArea) el.overviewArea.textContent = currentShowcase.location || 'Langkawi, Kedah';
    if (el.overviewModel) el.overviewModel.textContent = currentShowcase.model || 'DeepLabV3+';
    if (el.overviewSamples) el.overviewSamples.textContent = String(precomputedAnalyses.length);
    if (el.overviewStatus) el.overviewStatus.textContent = 'Showcase Only';
    if (el.overviewUpdated) el.overviewUpdated.textContent = formatDate(currentShowcase.createdAt);
  }

  // ==============================
  // Render Precomputed Langkawi Showcase
  // ==============================
  function renderPrecomputedAnalyses() {
    if (!el.langkawiList || !el.langkawiEmpty) return;

    el.langkawiList.className = 'grid grid-cols-1 gap-6';

    if (!precomputedAnalyses.length) {
      el.langkawiList.innerHTML = '';
      el.langkawiEmpty.classList.remove('hidden');
      el.langkawiEmpty.innerHTML = '<p class="text-gray-500 text-center py-8">No precomputed analysis available yet.</p>';
      return;
    }

    el.langkawiEmpty.classList.add('hidden');

    if (!currentShowcase) {
      currentShowcase = precomputedAnalyses[0];
    }

    el.langkawiList.innerHTML = `
      <div class="w-full space-y-6">
        <!-- Overview Bar -->
        <div class="grid grid-cols-2 xl:grid-cols-5 gap-3">
          ${renderOverviewBox('Area', currentShowcase.location || 'Langkawi, Kedah, Malaysia')}
          ${renderOverviewBox('Model', currentShowcase.model || 'DeepLabV3+')}
          ${renderOverviewBox('Samples', precomputedAnalyses.length)}
          ${renderOverviewBox('Status', 'Showcase Only')}
          ${renderOverviewBox('Updated', formatDate(currentShowcase.createdAt))}
        </div>

        <!-- Main Dashboard -->
        <div class="grid grid-cols-1 xl:grid-cols-[minmax(0,1.7fr)_380px] gap-6 items-start">
          <!-- Large Viewer -->
          <div class="border border-gray-200 rounded-2xl overflow-hidden bg-white shadow-sm">
            <div class="p-5 border-b border-gray-200 bg-gray-50">
              <h3 class="text-2xl font-bold text-gray-800">${escapeHtml(currentShowcase.title)}</h3>
              <p class="text-gray-600 mt-1">${escapeHtml(currentShowcase.location)}</p>
            </div>

            <div class="p-5">
              <div class="flex gap-3 mb-5">
                <button type="button" data-view-mode="before" class="viewer-toggle px-5 py-2.5 rounded-xl text-sm font-medium transition">
                  Before
                </button>

                <button type="button" data-view-mode="after" class="viewer-toggle px-5 py-2.5 rounded-xl text-sm font-medium transition">
                  After
                </button>
              </div>

              <div class="w-full h-[680px] rounded-2xl border border-gray-200 bg-gray-50 flex items-center justify-center overflow-hidden">
                <img
                  id="showcaseMainImage"
                  src="${escapeAttr(getCurrentShowcaseImage())}"
                  alt="${escapeAttr(currentShowcase.title)}"
                  class="w-full h-full object-contain mx-auto"
                  style="transform: scale(1.12); transform-origin: center;"
                >
              </div>
            </div>
          </div>

          <!-- Analytics Summary -->
          <div class="space-y-6">
            <div class="border border-gray-200 rounded-2xl bg-white p-6 shadow-sm">
              <h3 class="text-lg font-semibold text-gray-800 mb-4">Analytics Summary</h3>

              <div class="flex items-center gap-4 mb-6">
                <div class="relative w-20 h-20">
                  <svg class="w-20 h-20 -rotate-90" viewBox="0 0 100 100">
                    <circle cx="50" cy="50" r="42" stroke="#e5e7eb" stroke-width="10" fill="none"></circle>
                    <circle
                      cx="50"
                      cy="50"
                      r="42"
                      stroke="#2563eb"
                      stroke-width="10"
                      fill="none"
                      stroke-linecap="round"
                      stroke-dasharray="${getDonutDashArray(currentShowcase.mangroveCoverage)}"
                    ></circle>
                  </svg>

                  <div class="absolute inset-0 flex items-center justify-center text-sm font-bold text-gray-800">
                    ${formatPercent(currentShowcase.mangroveCoverage)}
                  </div>
                </div>

                <div>
                  <p class="text-sm text-gray-500">Mangrove Coverage</p>
                  <p class="text-lg font-semibold text-gray-800">${formatPercent(currentShowcase.mangroveCoverage)}</p>
                </div>
              </div>

              <div class="space-y-3">
                ${renderMetricRow('Area (ha)', formatHectares(currentShowcase.totalAreaHectares))}
                ${renderMetricRow('Area (m²)', formatSquareMeters(currentShowcase.totalAreaM2))}
                ${renderMetricRow('Carbon Stock', formatTons(currentShowcase.carbonStock))}
                ${renderMetricRow('CO₂ Equivalent', formatTons(currentShowcase.co2Equivalent))}
                ${renderMetricRow('Pixel Size', currentShowcase.pixelSizeM != null ? `${currentShowcase.pixelSizeM} m` : 'N/A')}
                ${renderMetricRow('Model Used', escapeHtml(currentShowcase.model || 'Unknown'))}
              </div>

              <button
                type="button"
                id="showcaseDetailBtn"
                class="mt-5 w-full bg-gray-800 text-white py-2.5 rounded-lg hover:bg-gray-700 transition font-medium"
              >
                View Detail Modal
              </button>
            </div>

            <div class="border border-gray-200 rounded-2xl bg-white p-6 shadow-sm">
              <h3 class="text-lg font-semibold text-gray-800 mb-4">Project Insight</h3>

              <div class="space-y-4 text-sm text-gray-600">
                <div>
                  <p class="text-xs uppercase tracking-wide text-gray-500 mb-1">Notes</p>
                  <p>${escapeHtml(currentShowcase.notes || 'No notes available.')}</p>
                </div>

                <div>
                  <p class="text-xs uppercase tracking-wide text-gray-500 mb-1">Methodology</p>
                  <p>${escapeHtml(currentShowcase.methodology || 'N/A')}</p>
                </div>

                <div>
                  <p class="text-xs uppercase tracking-wide text-gray-500 mb-1">Updated</p>
                  <p>${formatDate(currentShowcase.createdAt)}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;

    bindShowcaseInteractions();
    updateViewerToggleUI();
  }

  function renderOverviewBox(label, value) {
    return `
      <div class="bg-gray-50 border border-gray-200 rounded-xl p-4">
        <p class="text-[11px] uppercase tracking-wide text-gray-500 mb-1">${escapeHtml(label)}</p>
        <p class="text-xl font-semibold text-gray-800 leading-snug">
          ${escapeHtml(value)}
        </p>
      </div>
    `;
  }

  function renderMetricRow(label, value) {
    return `
      <div class="flex items-center justify-between p-3 bg-gray-50 rounded-lg border border-gray-200">
        <span class="text-gray-600">${escapeHtml(label)}</span>
        <span class="font-semibold text-gray-800">${value}</span>
      </div>
    `;
  }

  function bindShowcaseInteractions() {
    const toggleButtons = document.querySelectorAll('.viewer-toggle');
    const detailBtn = document.getElementById('showcaseDetailBtn');

    toggleButtons.forEach((button) => {
      button.addEventListener('click', () => {
        currentViewerMode = button.dataset.viewMode || 'before';
        updateShowcaseMainImage();
        updateViewerToggleUI();
      });
    });

    detailBtn?.addEventListener('click', () => {
      if (currentShowcase) {
        viewAnalysisDetail(currentShowcase.id);
      }
    });
  }

  function getCurrentShowcaseImage() {
    if (!currentShowcase) return '';

    if (currentViewerMode === 'before') {
      return currentShowcase.originalImagePath || currentShowcase.resultImagePath || '';
    }

    return currentShowcase.resultImagePath || currentShowcase.originalImagePath || '';
  }

  function updateShowcaseMainImage() {
    const image = document.getElementById('showcaseMainImage');
    if (!image) return;

    image.src = getCurrentShowcaseImage();
  }

  function updateViewerToggleUI() {
    const buttons = document.querySelectorAll('.viewer-toggle');

    buttons.forEach((button) => {
      const active = button.dataset.viewMode === currentViewerMode;

      button.className = active
        ? 'viewer-toggle px-5 py-2.5 rounded-xl text-sm font-medium transition bg-gray-800 text-white shadow'
        : 'viewer-toggle px-5 py-2.5 rounded-xl text-sm font-medium transition bg-gray-100 text-gray-700 hover:bg-gray-200';
    });
  }

  function getDonutDashArray(value) {
    const percent = Number(value);
    const circumference = 2 * Math.PI * 42;

    if (Number.isNaN(percent) || percent < 0) {
      return `0 ${circumference}`;
    }

    const normalized = Math.max(0, Math.min(percent, 100));
    const filled = (normalized / 100) * circumference;
    const rest = circumference - filled;

    return `${filled} ${rest}`;
  }

  // ==============================
  // Modal
  // ==============================
  function viewAnalysisDetail(analysisId) {
    const analysis =
      allAnalyses.find(item => String(item.id) === String(analysisId)) ||
      precomputedAnalyses.find(item => String(item.id) === String(analysisId));

    if (!analysis) return;

    if (el.modalTitle) {
      el.modalTitle.textContent = analysis.title || 'Analysis Details';
    }

    if (el.modalSubtitle) {
      const dateStr = formatDate(analysis.createdAt);
      const type = analysis.type === 'uploaded' ? '📤 Uploaded' : '📦 Precomputed';
      el.modalSubtitle.textContent = `${type} • ${dateStr}`;
    }

    // Uploaded history shows only after image.
    // Precomputed showcase can show before image.
    if (analysis.type === 'precomputed' && analysis.originalImagePath) {
      el.beforeImageBlock?.classList.remove('hidden');
      if (el.modalOriginalImage) {
        el.modalOriginalImage.src = analysis.originalImagePath;
      }
    } else {
      el.beforeImageBlock?.classList.add('hidden');
      if (el.modalOriginalImage) {
        el.modalOriginalImage.src = '';
      }
    }

    if (el.modalResultImage) {
      el.modalResultImage.src = analysis.resultImagePath || analysis.maskPath || '';
    }

    setText(el.statCoverage, formatPercent(analysis.mangroveCoverage));
    setText(el.statAreaHa, formatHectares(analysis.totalAreaHectares));
    setText(el.statAreaM2, formatSquareMeters(analysis.totalAreaM2));
    setText(el.statCarbon, formatTons(analysis.carbonStock));
    setText(el.statCO2, formatTons(analysis.co2Equivalent));
    setText(el.statModel, analysis.model || 'Unknown');
    setText(el.statPixelSize, analysis.pixelSizeM != null ? `${analysis.pixelSizeM} m` : 'N/A');
    setText(el.statDate, formatDate(analysis.createdAt));

    el.detailModal?.classList.remove('hidden');
  }

  function closeDetailModal() {
    el.detailModal?.classList.add('hidden');
  }

  function setText(element, text) {
    if (element) element.textContent = text;
  }

  // ==============================
  // Helpers
  // ==============================
  function notify(message, type = 'info') {
    if (typeof window.showToast === 'function') {
      window.showToast(message, type);
      return;
    }

    console.log(`[${type}] ${message}`);
  }

  function formatDate(dateStr) {
    if (!dateStr) return 'N/A';

    const date = new Date(dateStr);
    if (Number.isNaN(date.getTime())) return 'N/A';

    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  }

  function formatPercent(value) {
    if (value == null || value === '') return 'N/A';

    const num = Number(value);
    if (Number.isNaN(num)) return 'N/A';

    return `${num.toFixed(2)}%`;
  }

  function formatHectares(value) {
    if (value == null || value === '') return 'N/A';

    const num = Number(value);
    if (Number.isNaN(num)) return 'N/A';

    return `${num.toFixed(4)} ha`;
  }

  function formatSquareMeters(value) {
    if (value == null || value === '') return 'N/A';

    const num = Number(value);
    if (Number.isNaN(num)) return 'N/A';

    return `${num.toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    })} m²`;
  }

  function formatTons(value) {
    if (value == null || value === '') return 'N/A';

    const num = Number(value);
    if (Number.isNaN(num)) return 'N/A';

    return `${num.toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    })} tons`;
  }

  function escapeHtml(value) {
    if (value == null) return '';

    return String(value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function escapeAttr(value) {
    return escapeHtml(value);
  }

  // Expose only these because some old inline HTML/code may still call them
  window.viewAnalysisDetail = viewAnalysisDetail;
  window.deleteUploadedAnalysis = deleteUploadedAnalysis;

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();