// filepath: /static/analytics.js
// Analytics page functionality

const tabUploaded = document.getElementById('tabUploaded');
const tabStudyAreas = document.getElementById('tabStudyAreas');
const uploadedContent = document.getElementById('uploadedContent');
const studyAreasContent = document.getElementById('studyAreasContent');
const uploadedList = document.getElementById('uploadedList');
const uploadedEmpty = document.getElementById('uploadedEmpty');
const langkawiList = document.getElementById('langkawiList');
const langkawiEmpty = document.getElementById('langkawiEmpty');
const studyAreasList = document.getElementById('studyAreasList');
const sortBy = document.getElementById('sortBy');

const detailModal = document.getElementById('detailModal');
const closeModal = document.getElementById('closeModal');
const modalTitle = document.getElementById('modalTitle');
const modalSubtitle = document.getElementById('modalSubtitle');

const userMenu = document.getElementById('userMenu');
const userMenuBtn = document.getElementById('userMenuBtn');
const dropdownMenu = document.getElementById('dropdownMenu');
const logoutBtn = document.getElementById('logoutBtn');
const userName = document.getElementById('userName');

const toast = document.getElementById('toast');

// Study area static header controls from HTML
const showcaseStatusBadge = document.getElementById('showcaseStatusBadge');
const showcaseSampleSelect = document.getElementById('showcaseSampleSelect');
const overviewArea = document.getElementById('overviewArea');
const overviewModel = document.getElementById('overviewModel');
const overviewSamples = document.getElementById('overviewSamples');
const overviewStatus = document.getElementById('overviewStatus');
const overviewUpdated = document.getElementById('overviewUpdated');

let allAnalyses = [];
let studyAreas = [];
let precomputedAnalyses = [];
let currentShowcase = null;
let currentViewerMode = 'before';

// ==============================
// Fixed Precomputed Sample
// ==============================
precomputedAnalyses = [
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

    mangroveCoverage: 0,
    totalAreaHectares: 0,
    totalAreaM2: 0,
    carbonStock: 0,
    co2Equivalent: 0,

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

    mangroveCoverage: 0,
    totalAreaHectares: 0,
    totalAreaM2: 0,
    carbonStock: 0,
    co2Equivalent: 0,

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

    mangroveCoverage: 0,
    totalAreaHectares: 0,
    totalAreaM2: 0,
    carbonStock: 0,
    co2Equivalent: 0,

    model: 'DeepLabV3+',
    pixelSizeM: 10,
    createdAt: '2026-04-28T10:00:00',
    notes: 'This sample is a fixed showcase result from the Langkawi study area for presentation purposes only.',
    methodology: 'Satellite Image → DeepLabV3+ → Mangrove Mapping → Area & Carbon Estimation'
  }
];
  
  currentShowcase = precomputedAnalyses[0] || null;
  initializeShowcaseControls();
  updateShowcaseOverview();
  renderPrecomputedAnalyses();

// ==============================
// Toast Notification
// ==============================
function showToast(message, type = 'info') {
  if (!toast) return;

  toast.textContent = message;

  let bg = 'bg-blue-500';
  if (type === 'success') bg = 'bg-green-500';
  if (type === 'error') bg = 'bg-red-500';

  toast.className = `fixed top-6 left-1/2 transform -translate-x-1/2 translate-y-0 opacity-100 px-6 py-4 ${bg} text-white font-semibold rounded-xl shadow-2xl transition-all duration-300 z-50 max-w-md text-center`;

  setTimeout(() => {
    toast.className = 'fixed top-6 left-1/2 transform -translate-x-1/2 -translate-y-20 opacity-0 px-6 py-4 bg-green-500 text-white font-semibold rounded-xl shadow-2xl transition-all duration-300 z-50 max-w-md text-center';
  }, 3500);
}

// ==============================
// Tab Switching
// ==============================
function switchTab(tab) {
  if (tab === 'uploaded') {
    tabUploaded?.classList.add('border-blue-600', 'text-gray-800');
    tabUploaded?.classList.remove('border-transparent', 'text-gray-500');

    tabStudyAreas?.classList.remove('border-blue-600', 'text-gray-800');
    tabStudyAreas?.classList.add('border-transparent', 'text-gray-500');

    uploadedContent?.classList.remove('hidden');
    studyAreasContent?.classList.add('hidden');
  } else {
    tabStudyAreas?.classList.add('border-blue-600', 'text-gray-800');
    tabStudyAreas?.classList.remove('border-transparent', 'text-gray-500');

    tabUploaded?.classList.remove('border-blue-600', 'text-gray-800');
    tabUploaded?.classList.add('border-transparent', 'text-gray-500');

    uploadedContent?.classList.add('hidden');
    studyAreasContent?.classList.remove('hidden');
  }
}

// ==============================
// Auth & User Menu
// ==============================
async function checkAuth() {
  try {
    const response = await fetch('/auth_status');
    const data = await response.json();

    if (!data.authenticated) {
      window.location.href = '/auth';
      return;
    }

    if (userMenu) userMenu.classList.remove('hidden');
    if (userName) userName.textContent = data.username || 'User';
  } catch (error) {
    console.error('Auth check failed:', error);
    window.location.href = '/auth';
    return;
  }

  // Run page setup separately so UI errors don't trigger auth redirect
  try {
    loadPrecomputedAnalyses();
  } catch (error) {
    console.error('Showcase init failed:', error);
  }

  loadAnalyses();
  loadStudyAreas();
}

function toggleUserMenu() {
  dropdownMenu?.classList.toggle('hidden');
}

userMenuBtn?.addEventListener('click', toggleUserMenu);

logoutBtn?.addEventListener('click', async () => {
  try {
    await fetch('/logout', { method: 'POST' });
    window.location.href = '/auth';
  } catch (error) {
    console.error('Logout failed:', error);
  }
});

// ==============================
// Load & Display Uploaded Analyses
// ==============================
async function loadAnalyses() {
  try {
    const response = await fetch('/api/analyses');
    const result = await response.json();

    if (!result.success) {
      showToast('Failed to load analyses', 'error');
      return;
    }

    allAnalyses = result.data || [];
    renderAnalyses();
  } catch (error) {
    console.error('Error loading analyses:', error);
    showToast('Error loading analyses', 'error');
  }
}

async function deleteUploadedAnalysis(analysisId, analysisTitle) {
  const confirmed = confirm(`Delete "${analysisTitle}" from your analysis history?`);
  if (!confirmed) return;

  try {
    const response = await fetch(`/api/analyses/${analysisId}`, {
      method: 'DELETE'
    });

    const result = await response.json();

    if (!response.ok || !result.success) {
      showToast(result.error || result.message || 'Failed to delete analysis', 'error');
      return;
    }

    allAnalyses = allAnalyses.filter(a => String(a.id) !== String(analysisId));
    renderAnalyses();
    showToast('Analysis deleted successfully', 'success');
  } catch (error) {
    console.error('Delete analysis error:', error);
    showToast('Error deleting analysis', 'error');
  }
}

function renderAnalyses() {
  let analyses = allAnalyses.filter(a => a.type === 'uploaded');

  if (sortBy?.value === 'oldest') {
    analyses = analyses.sort((a, b) => new Date(a.createdAt) - new Date(b.createdAt));
  } else {
    analyses = analyses.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
  }

  if (analyses.length === 0) {
    if (uploadedList) uploadedList.innerHTML = '';
    uploadedEmpty?.classList.remove('hidden');
    return;
  }

  uploadedEmpty?.classList.add('hidden');

  uploadedList.innerHTML = analyses.map(analysis => `
    <div class="bg-gradient-to-br from-white to-gray-50 rounded-xl border border-gray-200 overflow-hidden shadow-md hover:shadow-xl transition-all duration-300 cursor-pointer group relative" onclick="viewAnalysisDetail('${analysis.id}')">
      
      <!-- Delete Button -->
      <button
        type="button"
        class="absolute top-3 right-3 z-10 bg-white/90 hover:bg-red-50 text-red-600 border border-gray-200 hover:border-red-200 rounded-lg px-3 py-1.5 text-xs font-semibold shadow-sm transition"
        onclick="event.stopPropagation(); deleteUploadedAnalysis('${analysis.id}', '${safeText(analysis.title)}')"
      >
        Delete
      </button>

      <div class="h-48 bg-gray-200 overflow-hidden">
        <img src="${analysis.resultImagePath}" alt="${analysis.title}" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300">
      </div>

      <div class="p-4">
        <h3 class="font-semibold text-gray-800 truncate mb-2">${safeText(analysis.title)}</h3>

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
          <span class="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded font-medium">${safeText(analysis.model || 'Unknown')}</span>
        </div>
      </div>
    </div>
  `).join('');
}

// ==============================
// Load & Display Study Areas
// ==============================
async function loadStudyAreas() {
  try {
    const response = await fetch('/api/study-areas');
    const result = await response.json();

    if (!result.success) {
      studyAreas = [];
      renderStudyAreas();
      return;
    }

    studyAreas = result.data || [];
    renderStudyAreas();
  } catch (error) {
    console.error('Error loading study areas:', error);
    studyAreas = [];
    renderStudyAreas();
  }
}

function renderStudyAreas() {
  const fallbackStudyArea = [
    {
      name: 'Langkawi',
      location: 'Kedah, Malaysia',
      description: 'Precomputed mangrove study area in Langkawi'
    }
  ];

  const areasToRender = studyAreas.length > 0 ? studyAreas : fallbackStudyArea;

  studyAreasList.innerHTML = areasToRender.map(area => {
    const isLangkawi = (area.name || '').toLowerCase().includes('langkawi');
    const count = isLangkawi ? precomputedAnalyses.length : (area.images?.length || 0);

    return `
      <div class="bg-gradient-to-br from-white to-gray-50 rounded-xl border border-gray-200 p-6 shadow-md hover:shadow-lg transition-all duration-300">
        <div class="flex items-start gap-4">
          <div class="text-4xl">📍</div>
          <div class="flex-1">
            <h3 class="text-xl font-semibold text-gray-800 mb-1">${escapeHtml(area.name || 'Study Area')}</h3>
            <p class="text-sm text-gray-600 mb-2">${escapeHtml(area.location || '')}</p>
            <p class="text-sm text-gray-500 mb-4">${escapeHtml(area.description || '')}</p>

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
// Showcase Static Header Controls
// ==============================
function initializeShowcaseControls() {
  if (showcaseStatusBadge) {
    showcaseStatusBadge.textContent = 'Showcase Only';
  }

  if (showcaseSampleSelect) {
    showcaseSampleSelect.innerHTML = precomputedAnalyses.map(item => `
      <option value="${item.id}">${escapeHtml(item.title)}</option>
    `).join('');

    if (currentShowcase) {
      showcaseSampleSelect.value = currentShowcase.id;
    }

    showcaseSampleSelect.onchange = (e) => {
      const selectedId = e.target.value;
      const selected = precomputedAnalyses.find(item => item.id === selectedId);
      if (!selected) return;

      currentShowcase = selected;
      currentViewerMode = 'before';
      updateShowcaseOverview();
      renderPrecomputedAnalyses();
    };
  }
}

function updateShowcaseOverview() {
  if (!currentShowcase) return;

  if (overviewArea) overviewArea.textContent = currentShowcase.location || 'Langkawi, Kedah';
  if (overviewModel) overviewModel.textContent = currentShowcase.model || 'DeepLabV3+';
  if (overviewSamples) overviewSamples.textContent = String(precomputedAnalyses.length);
  if (overviewStatus) overviewStatus.textContent = 'Showcase Only';
  if (overviewUpdated) overviewUpdated.textContent = formatDate(currentShowcase.createdAt);
}

// ==============================
// Render Precomputed Analyses
// ==============================
function renderPrecomputedAnalyses() {
  if (!langkawiList || !langkawiEmpty) return;

  langkawiList.className = 'grid grid-cols-1 gap-6';

  if (!precomputedAnalyses.length) {
    langkawiList.innerHTML = '';
    langkawiEmpty.classList.remove('hidden');
    langkawiEmpty.innerHTML = '<p class="text-gray-500 text-center py-8">No precomputed analysis available yet.</p>';
    return;
  }

  langkawiEmpty.classList.add('hidden');

  if (!currentShowcase) {
    currentShowcase = precomputedAnalyses[0];
  }

  langkawiList.innerHTML = `
    <div class="w-full space-y-6">
    <!-- Overview Bar -->
    <div class="grid grid-cols-2 xl:grid-cols-5 gap-3">
      <div class="bg-gray-50 border border-gray-200 rounded-xl p-4">
        <p class="text-[11px] uppercase tracking-wide text-gray-500 mb-1">Area</p>
        <p class="text-xl font-semibold text-gray-800 leading-snug">
          ${escapeHtml(currentShowcase.location || 'Langkawi, Kedah, Malaysia')}
        </p>
      </div>

      <div class="bg-gray-50 border border-gray-200 rounded-xl p-4">
        <p class="text-[11px] uppercase tracking-wide text-gray-500 mb-1">Model</p>
        <p class="text-xl font-semibold text-gray-800">
          ${escapeHtml(currentShowcase.model || 'DeepLabV3+')}
        </p>
      </div>

      <div class="bg-gray-50 border border-gray-200 rounded-xl p-4">
        <p class="text-[11px] uppercase tracking-wide text-gray-500 mb-1">Samples</p>
        <p class="text-xl font-semibold text-gray-800">
          ${precomputedAnalyses.length}
        </p>
      </div>

      <div class="bg-gray-50 border border-gray-200 rounded-xl p-4">
        <p class="text-[11px] uppercase tracking-wide text-gray-500 mb-1">Status</p>
        <p class="text-xl font-semibold text-gray-800">
          Showcase Only
        </p>
      </div>

      <div class="bg-gray-50 border border-gray-200 rounded-xl p-4">
        <p class="text-[11px] uppercase tracking-wide text-gray-500 mb-1">Updated</p>
        <p class="text-xl font-semibold text-gray-800 leading-snug">
          ${formatDate(currentShowcase.createdAt)}
        </p>
      </div>
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
                src="${getCurrentShowcaseImage()}"
                alt="${escapeHtml(currentShowcase.title)}"
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
                  ${currentShowcase.mangroveCoverage != null ? `${currentShowcase.mangroveCoverage}%` : 'N/A'}
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

function renderMetricRow(label, value) {
  return `
    <div class="flex items-center justify-between p-3 bg-gray-50 rounded-lg border border-gray-200">
      <span class="text-gray-600">${label}</span>
      <span class="font-semibold text-gray-800">${value}</span>
    </div>
  `;
}

function bindShowcaseInteractions() {
  const toggleButtons = document.querySelectorAll('.viewer-toggle');
  const detailBtn = document.getElementById('showcaseDetailBtn');

  toggleButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      currentViewerMode = btn.dataset.viewMode || 'before';
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
  return currentViewerMode === 'before'
    ? (currentShowcase.originalImagePath || currentShowcase.resultImagePath || '')
    : (currentShowcase.resultImagePath || currentShowcase.originalImagePath || '');
}

function updateShowcaseMainImage() {
  const image = document.getElementById('showcaseMainImage');
  if (!image) return;
  image.src = getCurrentShowcaseImage();
}

function updateViewerToggleUI() {
  const buttons = document.querySelectorAll('.viewer-toggle');
  buttons.forEach(btn => {
    const active = btn.dataset.viewMode === currentViewerMode;
    btn.className = active
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
// Modal & Detail View
// ==============================
function viewAnalysisDetail(analysisId) {
  const analysis =
    allAnalyses.find(a => String(a.id) === String(analysisId)) ||
    precomputedAnalyses.find(a => String(a.id) === String(analysisId));

  if (!analysis) return;

  modalTitle.textContent = analysis.title || 'Analysis Details';

  const dateStr = formatDate(analysis.createdAt);
  const type = analysis.type === 'uploaded' ? '📤 Uploaded' : '📦 Precomputed';
  modalSubtitle.textContent = `${type} • ${dateStr}`;

  document.getElementById('modalOriginalImage').src = analysis.originalImagePath || '';
  document.getElementById('modalResultImage').src = analysis.resultImagePath || '';

  document.getElementById('statCoverage').textContent = formatPercent(analysis.mangroveCoverage);
  document.getElementById('statAreaHa').textContent = formatHectares(analysis.totalAreaHectares);
  document.getElementById('statAreaM2').textContent = formatSquareMeters(analysis.totalAreaM2);
  document.getElementById('statCarbon').textContent = formatTons(analysis.carbonStock);
  document.getElementById('statCO2').textContent = formatTons(analysis.co2Equivalent);
  document.getElementById('statModel').textContent = analysis.model || 'Unknown';
  document.getElementById('statPixelSize').textContent = analysis.pixelSizeM != null ? `${analysis.pixelSizeM} m` : 'N/A';
  document.getElementById('statDate').textContent = dateStr;

  detailModal?.classList.remove('hidden');
}

window.viewAnalysisDetail = viewAnalysisDetail;

closeModal?.addEventListener('click', () => {
  detailModal?.classList.add('hidden');
});

detailModal?.addEventListener('click', (e) => {
  if (e.target === detailModal) {
    detailModal.classList.add('hidden');
  }
});

// ==============================
// Event Listeners
// ==============================
tabUploaded?.addEventListener('click', () => switchTab('uploaded'));
tabStudyAreas?.addEventListener('click', () => switchTab('studyareas'));

sortBy?.addEventListener('change', () => {
  renderAnalyses();
});

// ==============================
// Utility Functions
// ==============================
function formatDate(dateStr) {
  if (!dateStr) return 'N/A';

  try {
    const date = new Date(dateStr);
    if (Number.isNaN(date.getTime())) return 'N/A';

    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  } catch (e) {
    return 'N/A';
  }
}

function formatPercent(value) {
  return value == null || value === '' ? 'N/A' : `${value}%`;
}

function formatHectares(value) {
  return value == null || value === '' ? 'N/A' : `${value} ha`;
}

function formatSquareMeters(value) {
  if (value == null || value === '') return 'N/A';
  const num = Number(value);
  return Number.isNaN(num) ? 'N/A' : `${num.toLocaleString()} m²`;
}

function formatTons(value) {
  return value == null || value === '' ? 'N/A' : `${value} tons`;
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

// Keep compatibility with your earlier code
const safeText = escapeHtml;

// ==============================
// Initialize
// ==============================
document.addEventListener('DOMContentLoaded', checkAuth);