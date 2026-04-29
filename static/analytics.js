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
const userInitial = document.getElementById('userInitial');

const toast = document.getElementById('toast');

let allAnalyses = [];
let studyAreas = [];
let precomputedAnalyses = [];

// ==============================
// Fixed Precomputed Sample
// ==============================
function loadPrecomputedAnalyses() {
  precomputedAnalyses = [
    {
      id: 'precomputed-langkawi-1',
      type: 'precomputed',
      title: 'Langkawi Analysis 1',
      location: 'Langkawi, Kedah, Malaysia',
      originalImagePath: '/static/precomputed/before_mapping/Langkawi1image.png',
      resultImagePath: '/static/precomputed/after_mapping/Langkawi1.png',

      // Replace these values later with your real tested output
      mangroveCoverage: null,
      totalAreaHectares: null,
      totalAreaM2: null,
      carbonStock: null,
      co2Equivalent: null,
      model: 'DeeplabV3+',
      pixelSizeM: 10,
      createdAt: '2026-04-28T10:00:00'
    }
  ];

  renderPrecomputedAnalyses();
}

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

    // Load fixed precomputed sample first
    loadPrecomputedAnalyses();

    // Then load user-specific and study area data
    loadAnalyses();
    loadStudyAreas();
  } catch (error) {
    console.error('Auth check failed:', error);
    window.location.href = '/auth';
  }
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
    <div class="bg-gradient-to-br from-white to-gray-50 rounded-xl border border-gray-200 overflow-hidden shadow-md hover:shadow-xl transition-all duration-300 cursor-pointer group" onclick="viewAnalysisDetail('${analysis.id}')">
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
            <h3 class="text-xl font-semibold text-gray-800 mb-1">${safeText(area.name || 'Study Area')}</h3>
            <p class="text-sm text-gray-600 mb-2">${safeText(area.location || '')}</p>
            <p class="text-sm text-gray-500 mb-4">${safeText(area.description || '')}</p>

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
// Render Precomputed Analyses
// ==============================
function renderPrecomputedAnalyses() {
  if (!precomputedAnalyses.length) {
    if (langkawiList) langkawiList.innerHTML = '';
    if (langkawiEmpty) {
      langkawiEmpty.classList.remove('hidden');
      langkawiEmpty.innerHTML = '<p class="text-gray-500 text-center py-8">No precomputed analysis available yet.</p>';
    }
    return;
  }

  langkawiEmpty?.classList.add('hidden');

  langkawiList.innerHTML = precomputedAnalyses.map(analysis => `
    <div class="bg-gradient-to-br from-white to-gray-50 rounded-xl border border-gray-200 overflow-hidden shadow-md hover:shadow-xl transition-all duration-300 cursor-pointer group" onclick="viewAnalysisDetail('${analysis.id}')">
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
          <span class="text-xs text-gray-500">${safeText(analysis.location || 'Study Area')}</span>
          <span class="text-xs bg-purple-100 text-purple-800 px-2 py-1 rounded font-medium">Precomputed</span>
        </div>
      </div>
    </div>
  `).join('');
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

function safeText(value) {
  if (value == null) return '';
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

// ==============================
// Initialize
// ==============================
document.addEventListener('DOMContentLoaded', checkAuth);