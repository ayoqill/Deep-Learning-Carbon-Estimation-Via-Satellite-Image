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
    tabUploaded.classList.add('border-blue-600');
    tabUploaded.classList.remove('border-transparent');
    tabUploaded.classList.add('text-gray-800');
    tabUploaded.classList.remove('text-gray-500');

    tabStudyAreas.classList.remove('border-blue-600');
    tabStudyAreas.classList.add('border-transparent');
    tabStudyAreas.classList.remove('text-gray-800');
    tabStudyAreas.classList.add('text-gray-500');

    uploadedContent.classList.remove('hidden');
    studyAreasContent.classList.add('hidden');
  } else {
    tabStudyAreas.classList.add('border-blue-600');
    tabStudyAreas.classList.remove('border-transparent');
    tabStudyAreas.classList.add('text-gray-800');
    tabStudyAreas.classList.remove('text-gray-500');

    tabUploaded.classList.remove('border-blue-600');
    tabUploaded.classList.add('border-transparent');
    tabUploaded.classList.remove('text-gray-800');
    tabUploaded.classList.add('text-gray-500');

    uploadedContent.classList.add('hidden');
    studyAreasContent.classList.remove('hidden');
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

    loadAnalyses();
    loadStudyAreas();
  } catch (error) {
    console.error('Auth check failed:', error);
    window.location.href = '/auth';
  }
}

function toggleUserMenu() {
  if (dropdownMenu) {
    dropdownMenu.classList.toggle('hidden');
  }
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
// Load & Display Analyses
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
  const analyses = allAnalyses.filter(a => a.type === 'uploaded');

  if (analyses.length === 0) {
    uploadedList.innerHTML = '';
    uploadedEmpty.classList.remove('hidden');
    return;
  }

  uploadedEmpty.classList.add('hidden');

  uploadedList.innerHTML = analyses.map(analysis => `
    <div class="bg-gradient-to-br from-white to-gray-50 rounded-xl border border-gray-200 overflow-hidden shadow-md hover:shadow-xl transition-all duration-300 cursor-pointer group" onclick="viewAnalysisDetail('${analysis.id}')">
      <!-- Thumbnail -->
      <div class="h-48 bg-gray-200 overflow-hidden">
        <img src="${analysis.resultImagePath}" alt="${analysis.title}" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300">
      </div>

      <!-- Content -->
      <div class="p-4">
        <h3 class="font-semibold text-gray-800 truncate mb-2">${analysis.title}</h3>
        
        <div class="space-y-2 text-sm mb-4">
          <div class="flex justify-between">
            <span class="text-gray-600">Coverage:</span>
            <span class="font-semibold text-gray-800">${analysis.mangroveCoverage}%</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-600">Area:</span>
            <span class="font-semibold text-gray-800">${analysis.totalAreaHectares} ha</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-600">Carbon:</span>
            <span class="font-semibold text-gray-800">${analysis.carbonStock} t</span>
          </div>
        </div>

        <!-- Footer -->
        <div class="border-t border-gray-200 pt-3 flex justify-between items-center">
          <span class="text-xs text-gray-500">${formatDate(analysis.createdAt)}</span>
          <span class="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded font-medium">${analysis.model}</span>
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
      showToast('Failed to load study areas', 'error');
      return;
    }

    studyAreas = result.data || [];
    renderStudyAreas();
    loadPrecomputedAnalyses();
  } catch (error) {
    console.error('Error loading study areas:', error);
    showToast('Error loading study areas', 'error');
  }
}

function renderStudyAreas() {
  studyAreasList.innerHTML = studyAreas.map(area => `
    <div class="bg-gradient-to-br from-white to-gray-50 rounded-xl border border-gray-200 p-6 shadow-md hover:shadow-lg transition-all duration-300">
      <div class="flex items-start gap-4">
        <div class="text-4xl">📍</div>
        <div class="flex-1">
          <h3 class="text-xl font-semibold text-gray-800 mb-1">${area.name}</h3>
          <p class="text-sm text-gray-600 mb-2">${area.location}</p>
          <p class="text-sm text-gray-500 mb-4">${area.description}</p>
          
          <div class="bg-gray-100 rounded-lg p-3 text-sm">
            <p class="text-gray-700">
              <strong>${area.images?.length || 0}</strong> precomputed images available
            </p>
          </div>
        </div>
      </div>
    </div>
  `).join('');
}

// ==============================
// Load Precomputed Analyses
// ==============================
async function loadPrecomputedAnalyses() {
  try {
    const response = await fetch('/api/analyses/type/precomputed');
    const result = await response.json();

    if (!result.success) {
      precomputedAnalyses = [];
      renderPrecomputedAnalyses();
      return;
    }

    precomputedAnalyses = result.data || [];
    renderPrecomputedAnalyses();
  } catch (error) {
    console.error('Error loading precomputed analyses:', error);
    precomputedAnalyses = [];
    renderPrecomputedAnalyses();
  }
}

function renderPrecomputedAnalyses() {
  if (precomputedAnalyses.length === 0) {
    langkawiList.innerHTML = '';
    langkawiEmpty.innerHTML = '<p class="text-gray-500 text-center py-8">Precomputed analyses will be loaded and cached here.</p>';
    return;
  }

  langkawiEmpty.classList.add('hidden');

  langkawiList.innerHTML = precomputedAnalyses.map(analysis => `
    <div class="bg-gradient-to-br from-white to-gray-50 rounded-xl border border-gray-200 overflow-hidden shadow-md hover:shadow-xl transition-all duration-300 cursor-pointer group" onclick="viewAnalysisDetail('${analysis.id}')">
      <!-- Thumbnail -->
      <div class="h-48 bg-gray-200 overflow-hidden">
        <img src="${analysis.resultImagePath}" alt="${analysis.title}" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300">
      </div>

      <!-- Content -->
      <div class="p-4">
        <h3 class="font-semibold text-gray-800 truncate mb-2">${analysis.title}</h3>
        
        <div class="space-y-2 text-sm mb-4">
          <div class="flex justify-between">
            <span class="text-gray-600">Coverage:</span>
            <span class="font-semibold text-gray-800">${analysis.mangroveCoverage}%</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-600">Area:</span>
            <span class="font-semibold text-gray-800">${analysis.totalAreaHectares} ha</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-600">Carbon:</span>
            <span class="font-semibold text-gray-800">${analysis.carbonStock} t</span>
          </div>
        </div>

        <!-- Footer -->
        <div class="border-t border-gray-200 pt-3 flex justify-between items-center">
          <span class="text-xs text-gray-500">${analysis.location || 'Study Area'}</span>
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
  const analysis = allAnalyses.find(a => a.id === analysisId);
  if (!analysis) return;

  modalTitle.textContent = analysis.title;
  
  const dateStr = formatDate(analysis.createdAt);
  const type = analysis.type === 'uploaded' ? '📤 Uploaded' : '📦 Precomputed';
  modalSubtitle.textContent = `${type} • ${dateStr}`;

  document.getElementById('modalOriginalImage').src = analysis.originalImagePath;
  document.getElementById('modalResultImage').src = analysis.resultImagePath;

  document.getElementById('statCoverage').textContent = `${analysis.mangroveCoverage}%`;
  document.getElementById('statAreaHa').textContent = `${analysis.totalAreaHectares} ha`;
  document.getElementById('statAreaM2').textContent = `${analysis.totalAreaM2.toLocaleString()} m²`;
  document.getElementById('statCarbon').textContent = `${analysis.carbonStock} tons`;
  document.getElementById('statCO2').textContent = `${analysis.co2Equivalent} tons`;
  document.getElementById('statModel').textContent = analysis.model || 'Unknown';
  document.getElementById('statPixelSize').textContent = `${analysis.pixelSizeM || '-'} m`;
  document.getElementById('statDate').textContent = dateStr;

  detailModal.classList.remove('hidden');
}

closeModal?.addEventListener('click', () => {
  detailModal.classList.add('hidden');
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
  try {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  } catch (e) {
    return dateStr;
  }
}

// ==============================
// Initialize
// ==============================
document.addEventListener('DOMContentLoaded', checkAuth);
