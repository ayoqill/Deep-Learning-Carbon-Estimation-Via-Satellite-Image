// /static/script.js (FULL REPLACEMENT) — aligned to NEW pipeline (segmentation + area + carbon)

const uploadBox = document.getElementById('uploadBox');
const imageInput = document.getElementById('imageInput');
const uploadSection = document.getElementById('uploadSection');
const headerSection = document.getElementById('headerSection');
const mainContent = document.getElementById('mainContent');
const exampleSidebar = document.getElementById('exampleSidebar');

const resultsSection = document.getElementById('results');
const loadingDiv = document.getElementById('loading');
const errorDiv = document.getElementById('error');
const resetBtn = document.getElementById('resetBtn');
const toast = document.getElementById('toast');

// Auth elements
const loginBtn = document.getElementById('loginBtn');
const userMenu = document.getElementById('userMenu');
const userMenuBtn = document.getElementById('userMenuBtn');
const dropdownMenu = document.getElementById('dropdownMenu');
const logoutBtn = document.getElementById('logoutBtn');
const loginModal = document.getElementById('loginModal');
const loginForm = document.getElementById('loginForm');
const nameInput = document.getElementById('nameInput');
const passwordInput = document.getElementById('passwordInput');
const userName = document.getElementById('userName');
const userInitial = document.getElementById('userInitial');

// Result UI elements (new pipeline)
const resultImage = document.getElementById('resultImage');
const coveragePercentEl = document.getElementById('coveragePercent');
const areaHectaresEl = document.getElementById('areaHectares');
const areaM2El = document.getElementById('areaM2');
const carbonTonsEl = document.getElementById('carbonTons');
const carbonCO2El = document.getElementById('carbonCO2');

const pixelSizeSourceEl = document.getElementById('pixelSizeSource');
const pixelSizeValueEl = document.getElementById('pixelSizeValue');
const warningBox = document.getElementById('warningBox');

const ALLOWED_EXT = new Set(['png', 'jpg', 'jpeg', 'tif', 'tiff']);

// -------------------------
// Toast
// -------------------------
function showToast(message, type = 'info') {
  if (!toast) return;

  toast.textContent = message;

  let bg = 'bg-blue-500';
  if (type === 'success') bg = 'bg-green-500';
  if (type === 'error') bg = 'bg-red-500';

  toast.className =
    `fixed top-6 left-1/2 transform -translate-x-1/2 translate-y-0 opacity-100 px-6 py-4 ${bg} text-white font-semibold rounded-xl shadow-2xl transition-all duration-300 z-50 max-w-md text-center`;

  setTimeout(() => {
    toast.className =
      'fixed top-6 left-1/2 transform -translate-x-1/2 -translate-y-20 opacity-0 px-6 py-4 bg-green-500 text-white font-semibold rounded-xl shadow-2xl transition-all duration-300 z-50 max-w-md text-center';
  }, 3500);
}

// -------------------------
// UI helpers
// -------------------------
function showLoading() {
  if (uploadBox) uploadBox.style.display = 'none';
  if (resultsSection) resultsSection.style.display = 'none';
  if (errorDiv) errorDiv.style.display = 'none';
  if (loadingDiv) loadingDiv.style.display = 'flex';
}

function hideLoading() {
  if (loadingDiv) loadingDiv.style.display = 'none';
}

function showError(message) {
  if (!errorDiv) return;
  errorDiv.textContent = '❌ Error: ' + message;
  errorDiv.style.display = 'block';
  if (uploadBox) uploadBox.style.display = 'block';
}

function hideError() {
  if (errorDiv) errorDiv.style.display = 'none';
}

function setWarning(msg) {
  if (!warningBox) return;
  if (msg) {
    warningBox.textContent = msg;
    warningBox.classList.remove('hidden');
  } else {
    warningBox.textContent = '';
    warningBox.classList.add('hidden');
  }
}

function toNum(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

// -------------------------
// Upload click + drag drop
// -------------------------
if (uploadBox && imageInput) {
  uploadBox.addEventListener('click', () => imageInput.click());

  uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('dragover');
  });

  uploadBox.addEventListener('dragleave', () => uploadBox.classList.remove('dragover'));

  uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files && files.length > 0) handleImageUpload(files[0]);
  });

  imageInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleImageUpload(e.target.files[0]);
  });
}

// -------------------------
// Reset
// -------------------------
if (resetBtn) {
  resetBtn.addEventListener('click', () => {
    if (imageInput) imageInput.value = '';
    if (resultsSection) resultsSection.style.display = 'none';
    hideError();

    if (headerSection) headerSection.style.display = 'block';
    if (uploadSection) uploadSection.style.display = 'block';
    if (uploadBox) uploadBox.style.display = 'block';
    if (exampleSidebar) exampleSidebar.style.display = 'block';

    // back to 2-col layout
    if (mainContent) {
      mainContent.classList.remove('lg:col-span-3');
      mainContent.classList.add('lg:col-span-2');
    }

    setWarning(null);
  });
}

// -------------------------
// Main upload handler
// -------------------------
async function handleImageUpload(file) {
  hideError();

  const ext = (file.name.split('.').pop() || '').toLowerCase();
  if (!ALLOWED_EXT.has(ext)) {
    showError('Please upload GeoTIFF (.tif/.tiff) or PNG/JPG.');
    showToast('Invalid file type.', 'error');
    return;
  }

  showLoading();
  showToast('Uploading image...', 'info');

  try {
    const formData = new FormData();
    formData.append('image', file);

    const response = await fetch('/upload', { method: 'POST', body: formData });
    const data = await response.json().catch(() => ({}));

    if (!response.ok || data.success === false) {
      const msg = data.error || `Upload failed (HTTP ${response.status})`;
      showError(msg);
      showToast('Failed: ' + msg, 'error');
      return;
    }

    displayResults(data);
    showToast('Done! ✅', 'success');
  } catch (err) {
    showError('Network error: ' + (err?.message || err));
    showToast('Network error occurred', 'error');
  } finally {
    hideLoading();
  }
}

// -------------------------
// Render results (NEW pipeline)
// -------------------------
function displayResults(data) {
  // hide upload UI
  if (headerSection) headerSection.style.display = 'none';
  if (uploadSection) uploadSection.style.display = 'none';
  if (exampleSidebar) exampleSidebar.style.display = 'none';

  // full width
  if (mainContent) {
    mainContent.classList.remove('lg:col-span-2');
    mainContent.classList.add('lg:col-span-3');
  }

  // image overlay
  if (resultImage) resultImage.src = data.overlay || '';

  // numbers (robust parsing)
  const cov = toNum(data.coveragePercent, 0);
  const ha = toNum(data.areaHectares, 0);
  const m2 = toNum(data.areaM2, 0);
  const cTon = toNum(data.carbonTons, 0);
  const co2 = toNum(data.carbonCO2, 0);

  if (coveragePercentEl) coveragePercentEl.textContent = cov.toFixed(2);
  if (areaHectaresEl) areaHectaresEl.textContent = ha.toFixed(4);

  // show m² with 2 decimals (consistent)
  if (areaM2El) areaM2El.textContent = m2.toFixed(2);

  if (carbonTonsEl) carbonTonsEl.textContent = cTon.toFixed(2);
  if (carbonCO2El) carbonCO2El.textContent = co2.toFixed(2);

  // pixel size meta
  if (pixelSizeSourceEl) pixelSizeSourceEl.textContent = data.pixel_size_source || '-';
  if (pixelSizeValueEl) {
    pixelSizeValueEl.textContent =
      (data.pixel_size_m != null) ? Number(data.pixel_size_m).toFixed(3) : '-';
  }

  // warning
  setWarning(data.warning || null);

  // show results
  if (resultsSection) resultsSection.style.display = 'block';
  if (uploadBox) uploadBox.style.display = 'none';
}

// =========================
// Auth (UI only - localStorage)
// =========================
window.addEventListener('DOMContentLoaded', () => {
  const storedName = localStorage.getItem('userName');
  if (storedName) showUserMenu(storedName);
});

if (loginBtn && loginModal) {
  loginBtn.addEventListener('click', () => loginModal.classList.remove('hidden'));
}

if (loginForm) {
  loginForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const name = (nameInput?.value || '').trim();
    const password = (passwordInput?.value || '').trim();

    if (name && password) {
      localStorage.setItem('userName', name);
      showUserMenu(name);
      if (loginModal) loginModal.classList.add('hidden');
      if (nameInput) nameInput.value = '';
      if (passwordInput) passwordInput.value = '';
      showToast(`Welcome, ${name}!`, 'success');
    }
  });
}

if (loginModal) {
  loginModal.addEventListener('click', (e) => {
    if (e.target === loginModal) loginModal.classList.add('hidden');
  });
}

if (userMenuBtn && dropdownMenu) {
  userMenuBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    dropdownMenu.classList.toggle('hidden');
  });
}

document.addEventListener('click', () => {
  if (dropdownMenu) dropdownMenu.classList.add('hidden');
});

if (logoutBtn) {
  logoutBtn.addEventListener('click', () => {
    localStorage.removeItem('userName');
    if (loginBtn) loginBtn.classList.remove('hidden');
    if (userMenu) userMenu.classList.add('hidden');
    if (dropdownMenu) dropdownMenu.classList.add('hidden');
    showToast('Logged out', 'info');
  });
}

function showUserMenu(name) {
  if (loginBtn) loginBtn.classList.add('hidden');
  if (userMenu) userMenu.classList.remove('hidden');
  if (userName) userName.textContent = name;
  if (userInitial) userInitial.textContent = '🧑🏻‍🔬';
}

// =========================
// Navbar scroll hide/show
// =========================
(() => {
  const navbar = document.getElementById('mainNavbar');
  if (!navbar) return;

  let lastScrollY = window.scrollY;
  let ticking = false;

  function onScroll() {
    if (ticking) return;
    ticking = true;

    window.requestAnimationFrame(() => {
      const y = window.scrollY;
      if (y > lastScrollY && y > 40) {
        navbar.style.opacity = '0';
        navbar.style.transform = 'translateY(-40px)';
        navbar.style.pointerEvents = 'none';
      } else {
        navbar.style.opacity = '1';
        navbar.style.transform = 'translateY(0)';
        navbar.style.pointerEvents = '';
      }
      lastScrollY = y;
      ticking = false;
    });
  }

  window.addEventListener('scroll', onScroll);
})();
