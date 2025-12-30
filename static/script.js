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

// Authentication elements
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

// Toast notification function
function showToast(message, type = 'info') {
    toast.textContent = message;
    
    // Set color based on type
    if (type === 'success') {
        toast.className = 'fixed top-6 left-1/2 transform -translate-x-1/2 translate-y-0 opacity-100 px-6 py-4 bg-green-500 text-white font-semibold rounded-xl shadow-2xl transition-all duration-300 z-50 max-w-md text-center';
    } else if (type === 'error') {
        toast.className = 'fixed top-6 left-1/2 transform -translate-x-1/2 translate-y-0 opacity-100 px-6 py-4 bg-red-500 text-white font-semibold rounded-xl shadow-2xl transition-all duration-300 z-50 max-w-md text-center';
    } else {
        toast.className = 'fixed top-6 left-1/2 transform -translate-x-1/2 translate-y-0 opacity-100 px-6 py-4 bg-blue-500 text-white font-semibold rounded-xl shadow-2xl transition-all duration-300 z-50 max-w-md text-center';
    }
    
    setTimeout(() => {
        toast.className = 'fixed top-6 left-1/2 transform -translate-x-1/2 -translate-y-20 opacity-0 px-6 py-4 bg-green-500 text-white font-semibold rounded-xl shadow-2xl transition-all duration-300 z-50 max-w-md text-center';
    }, 4000);
}

// Upload box click
uploadBox.addEventListener('click', () => imageInput.click());

// Drag and drop
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('dragover');
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('dragover');
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleImageUpload(files[0]);
    }
});

// File input change
imageInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleImageUpload(e.target.files[0]);
    }
});

// Reset button
resetBtn.addEventListener('click', () => {
    imageInput.value = '';
    resultsSection.style.display = 'none';
    errorDiv.style.display = 'none';
    headerSection.style.display = 'block';
    uploadSection.style.display = 'block';
    uploadBox.style.display = 'block';
    exampleSidebar.style.display = 'block';
    mainContent.classList.remove('lg:col-span-3');
    mainContent.classList.add('lg:col-span-2');
});

async function handleImageUpload(file) {
    // Validate file
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file');
        showToast('Invalid file type. Please select an image.', 'error');
        return;
    }

    // Show loading
    uploadBox.style.display = 'none';
    resultsSection.style.display = 'none';
    loadingDiv.style.display = 'flex';
    errorDiv.style.display = 'none';
    showToast('Uploading image...', 'info');

    try {
        // Create form data
        const formData = new FormData();
        formData.append('image', file);

        // Send to server
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok || data.error) {
            showError(data.error || 'Upload failed');
            showToast('Detection failed: ' + (data.error || 'Unknown error'), 'error');
            return;
        }

        // Display results
        displayResults(data);
        showToast('Detection completed successfully!', 'success');

    } catch (error) {
        showError('Network error: ' + error.message);
        showToast('Network error occurred', 'error');
    } finally {
        loadingDiv.style.display = 'none';
    }
}

function displayResults(data) {
    // Hide upload section, header, and example sidebar
    headerSection.style.display = 'none';
    uploadSection.style.display = 'none';
    exampleSidebar.style.display = 'none';
    
    // Make main content full width
    mainContent.classList.remove('lg:col-span-2');
    mainContent.classList.add('lg:col-span-3');
    
    // Display image
    document.getElementById('resultImage').src = data.image;

    // Display total stats
    const carbonTotal = data.carbon.total || data.carbon;
    document.getElementById('detectionCount').textContent = data.num_detections;
    document.getElementById('areaHectares').textContent = carbonTotal.area_hectares.toFixed(2);
    document.getElementById('areaM2').textContent = carbonTotal.area_m2.toFixed(0);
    document.getElementById('carbonTons').textContent = carbonTotal.carbon_tons.toFixed(2);
    document.getElementById('carbonCO2').textContent = carbonTotal.carbon_co2_tons.toFixed(2);

    // Display per-object breakdown
    const bboxList = document.getElementById('bboxList');
    bboxList.innerHTML = '';
    
    data.bboxes.forEach((bbox, i) => {
        const item = document.createElement('div');
        item.className = 'bbox-item';
        
        // Get carbon data if available
        let carbonInfo = '';
        if (data.carbon.objects && data.carbon.objects[i]) {
            const carbonData = data.carbon.objects[i];
            carbonInfo = `
                <span>Area: ${carbonData.area_hectares.toFixed(4)} ha</span><br>
                <span>Carbon: ${carbonData.carbon_tons.toFixed(2)} tons C</span><br>
                <span>CO₂: ${carbonData.carbon_co2_tons.toFixed(2)} tons</span>
            `;
        } else {
            carbonInfo = `<span>Area: ${bbox.area_pixels} px</span>`;
        }
        
        item.innerHTML = `
            <strong>Object ${bbox.id || (i + 1)}:</strong><br>
            <span>Position: (${bbox.x}, ${bbox.y})</span><br>
            <span>Size: ${bbox.width}×${bbox.height} px</span><br>
            ${carbonInfo}
        `;
        bboxList.appendChild(item);
    });

    // Show results
    resultsSection.style.display = 'block';
    uploadBox.style.display = 'none';
}

function showError(message) {
    errorDiv.textContent = '❌ Error: ' + message;
    errorDiv.style.display = 'block';
    uploadBox.style.display = 'block';
}

// ===== Authentication Functionality =====

// Check if user is logged in on page load
window.addEventListener('DOMContentLoaded', () => {
    const storedName = localStorage.getItem('userName');
    if (storedName) {
        showUserMenu(storedName);
    }
});

// Show login modal
loginBtn.addEventListener('click', () => {
    loginModal.classList.remove('hidden');
});

// Handle login form submission
loginForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const name = nameInput.value.trim();
    const password = passwordInput.value.trim();
    
    // Dummy authentication - accepts any name and password
    if (name && password) {
        localStorage.setItem('userName', name);
        showUserMenu(name);
        loginModal.classList.add('hidden');
        nameInput.value = '';
        passwordInput.value = '';
        showToast(`Welcome, ${name}! Let's get started 🌟`, 'success');
    }
});

// Close modal when clicking outside
loginModal.addEventListener('click', (e) => {
    if (e.target === loginModal) {
        loginModal.classList.add('hidden');
    }
});

// Toggle dropdown menu
userMenuBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    dropdownMenu.classList.toggle('hidden');
});

// Close dropdown when clicking outside
document.addEventListener('click', () => {
    dropdownMenu.classList.add('hidden');
});

// Handle logout
logoutBtn.addEventListener('click', () => {
    localStorage.removeItem('userName');
    loginBtn.classList.remove('hidden');
    userMenu.classList.add('hidden');
    dropdownMenu.classList.add('hidden');
    showToast('Logged out successfully', 'info');
});

// Show user menu with name
function showUserMenu(name) {
    loginBtn.classList.add('hidden');
    userMenu.classList.remove('hidden');
    userName.textContent = name;
    userInitial.textContent = '🧑🏻‍🔬';
}
