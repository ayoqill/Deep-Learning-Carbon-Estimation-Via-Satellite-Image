const uploadBox = document.getElementById('uploadBox');
const imageInput = document.getElementById('imageInput');
const resultsSection = document.getElementById('results');
const loadingDiv = document.getElementById('loading');
const errorDiv = document.getElementById('error');
const resetBtn = document.getElementById('resetBtn');
const toast = document.getElementById('toast');

// Toast notification function
function showToast(message, type = 'info') {
    toast.textContent = message;
    toast.className = `toast toast-${type} show`;
    setTimeout(() => {
        toast.className = 'toast';
    }, 3000);
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
    uploadBox.style.display = 'block';
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
