const uploadBox = document.getElementById('uploadBox');
const imageInput = document.getElementById('imageInput');
const resultsSection = document.getElementById('results');
const loadingDiv = document.getElementById('loading');
const errorDiv = document.getElementById('error');
const resetBtn = document.getElementById('resetBtn');

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
        return;
    }

    // Show loading
    uploadBox.style.display = 'none';
    resultsSection.style.display = 'none';
    loadingDiv.style.display = 'flex';
    errorDiv.style.display = 'none';

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

        if (!response.ok) {
            showError(data.error || 'Upload failed');
            return;
        }

        // Display results
        displayResults(data);

    } catch (error) {
        showError('Network error: ' + error.message);
    } finally {
        loadingDiv.style.display = 'none';
    }
}

function displayResults(data) {
    // Display image
    document.getElementById('resultImage').src = data.image;

    // Display stats
    document.getElementById('detectionCount').textContent = data.num_detections;
    document.getElementById('areaHectares').textContent = data.carbon.area_hectares.toFixed(2);
    document.getElementById('areaM2').textContent = data.carbon.area_m2.toFixed(0);
    document.getElementById('carbonTons').textContent = data.carbon.carbon_tons.toFixed(2);
    document.getElementById('carbonCO2').textContent = data.carbon.carbon_co2_tons.toFixed(2);

    // Display bboxes
    const bboxList = document.getElementById('bboxList');
    bboxList.innerHTML = '';
    data.bboxes.forEach((bbox, i) => {
        const item = document.createElement('div');
        item.className = 'bbox-item';
        item.innerHTML = `
            <strong>Object ${i + 1}:</strong><br>
            <span>Position: (${bbox.x}, ${bbox.y}) | Size: ${bbox.width}×${bbox.height} | Area: ${bbox.area_pixels} px</span>
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
