let currentPdfPath = null;
let currentChunks = [];
let currentPages = [];
let selectedChunkIndex = -1;

// Wait for pywebview to be ready
window.addEventListener('pywebviewready', function() {
    console.log('PyWebView is ready');
    initApp();
});

async function initApp() {
    try {
        const algos = await window.pywebview.api.get_algorithms();
        const select = document.getElementById('algorithm-select');
        algos.forEach(algo => {
            const option = document.createElement('option');
            option.value = algo.name;
            option.textContent = algo.name;
            select.appendChild(option);
        });
    } catch (e) {
        console.error("Error loading algorithms", e);
    }
}

document.getElementById('btn-select-pdf').addEventListener('click', async () => {
    const path = await window.pywebview.api.select_pdf();
    if (path) {
        currentPdfPath = path;

        // Format path: remove leading slash and replace / with ▶
        // Adjust replacement logic based on OS if necessary, but simple string replace is fine for display
        const formattedPath = path.replace(/^\//, '').split('/').join(' ▶ ');
        document.getElementById('selected-file').textContent = formattedPath;
    }
});

document.getElementById('btn-process').addEventListener('click', async () => {
    if (!currentPdfPath) {
        alert("Please select a PDF first.");
        return;
    }

    const algoName = document.getElementById('algorithm-select').value;
    document.getElementById('btn-process').textContent = "Processing...";
    document.getElementById('btn-process').disabled = true;

    try {
        const result = await window.pywebview.api.process_pdf(currentPdfPath, algoName);

        if (result.error) {
            alert("Error: " + result.error);
            return;
        }

        currentChunks = result.chunks;
        currentPages = result.pages; // [{page, width, height}]

        renderPdfPages(result.page_count);
        renderChunksList();

        if (currentChunks.length > 0) {
            selectChunk(0);
        }

    } catch (e) {
        console.error(e);
        alert("An error occurred processing the PDF.");
    } finally {
        document.getElementById('btn-process').textContent = "Process";
        document.getElementById('btn-process').disabled = false;
    }
});

document.getElementById('btn-prev').addEventListener('click', () => {
    if (selectedChunkIndex > 0) {
        selectChunk(selectedChunkIndex - 1);
    }
});

document.getElementById('btn-next').addEventListener('click', () => {
    if (selectedChunkIndex < currentChunks.length - 1) {
        selectChunk(selectedChunkIndex + 1);
    }
});

async function renderPdfPages(count) {
    const wrapper = document.getElementById('pages-wrapper');
    wrapper.innerHTML = '';

    const loadPromises = [];

    for (let i = 0; i < count; i++) {
        // Create container
        const container = document.createElement('div');
        container.className = 'page-container';
        container.id = `page-${i}`;

        // Create image
        const img = document.createElement('img');
        img.alt = `Page ${i+1}`;

        // Overlay div
        const overlay = document.createElement('div');
        overlay.className = 'bbox-overlay';
        overlay.id = `overlay-${i}`;

        container.appendChild(img);
        container.appendChild(overlay);
        wrapper.appendChild(container);

        // Fetch image data
        // Note: This might be slow for large PDFs. In a prod app, implement lazy loading.
        const p = window.pywebview.api.get_page_image(currentPdfPath, i, 1.5).then(dataUrl => {
            return new Promise((resolve) => {
                img.onload = () => resolve();
                img.src = dataUrl;
            });
        });
        loadPromises.push(p);
    }

    await Promise.all(loadPromises);
}

function renderChunksList() {
    const list = document.getElementById('chunk-list');
    list.innerHTML = '';

    currentChunks.forEach((chunk, index) => {
        const item = document.createElement('div');
        item.className = 'chunk-item';
        item.id = `chunk-item-${index}`;
        item.textContent = chunk.text.substring(0, 100) + (chunk.text.length > 100 ? '...' : '');
        item.addEventListener('click', () => selectChunk(index));
        list.appendChild(item);
    });
}

function selectChunk(index) {
    // Update selected state
    if (selectedChunkIndex !== -1) {
        const prevItem = document.getElementById(`chunk-item-${selectedChunkIndex}`);
        if (prevItem) prevItem.classList.remove('active');
    }

    selectedChunkIndex = index;
    const newItem = document.getElementById(`chunk-item-${index}`);
    if (newItem) {
        newItem.classList.add('active');
        newItem.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // Highlight on PDF
    highlightChunkOnPdf(index);
}

function highlightChunkOnPdf(index) {
    // Clear existing highlights
    document.querySelectorAll('.bbox-rect').forEach(el => el.remove());

    const chunk = currentChunks[index];
    if (!chunk || !chunk.bboxes) return;

    // Group bboxes by page
    const bboxesByPage = {};
    chunk.bboxes.forEach(bbox => {
        if (!bboxesByPage[bbox.page]) bboxesByPage[bbox.page] = [];
        bboxesByPage[bbox.page].push(bbox);
    });

    let firstBboxElement = null;

    Object.keys(bboxesByPage).forEach(pageIndex => {
        const pageOverlay = document.getElementById(`overlay-${pageIndex}`);
        if (!pageOverlay) return;

        const pageImg = pageOverlay.previousElementSibling; // The <img> tag
        if (!pageImg || !pageImg.complete) {
             // If image isn't loaded yet, we can't calculate scale accurately.
             // Fallback: try to use the page info from backend or wait for load.
             // For now, assume it's loaded or near enough.
        }

        const naturalWidth = pageImg.naturalWidth;
        const naturalHeight = pageImg.naturalHeight;

        // Get PDF dimensions from our stored metadata
        const pdfPageInfo = currentPages[pageIndex]; // {page, width, height}
        if (!pdfPageInfo) return;

        bboxesByPage[pageIndex].forEach(bbox => {
            const rect = document.createElement('div');
            rect.className = 'bbox-rect';

            // Use percentage positioning for responsiveness
            const left = (bbox.x0 / pdfPageInfo.width) * 100;
            const top = (bbox.y0 / pdfPageInfo.height) * 100;
            const width = ((bbox.x1 - bbox.x0) / pdfPageInfo.width) * 100;
            const height = ((bbox.y1 - bbox.y0) / pdfPageInfo.height) * 100;

            rect.style.left = `${left}%`;
            rect.style.top = `${top}%`;
            rect.style.width = `${width}%`;
            rect.style.height = `${height}%`;

            pageOverlay.appendChild(rect);

            if (!firstBboxElement) firstBboxElement = rect;
        });
    });

    if (firstBboxElement) {
        firstBboxElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}

