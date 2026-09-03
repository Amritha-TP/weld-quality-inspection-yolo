/**
 * Weld Quality Inspection — YOLO AI System
 * Frontend Application Controller
 */

// Global Application State
let selectedFile = null;
let webcamStream = null;
let isInspectingWebcam = false;
let isProcessingFrame = false;
let confidenceThreshold = 0.25;
let animationFrameId = null;
let lastFrameTimestamp = 0;

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", () => {
    initConfidenceSlider();
    initDropZone();
});

/* ================= CONFIDENCE SLIDER ================= */
function initConfidenceSlider() {
    const slider = document.getElementById("conf-slider");
    const displayVal = document.getElementById("conf-val-display");
    const decimalVal = document.querySelector(".conf-decimal");

    if (slider) {
        slider.addEventListener("input", (e) => {
            confidenceThreshold = parseFloat(e.target.value);
            const pct = Math.round(confidenceThreshold * 100);
            displayVal.textContent = `${pct}%`;
            if (decimalVal) decimalVal.textContent = `(${confidenceThreshold.toFixed(2)})`;
        });
    }
}

/* ================= TAB NAVIGATION ================= */
function switchTab(tabId) {
    document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(content => content.style.display = "none");

    if (tabId === "image-mode") {
        document.getElementById("tab-image-btn").classList.add("active");
        document.getElementById("image-mode").style.display = "block";
    } else if (tabId === "webcam-mode") {
        document.getElementById("tab-webcam-btn").classList.add("active");
        document.getElementById("webcam-mode").style.display = "block";
    }
}

/* ================= DRAG & DROP & FILE SELECTION ================= */
function initDropZone() {
    const dropZone = document.getElementById("drop-zone");

    if (!dropZone) return;

    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add("drag-over");
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove("drag-over");
        }, false);
    });

    dropZone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            handleFile(files[0]);
        }
    });
}

function handleFileSelect(event) {
    const files = event.target.files;
    if (files && files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFile(file) {
    const validTypes = ["image/jpeg", "image/png", "image/webp"];
    if (!validTypes.includes(file.type)) {
        alert("Please upload JPG, JPEG, PNG, or WEBP.");
        return;
    }

    selectedFile = file;

    // Update Drop Zone UI
    document.getElementById("drop-zone-content").style.display = "none";
    const previewStrip = document.getElementById("file-preview-strip");
    previewStrip.style.display = "flex";
    
    document.getElementById("selected-file-name").textContent = file.name;
    document.getElementById("selected-file-size").textContent = formatBytes(file.size);

    // Read and Preview original image
    const reader = new FileReader();
    reader.onload = function(e) {
        document.getElementById("img-original-preview").src = e.target.result;
    };
    reader.readAsDataURL(file);

    // Enable inspect button
    document.getElementById("btn-inspect-image").disabled = false;
}

function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

/* ================= IMAGE INSPECTION ================= */
async function runImageInspection() {
    if (!selectedFile) {
        alert("Please select a weld image first.");
        return;
    }

    const inspectBtn = document.getElementById("btn-inspect-image");
    const loadingOverlay = document.getElementById("image-loading");
    const resultsContainer = document.getElementById("image-results-container");

    inspectBtn.disabled = true;
    loadingOverlay.style.display = "flex";
    resultsContainer.style.display = "none";

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("confidence", confidenceThreshold);

    try {
        const response = await fetch("/detect", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!data.success) {
            alert(data.error || "Inspection failed.");
            return;
        }

        renderImageResults(data);
    } catch (err) {
        console.error("Error during inspection:", err);
        alert("Error connecting to server. Please ensure app.py is running on localhost:5000.");
    } finally {
        inspectBtn.disabled = false;
        loadingOverlay.style.display = "none";
    }
}

function renderImageResults(data) {
    const resultsContainer = document.getElementById("image-results-container");
    const banner = document.getElementById("img-result-banner");
    const icon = document.getElementById("img-banner-icon");
    const title = document.getElementById("img-banner-title");
    const msg = document.getElementById("img-banner-message");

    // 1. Prominent Result Banner
    banner.className = "result-banner"; // reset
    if (data.overall_result === "GOOD WELD") {
        banner.classList.add("banner-good");
        icon.textContent = "✓";
        title.textContent = "✓ GOOD WELD";
        msg.textContent = "All detected weld regions meet quality specifications.";
    } else if (data.overall_result === "BAD WELD") {
        banner.classList.add("banner-bad");
        icon.textContent = "✗";
        title.textContent = "✗ BAD WELD";
        msg.textContent = "Defect or bad weld detected. Inspection failed conservative quality criteria.";
    } else {
        banner.classList.add("banner-none");
        icon.textContent = "!";
        title.textContent = "! NO WELD DETECTED";
        msg.textContent = "No weld feature was detected above the configured confidence threshold.";
    }

    // 2. Processed Image Preview
    if (data.processed_image) {
        document.getElementById("img-annotated-preview").src = data.processed_image;
    }

    // 3. Summary Statistics
    document.getElementById("sum-overall").textContent = data.overall_result;
    document.getElementById("sum-overall").className = `summary-value ${data.overall_result === 'GOOD WELD' ? 'text-good' : (data.overall_result === 'BAD WELD' ? 'text-bad' : '')}`;
    document.getElementById("sum-total").textContent = data.detection_count;
    document.getElementById("sum-good").textContent = data.good_count;
    document.getElementById("sum-bad").textContent = data.bad_count;
    document.getElementById("sum-conf").textContent = `${data.highest_confidence_percent}%`;

    // 4. Detections Table
    const tbody = document.getElementById("detections-tbody");
    tbody.innerHTML = "";

    if (!data.detections || data.detections.length === 0) {
        tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; color: var(--text-dim); padding:20px;">No weld objects detected above ${Math.round(confidenceThreshold * 100)}% confidence threshold.</td></tr>`;
    } else {
        data.detections.forEach((det, idx) => {
            const tr = document.createElement("tr");
            
            const qualityBadge = det.quality === "GOOD"
                ? `<span class="badge-table badge-table-good">GOOD</span>`
                : `<span class="badge-table badge-table-bad">BAD / DEFECT</span>`;

            tr.innerHTML = `
                <td>${idx + 1}</td>
                <td><strong>${det.class_name}</strong></td>
                <td>${qualityBadge}</td>
                <td><strong>${det.confidence_percent}%</strong></td>
                <td><code class="bbox-code">[${det.bbox.join(", ")}]</code></td>
            `;
            tbody.appendChild(tr);
        });
    }

    resultsContainer.style.display = "block";
    resultsContainer.scrollIntoView({ behavior: "smooth" });
}

/* ================= LIVE WEBCAM INSPECTION ================= */
async function openWebcam() {
    const video = document.getElementById("webcam-video");
    const placeholder = document.getElementById("video-placeholder");
    const statusText = document.getElementById("cam-status-text");
    const dot = document.getElementById("cam-dot");

    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: "environment"
            },
            audio: false
        });

        video.srcObject = webcamStream;
        video.style.display = "block";
        placeholder.style.display = "none";

        dot.className = "status-dot dot-active";
        statusText.textContent = "Camera: Connected";

        document.getElementById("btn-open-cam").disabled = true;
        document.getElementById("btn-start-inspect").disabled = false;
        document.getElementById("btn-close-cam").disabled = false;

    } catch (err) {
        console.error("Camera Access Error:", err);
        alert("Camera access was denied or unavailable.\n\nPlease check camera connections and allow browser permissions in settings.");
        dot.className = "status-dot dot-inactive";
        statusText.textContent = "Camera: Permission Denied";
    }
}

function startWebcamInspection() {
    if (!webcamStream) return;

    isInspectingWebcam = true;
    
    // Hide raw video, show processed stream image
    document.getElementById("webcam-video").style.display = "none";
    document.getElementById("webcam-processed-stream").style.display = "block";

    // Status pill
    document.getElementById("cam-dot").className = "status-dot dot-inspecting";
    document.getElementById("cam-status-text").textContent = "Inspection: Running";
    document.getElementById("telemetry-status").textContent = "RUNNING";
    document.getElementById("telemetry-status").style.color = "var(--accent-cyan)";

    // Toolbar buttons
    document.getElementById("btn-start-inspect").disabled = true;
    document.getElementById("btn-stop-inspect").disabled = false;

    // Start frame capture loop
    processWebcamFrameLoop();
}

function stopWebcamInspection() {
    isInspectingWebcam = false;

    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }

    // Show raw video again
    document.getElementById("webcam-video").style.display = "block";
    document.getElementById("webcam-processed-stream").style.display = "none";

    // Status pill
    document.getElementById("cam-dot").className = "status-dot dot-active";
    document.getElementById("cam-status-text").textContent = "Inspection: Stopped";
    document.getElementById("telemetry-status").textContent = "STOPPED";
    document.getElementById("telemetry-status").style.color = "var(--text-muted)";

    // Toolbar buttons
    document.getElementById("btn-start-inspect").disabled = false;
    document.getElementById("btn-stop-inspect").disabled = true;
}

function closeWebcam() {
    stopWebcamInspection();

    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }

    const video = document.getElementById("webcam-video");
    video.srcObject = null;
    video.style.display = "none";
    document.getElementById("webcam-processed-stream").style.display = "none";
    document.getElementById("video-placeholder").style.display = "block";

    document.getElementById("cam-dot").className = "status-dot dot-inactive";
    document.getElementById("cam-status-text").textContent = "Camera: Disconnected";
    document.getElementById("telemetry-status").textContent = "IDLE";

    document.getElementById("btn-open-cam").disabled = false;
    document.getElementById("btn-start-inspect").disabled = true;
    document.getElementById("btn-stop-inspect").disabled = true;
    document.getElementById("btn-close-cam").disabled = true;

    // Reset Live HUD
    resetLiveHUD();
}

async function processWebcamFrameLoop() {
    if (!isInspectingWebcam) return;

    if (!isProcessingFrame) {
        isProcessingFrame = true;
        const startTime = performance.now();

        try {
            const frameBase64 = captureWebcamFrame();
            if (frameBase64) {
                const response = await fetch("/detect_frame", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        image: frameBase64,
                        confidence: confidenceThreshold
                    })
                });

                const data = await response.json();
                if (data.success && isInspectingWebcam) {
                    const elapsed = Math.round(performance.now() - startTime);
                    const fps = (1000 / elapsed).toFixed(1);
                    
                    renderLiveTelemetry(data, fps, elapsed);
                }
            }
        } catch (err) {
            console.error("Frame processing error:", err);
        } finally {
            isProcessingFrame = false;
        }
    }

    if (isInspectingWebcam) {
        setTimeout(processWebcamFrameLoop, 30); // ~30ms throttle for smooth execution
    }
}

function captureWebcamFrame() {
    const video = document.getElementById("webcam-video");
    const canvas = document.getElementById("frame-canvas");

    if (!video || video.readyState !== video.HAVE_ENOUGH_DATA) {
        return null;
    }

    // Scale canvas to max 640px for fast streaming
    const maxWidth = 640;
    const scale = Math.min(1, maxWidth / video.videoWidth);
    canvas.width = video.videoWidth * scale;
    canvas.height = video.videoHeight * scale;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    return canvas.toDataURL("image/jpeg", 0.85);
}

function renderLiveTelemetry(data, fps, latencyMs) {
    // 1. Update processed image view
    if (data.processed_image) {
        document.getElementById("webcam-processed-stream").src = data.processed_image;
    }

    // 2. Update Live HUD Banner
    const liveTitle = document.getElementById("live-banner-title");
    const liveIcon = document.getElementById("live-banner-icon");
    const liveDesc = document.getElementById("live-banner-desc");

    if (data.overall_result === "GOOD WELD") {
        liveTitle.textContent = "✓ GOOD WELD";
        liveTitle.style.color = "var(--status-good)";
        liveIcon.textContent = "✓";
        liveDesc.textContent = "Live Stream: All welds comply with specifications.";
    } else if (data.overall_result === "BAD WELD") {
        liveTitle.textContent = "✗ BAD WELD";
        liveTitle.style.color = "var(--status-bad)";
        liveIcon.textContent = "✗";
        liveDesc.textContent = "Live Stream: Defect detected in stream!";
    } else {
        liveTitle.textContent = "! NO WELD DETECTED";
        liveTitle.style.color = "var(--status-none)";
        liveIcon.textContent = "!";
        liveDesc.textContent = "Live Stream: Searching for weld features...";
    }

    // 3. Telemetry Rows
    document.getElementById("telemetry-fps").textContent = `${fps} FPS (${latencyMs}ms)`;
    document.getElementById("telemetry-count").textContent = data.detection_count;

    if (data.detections && data.detections.length > 0) {
        const topDet = data.detections[0];
        document.getElementById("telemetry-top-class").textContent = topDet.class_name;
        document.getElementById("telemetry-conf").textContent = `${topDet.confidence_percent}%`;
    } else {
        document.getElementById("telemetry-top-class").textContent = "-";
        document.getElementById("telemetry-conf").textContent = "0%";
    }

    // 4. Live Detections List
    const detUl = document.getElementById("live-det-ul");
    detUl.innerHTML = "";

    if (!data.detections || data.detections.length === 0) {
        detUl.innerHTML = '<li class="empty-list">No active detections</li>';
    } else {
        data.detections.forEach(det => {
            const li = document.createElement("li");
            const colorClass = det.quality === "GOOD" ? "text-good" : "text-bad";
            li.innerHTML = `
                <span>${det.class_name}</span>
                <strong class="${colorClass}">${det.confidence_percent}%</strong>
            `;
            detUl.appendChild(li);
        });
    }
}

function resetLiveHUD() {
    document.getElementById("live-banner-title").textContent = "INACTIVE";
    document.getElementById("live-banner-title").style.color = "var(--text-main)";
    document.getElementById("live-banner-icon").textContent = "⏳";
    document.getElementById("live-banner-desc").textContent = "Open webcam and start inspection to process frames.";
    document.getElementById("telemetry-fps").textContent = "0 FPS / 0ms";
    document.getElementById("telemetry-count").textContent = "0";
    document.getElementById("telemetry-top-class").textContent = "-";
    document.getElementById("telemetry-conf").textContent = "0%";
    document.getElementById("live-det-ul").innerHTML = '<li class="empty-list">No active detections</li>';
}

/* ================= RESET ALL ================= */
function resetAll() {
    selectedFile = null;
    document.getElementById("image-input").value = "";
    document.getElementById("drop-zone-content").style.display = "block";
    document.getElementById("file-preview-strip").style.display = "none";
    document.getElementById("btn-inspect-image").disabled = true;
    document.getElementById("image-results-container").style.display = "none";

    if (webcamStream) {
        closeWebcam();
    }
}
