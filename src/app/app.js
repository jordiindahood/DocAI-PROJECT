/* ================================================================
   DocAI — Web App Client
   ================================================================ */

const API_BASE = "/api/v1";
const VALID_EXTS = ["png", "jpg", "jpeg", "tiff", "tif", "bmp", "webp"];
const MAX_SIZE_MB = 20;

/* helper: shake an element briefly */
function shake(el) {
    el.style.animation = "none";
    void el.offsetHeight;
    el.style.animation = "shake 0.4s ease";
    el.addEventListener("animationend", () => { el.style.animation = ""; }, { once: true });
}

/* ── DOM refs ──────────────────────────────────────────────────── */
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const fileInfo = document.getElementById("fileInfo");
const previewImg = document.getElementById("previewImg");
const btnPdf = document.getElementById("btnPdf");
const btnJson = document.getElementById("btnJson");
const spinner = document.getElementById("spinner");
const spinnerText = document.getElementById("spinnerText");
const resultSection = document.getElementById("resultSection");
const resultTitle = document.getElementById("resultTitle");
const resultBody = document.getElementById("resultBody");
const healthDot = document.getElementById("healthDot");
const healthLabel = document.getElementById("healthLabel");

let selectedFile = null;
let processing = false;   // guard against double-clicks

/* ── Health Check ──────────────────────────────────────────────── */
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(5000) });
        if (res.ok) {
            healthDot.className = "health-dot ok";
            healthLabel.textContent = "API Online";
        } else {
            throw new Error();
        }
    } catch {
        healthDot.className = "health-dot err";
        healthLabel.textContent = "API Offline";
    }
}

checkHealth();
setInterval(checkHealth, 30000);

/* ── File Handling ─────────────────────────────────────────────── */
dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener("change", () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
});

function handleFile(file) {
    const ext = file.name.split(".").pop().toLowerCase();
    if (!VALID_EXTS.includes(ext)) {
        toast(`Invalid file type (.${ext}). Supported: ${VALID_EXTS.join(", ")}`, "error");
        return;
    }
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
        toast(`File too large (max ${MAX_SIZE_MB} MB)`, "error");
        return;
    }

    selectedFile = file;
    dropzone.classList.add("has-file");

    // show preview thumbnail
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        previewImg.style.display = "block";
    };
    reader.readAsDataURL(file);

    fileInfo.textContent = `${file.name}  (${(file.size / 1024).toFixed(1)} KB)`;

    // activate buttons
    btnPdf.classList.remove("btn-inactive");
    btnJson.classList.remove("btn-inactive");

    // hide previous results
    resultSection.classList.remove("active");
}

/* ── Process: Get PDF ──────────────────────────────────────────── */
btnPdf.addEventListener("click", async () => {
    if (!selectedFile) {
        toast("Please upload an image first", "error");
        shake(dropzone);
        return;
    }
    if (processing) return;
    processing = true;

    showSpinner("Processing document — generating PDF…");
    setButtonsInactive(true);

    try {
        const form = new FormData();
        form.append("image", selectedFile);

        const res = await fetch(`${API_BASE}/process`, { method: "POST", body: form });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || "Processing failed");
        }

        const blob = await res.blob();
        const url = URL.createObjectURL(blob);

        resultTitle.textContent = "Generated PDF";
        resultBody.innerHTML = "";

        const iframe = document.createElement("iframe");
        iframe.className = "pdf-frame";
        iframe.src = url;
        resultBody.appendChild(iframe);

        const a = document.createElement("a");
        a.href = url;
        a.download = selectedFile.name.replace(/\.[^.]+$/, "") + "_reconstructed.pdf";
        a.className = "download-link";
        a.innerHTML = "⬇ Download PDF";
        resultBody.appendChild(a);

        resultSection.classList.add("active");
        toast("PDF generated successfully!", "success");
    } catch (err) {
        toast(err.message, "error");
    } finally {
        hideSpinner();
        setButtonsInactive(false);
        processing = false;
    }
});

/* ── Process: Get JSON ─────────────────────────────────────────── */
btnJson.addEventListener("click", async () => {
    if (!selectedFile) {
        toast("Please upload an image first", "error");
        shake(dropzone);
        return;
    }
    if (processing) return;
    processing = true;

    showSpinner("Processing document — extracting layout…");
    setButtonsInactive(true);

    try {
        const form = new FormData();
        form.append("image", selectedFile);

        const res = await fetch(`${API_BASE}/process/json`, { method: "POST", body: form });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || "Processing failed");
        }

        const data = await res.json();

        resultTitle.textContent = "Layout Structure (JSON)";
        resultBody.innerHTML = "";

        // Reconstructed text (if present)
        const reconText = data?.layout?.reconstructed_text;
        if (reconText) {
            const label = document.createElement("div");
            label.className = "card-title";
            label.style.marginTop = "0";
            label.textContent = "Reconstructed Text";
            resultBody.appendChild(label);

            const pre = document.createElement("div");
            pre.className = "recon-text";
            pre.textContent = reconText;
            resultBody.appendChild(pre);
        }

        // Full JSON tree
        const treeLabel = document.createElement("div");
        treeLabel.className = "card-title";
        treeLabel.textContent = "Full Response";
        resultBody.appendChild(treeLabel);

        const tree = document.createElement("div");
        tree.className = "json-tree";
        tree.innerHTML = renderJsonTree(data);
        resultBody.appendChild(tree);

        tree.querySelectorAll(".json-toggle").forEach((el) => {
            el.addEventListener("click", () => {
                el.classList.toggle("open");
                const children = el.nextElementSibling;
                if (children) children.classList.toggle("open");
            });
        });

        resultSection.classList.add("active");
        toast("Layout extracted!", "success");
    } catch (err) {
        toast(err.message, "error");
    } finally {
        hideSpinner();
        setButtonsInactive(false);
        processing = false;
    }
});

/* ── JSON Tree Renderer ────────────────────────────────────────── */
function renderJsonTree(obj, depth = 0) {
    if (obj === null) return '<span class="json-null">null</span>';
    if (typeof obj === "boolean") return `<span class="json-bool">${obj}</span>`;
    if (typeof obj === "number") return `<span class="json-number">${obj}</span>`;
    if (typeof obj === "string") {
        const escaped = escapeHtml(obj);
        if (escaped.length > 200) {
            return `<span class="json-string">"${escaped.substring(0, 200)}…"</span>`;
        }
        return `<span class="json-string">"${escaped}"</span>`;
    }

    if (Array.isArray(obj)) {
        if (obj.length === 0) return "[]";
        const collapsed = depth > 1;
        let html = `<span class="json-toggle ${collapsed ? "" : "open"}">[${obj.length} items]</span>`;
        html += `<div class="json-children ${collapsed ? "" : "open"}">`;
        obj.forEach((item, i) => {
            html += `${renderJsonTree(item, depth + 1)}${i < obj.length - 1 ? "," : ""}\n`;
        });
        html += "</div>";
        return html;
    }

    const keys = Object.keys(obj);
    if (keys.length === 0) return "{}";
    const collapsed = depth > 1;
    let html = `<span class="json-toggle ${collapsed ? "" : "open"}">{${keys.length} keys}</span>`;
    html += `<div class="json-children ${collapsed ? "" : "open"}">`;
    keys.forEach((key, i) => {
        html += `<span class="json-key">"${escapeHtml(key)}"</span>: ${renderJsonTree(obj[key], depth + 1)}${i < keys.length - 1 ? "," : ""}\n`;
    });
    html += "</div>";
    return html;
}

function escapeHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

/* ── UI Helpers ────────────────────────────────────────────────── */
function showSpinner(msg) {
    spinnerText.textContent = msg;
    spinner.classList.add("active");
    resultSection.classList.remove("active");
}

function hideSpinner() {
    spinner.classList.remove("active");
}

function setButtonsInactive(state) {
    btnPdf.classList.toggle("btn-inactive", state);
    btnJson.classList.toggle("btn-inactive", state);
}

/* ── Toast Notifications ───────────────────────────────────────── */
function toast(message, type = "info") {
    const container =
        document.querySelector(".toast-container") || createToastContainer();
    const el = document.createElement("div");
    el.className = `toast ${type}`;
    el.textContent = message;
    container.appendChild(el);
    setTimeout(() => el.remove(), 4000);
}

function createToastContainer() {
    const c = document.createElement("div");
    c.className = "toast-container";
    document.body.appendChild(c);
    return c;
}
