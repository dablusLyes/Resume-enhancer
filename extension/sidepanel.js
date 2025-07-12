// Updated popup.js with LinkedIn job description scraping and semantic evaluation
const SERVER_URL = "http://localhost:5000";
console.log("Initializing CVIMG side panel...");

let selectedFiles = {
	pdf: null,
	jobDescription: null, // Changed from text to jobDescription
};

let currentSessionId = null;
let isProcessing = false;
let lastEvaluationResult = null;

// DOM elements
const serverStatus = document.getElementById("server-status");
const pdfFileInput = document.getElementById("pdf-file");
const pdfFileName = document.getElementById("pdf-file-name");
const jobDescriptionBtn = document.getElementById("job-description-btn");
const jobDescriptionStatus = document.getElementById("job-description-status");
const processBtn = document.getElementById("process-btn");
const resetBtn = document.getElementById("reset-btn");
const statusDiv = document.getElementById("status");

// Initialize popup
document.addEventListener("DOMContentLoaded", function () {
	checkServerStatus();
	setupEventListeners();
	checkCurrentPage();
});

function setupEventListeners() {
	// File input handlers
	pdfFileInput.addEventListener("change", function (e) {
		handleFileSelection(e, "pdf", pdfFileName);
	});

	// Job description scraping button
	jobDescriptionBtn.addEventListener("click", scrapeJobDescription);

	// Button handlers
	processBtn.addEventListener("click", processPDF);
	resetBtn.addEventListener("click", resetUI);

	// Listen for job description changes
	chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
		if (request.action === "jobDescriptionChanged") {
			updateJobDescriptionDisplay(request.content);
		}
	});
}

function handleFileSelection(event, type, displayElement) {
	const file = event.target.files[0];
	if (file) {
		selectedFiles[type] = file;
		displayElement.textContent = `Selected: ${file.name}`;
		displayElement.style.color = "#27ae60";
	} else {
		selectedFiles[type] = null;
		displayElement.textContent = "";
	}

	updateProcessButton();
	// Check if we can run semantic evaluation
	checkAndRunSemanticEvaluation();
}

async function checkCurrentPage() {
	try {
		const [tab] = await chrome.tabs.query({
			active: true,
			currentWindow: true,
		});

		// Check if we're on a LinkedIn job page
		chrome.tabs.sendMessage(
			tab.id,
			{ action: "checkLinkedInJob" },
			(response) => {
				console.log("Current page response:", response);

				if (chrome.runtime.lastError) {
					// Content script not loaded, show manual mode
					showManualMode();
					return;
				}

				if (response && response.isLinkedInJob) {
					showLinkedInMode();
				} else {
					showManualMode();
				}
			},
		);
	} catch (error) {
		console.error("Error checking current page:", error);
		showManualMode();
	}
}

function showLinkedInMode() {
	jobDescriptionBtn.style.display = "block";
	jobDescriptionBtn.disabled = false;
	jobDescriptionStatus.textContent =
		"Click to scrape job description from current LinkedIn page";
	jobDescriptionStatus.style.color = "#3498db";
}

function showManualMode() {
	jobDescriptionBtn.style.display = "none";
	jobDescriptionStatus.textContent =
		"Navigate to a LinkedIn job page to scrape job description";
	jobDescriptionStatus.style.color = "#e74c3c";
}

async function scrapeJobDescription() {
	console.log("Scraping job description...");

	jobDescriptionBtn.disabled = true;
	jobDescriptionBtn.innerHTML = '<div class="spinner"></div>Scraping...';
	jobDescriptionStatus.textContent = "Extracting job description...";
	jobDescriptionStatus.style.color = "#f39c12";

	try {
		const [tab] = await chrome.tabs.query({
			active: true,
			currentWindow: true,
		});

		chrome.tabs.sendMessage(
			tab.id,
			{ action: "extractJobDescription" },
			(response) => {
				if (chrome.runtime.lastError) {
					jobDescriptionStatus.textContent =
						"Error: Content script not loaded. Please refresh the LinkedIn page.";
					jobDescriptionStatus.style.color = "#e74c3c";
					resetJobDescriptionButton();
					return;
				}

				if (response && response.success) {
					// Create a virtual "file" object for the job description
					const jobDescriptionText = response.content;
					selectedFiles.jobDescription = {
						content: jobDescriptionText,
						name: "LinkedIn Job Description",
						size: jobDescriptionText.length,
					};

					updateJobDescriptionDisplay(jobDescriptionText);
					updateProcessButton();
					// Check if we can run semantic evaluation
					checkAndRunSemanticEvaluation();
				} else {
					jobDescriptionStatus.textContent = `Error: ${
						response.error || "Failed to extract job description"
					}`;
					jobDescriptionStatus.style.color = "#e74c3c";
				}

				resetJobDescriptionButton();
			},
		);
	} catch (error) {
		console.error("Error scraping job description:", error);
		jobDescriptionStatus.textContent = "Error scraping job description";
		jobDescriptionStatus.style.color = "#e74c3c";
		resetJobDescriptionButton();
	}
}

function updateJobDescriptionDisplay(content) {
	const preview =
		content.length > 100 ? content.substring(0, 100) + "..." : content;
	jobDescriptionStatus.innerHTML = `
		<div style="color: #27ae60; font-weight: bold;">✓ Job Description Extracted</div>
		<div style="font-size: 11px; color: #7f8c8d; margin-top: 5px; background: #f8f9fa; padding: 5px; border-radius: 3px; border-left: 3px solid #27ae60;">
			${preview}
		</div>
		<div style="font-size: 10px; color: #95a5a6; margin-top: 3px;">
			${content.length} characters extracted
		</div>
	`;
	checkAndRunSemanticEvaluation();
}

function resetJobDescriptionButton() {
	jobDescriptionBtn.disabled = false;
	jobDescriptionBtn.textContent = "Extract Job Description";
}

function updateProcessButton() {
	const canProcess =
		selectedFiles.pdf && selectedFiles.jobDescription && !isProcessing;
	processBtn.disabled = !canProcess;

	if (isProcessing) {
		processBtn.textContent = "Processing...";
	} else if (canProcess) {
		processBtn.textContent = "Process PDF";
	} else {
		processBtn.textContent = "Select PDF and extract job description";
	}
}

async function checkAndRunSemanticEvaluation() {
	// Only run if both PDF and job description are available
	if (!selectedFiles.pdf || !selectedFiles.jobDescription) {
		return;
	}

	await runSemanticEvaluation();
}

async function runSemanticEvaluation() {
	try {
		// Show evaluation status
		showEvaluationStatus("Analyzing CV compatibility...", "processing");

		// Create form data
		const formData = new FormData();
		formData.append("resume_file", selectedFiles.pdf);

		// Create a blob for the job description text
		const jobDescriptionBlob = new Blob(
			[selectedFiles.jobDescription.content],
			{
				type: "text/plain",
			},
		);
		formData.append(
			"job_description",
			jobDescriptionBlob,
			"job_description.txt",
		);

		// Send to server
		const response = await fetch(`${SERVER_URL}/evaluate-ats`, {
			method: "POST",
			body: formData,
		});

		const result = await response.json();
		console.log("Semantic evaluation result:", result);

		if (response.ok && result.success) {
			lastEvaluationResult = result;
			showEvaluationResult(result);
		} else {
			throw new Error(result.error || "Evaluation failed");
		}
	} catch (error) {
		console.error("Semantic evaluation error:", error);
		showEvaluationStatus(`Evaluation error: ${error.message}`, "error");
	}
}

function showEvaluationStatus(message, type) {
	// Create or update evaluation status area
	let evalStatusDiv = document.getElementById("evaluation-status");
	if (!evalStatusDiv) {
		evalStatusDiv = document.createElement("div");
		evalStatusDiv.id = "evaluation-status";
		evalStatusDiv.className = "evaluation-status";

		// Insert after job description status
		const jobDescStatus = document.getElementById("job-description-status");
		jobDescStatus.parentNode.insertBefore(
			evalStatusDiv,
			jobDescStatus.nextSibling,
		);
	}

	evalStatusDiv.innerHTML = `
		<div class="evaluation-message ${type}">
			${type === "processing" ? '<div class="spinner"></div>' : ""}
			${message}
		</div>
	`;
}

function showEvaluationResult(result) {
	const { similarity_score, interpretation, evaluation_results } = result;
	console.log(evaluation_results);

	// Create evaluation result display
	let evalStatusDiv = document.getElementById("evaluation-status");
	if (!evalStatusDiv) {
		evalStatusDiv = document.createElement("div");
		evalStatusDiv.id = "evaluation-status";
		evalStatusDiv.className = "evaluation-status";

		// Insert after job description status
		const jobDescStatus = document.getElementById("job-description-status");
		jobDescStatus.parentNode.insertBefore(
			evalStatusDiv,
			jobDescStatus.nextSibling,
		);
	}

	// Get color based on similarity level
	const getScoreColor = (score) => {
		if (score > 0.8) return "#27ae60"; // Green
		if (score > 0.6) return "#2ecc71"; // Light green
		if (score > 0.4) return "#f39c12"; // Orange
		if (score > 0.2) return "#e67e22"; // Dark orange
		return "#e74c3c"; // Red
	};

	// Get emoji based on level
	const getScoreEmoji = (level) => {
		switch (level) {
			case "very_high":
				return "🎯";
			case "high":
				return "✅";
			case "moderate":
				return "⚠️";
			case "low":
				return "❌";
			case "very_low":
				return "🚫";
			default:
				return "📊";
		}
	};

	const scoreColor = getScoreColor(similarity_score);
	const scoreEmoji = getScoreEmoji(interpretation.level);

	evalStatusDiv.innerHTML = `
		<div class="evaluation-result">
			<div class="score-header" style="color: ${scoreColor};">
				${scoreEmoji} <strong>Compatibility Score: ${Math.round(
		similarity_score * 100,
	)}%</strong>
			</div>
			<div class="score-bar">
				<div class="score-fill" style="width: ${
					similarity_score * 100
				}%; background-color: ${scoreColor};"></div>
			</div>
			<div class="interpretation">
				<div class="interpretation-message">${interpretation.message}</div>
				<div class="interpretation-recommendation">${
					interpretation.recommendation
				}</div>
			</div>
		</div>
	`;
}

async function checkServerStatus() {
	try {
		const response = await fetch(`${SERVER_URL}/health`);
		const data = await response.json();

		if (response.ok && data.status === "healthy") {
			serverStatus.className = "server-status server-online";
			serverStatus.innerHTML = "✅ Server Online";
		} else {
			throw new Error("Server unhealthy");
		}
	} catch (error) {
		serverStatus.className = "server-status server-offline";
		serverStatus.innerHTML = "❌ Server Offline";
		processBtn.disabled = true;
		processBtn.textContent = "Server Offline";
	}
}

async function processPDF() {
	if (!selectedFiles.pdf || !selectedFiles.jobDescription) {
		showStatus("Please select PDF and extract job description", "error");
		return;
	}

	if (isProcessing) {
		return; // Prevent double-processing
	}

	// Set processing state
	isProcessing = true;
	currentSessionId = null;
	updateProcessButton();

	// Show processing status with loading indicator
	showStatus("Processing PDF... Please wait", "processing", true);

	try {
		// Create form data
		const formData = new FormData();
		formData.append("pdf_file", selectedFiles.pdf);

		// Create a blob for the job description text
		const jobDescriptionBlob = new Blob(
			[selectedFiles.jobDescription.content],
			{
				type: "text/plain",
			},
		);
		formData.append("text_file", jobDescriptionBlob, "job_description.txt");

		// Send to server
		const response = await fetch(`${SERVER_URL}/process-pdf`, {
			method: "POST",
			body: formData,
		});

		const result = await response.json();
		console.log("Processing result:", result);

		if (response.ok && result.success) {
			currentSessionId = result.session_id;

			// Show success message with download button
			showStatus("PDF processed successfully!", "success");
			showDownloadButton();

			// Store session ID for cleanup
			chrome.storage.local.set({ lastSessionId: result.session_id });
		} else {
			throw new Error(result.error || "Processing failed");
		}
	} catch (error) {
		console.error("Processing error:", error);
		showStatus(`Error: ${error.message}`, "error");
	} finally {
		isProcessing = false;
		updateProcessButton();
	}
}

function showDownloadButton() {
	// Remove any existing download button
	const existingDownloadBtn = statusDiv.querySelector(".download-btn");
	if (existingDownloadBtn) {
		existingDownloadBtn.remove();
	}

	// Create new download button
	const downloadBtn = document.createElement("button");
	downloadBtn.className = "download-btn";
	downloadBtn.textContent = "Download PDF";
	downloadBtn.onclick = handleDownload;

	statusDiv.appendChild(downloadBtn);
}

async function handleDownload() {
	if (!currentSessionId) {
		showStatus("No PDF ready for download", "error");
		return;
	}

	const downloadBtn = statusDiv.querySelector(".download-btn");
	if (downloadBtn) {
		downloadBtn.disabled = true;
		downloadBtn.innerHTML = '<div class="spinner"></div>Downloading...';
	}

	try {
		await downloadFile(currentSessionId);
	} finally {
		if (downloadBtn) {
			downloadBtn.disabled = false;
			downloadBtn.textContent = "Download PDF";
		}
	}
}

async function cleanupFiles(sessionId) {
	try {
		await fetch(`${SERVER_URL}/cleanup/${sessionId}`, {
			method: "DELETE",
		});
		console.log("Server files cleaned up");
	} catch (error) {
		console.error("Cleanup error:", error);
	}
}

async function downloadFile(sessionId) {
	try {
		const downloadUrl = `${SERVER_URL}/download/${sessionId}`;

		// Use Chrome downloads API
		chrome.downloads.download(
			{
				url: downloadUrl,
				filename: `processed_${Date.now()}.pdf`,
				saveAs: true,
			},
			function (downloadId) {
				if (chrome.runtime.lastError) {
					console.error("Download failed:", chrome.runtime.lastError);
					showStatus("Download failed. Please try again.", "error");
				} else {
					showStatus("Download started successfully!", "success");
					// Clean up server files after download
					setTimeout(() => cleanupFiles(sessionId), 5000);
				}
			},
		);
	} catch (error) {
		console.error("Download error:", error);
		showStatus("Download failed", "error");
	}
}

function showStatus(message, type, showSpinner = false) {
	// Clear previous content
	statusDiv.innerHTML = "";

	// Create status message
	const messageDiv = document.createElement("div");
	messageDiv.className = "status-message";

	if (showSpinner) {
		messageDiv.innerHTML = `<div class="spinner"></div>${message}`;
	} else {
		messageDiv.textContent = message;
	}

	statusDiv.appendChild(messageDiv);
	statusDiv.className = `status ${type}`;
	statusDiv.style.display = "block";
}

function resetUI() {
	// Reset file inputs
	pdfFileInput.value = "";
	pdfFileName.textContent = "";

	// Reset job description
	jobDescriptionStatus.textContent =
		"Click to scrape job description from current LinkedIn page";
	jobDescriptionStatus.style.color = "#3498db";

	// Reset evaluation status
	const evalStatusDiv = document.getElementById("evaluation-status");
	if (evalStatusDiv) {
		evalStatusDiv.remove();
	}

	// Reset state
	selectedFiles = { pdf: null, jobDescription: null };
	currentSessionId = null;
	isProcessing = false;
	lastEvaluationResult = null;

	// Reset UI
	updateProcessButton();
	statusDiv.style.display = "none";
	statusDiv.innerHTML = "";

	// Re-check current page
	checkCurrentPage();
}

// Clean up any previous session files on popup open
chrome.storage.local.get(["lastSessionId"], function (result) {
	if (result.lastSessionId) {
		cleanupFiles(result.lastSessionId);
		chrome.storage.local.remove(["lastSessionId"]);
	}
});
