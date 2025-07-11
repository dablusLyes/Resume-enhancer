function extractJobDescription() {
	console.log("Extracting job description...");

	try {
		// Try the provided XPath first
		const jobDetailsElement = document.evaluate(
			'//*[@id="job-details"]',
			document,
			null,
			XPathResult.FIRST_ORDERED_NODE_TYPE,
			null,
		).singleNodeValue;

		if (jobDetailsElement) {
			return {
				success: true,
				content: jobDetailsElement.innerText.trim(),
				source: "job-details",
			};
		}

		// Fallback selectors for different LinkedIn layouts
		const fallbackSelectors = [
			".jobs-search__job-details--container",
			".job-view-layout",
			".jobs-details__main-content",
			".job-details-jobs-unified-top-card__container",
			"[data-job-id]",
			".jobs-description-content__text",
		];

		for (const selector of fallbackSelectors) {
			const element = document.querySelector(selector);
			if (element && element.innerText.trim()) {
				return {
					success: true,
					content: element.innerText.trim(),
					source: selector,
				};
			}
		}

		return {
			success: false,
			error: "Job description not found on this page",
		};
	} catch (error) {
		return {
			success: false,
			error: error.message,
		};
	}
}

// Function to check if current page is a LinkedIn job page
function isLinkedInJobPage() {
	const url = window.location.href;
	console.log("Checking if current page is a LinkedIn job page:", url);
	return url.includes("linkedin.com/jobs/");
}

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
	if (request.action === "extractJobDescription") {
		const result = extractJobDescription();
		sendResponse(result);
	} else if (request.action === "checkLinkedInJob") {
		sendResponse({
			isLinkedInJob: isLinkedInJobPage(),
			url: window.location.href,
		});
	}
});
console.log("nik zebi ");

// Auto-detect when job description changes (for single-page app navigation)
let lastJobContent = "";
const observer = new MutationObserver(() => {
	if (isLinkedInJobPage()) {
		const currentContent = extractJobDescription();
		console.log("Current job description content:", currentContent);

		if (
			currentContent.success &&
			currentContent.content !== lastJobContent
		) {
			lastJobContent = currentContent.content;
			// Notify popup if it's open
			chrome.runtime
				.sendMessage({
					action: "jobDescriptionChanged",
					content: currentContent.content,
				})
				.catch(() => {
					// Popup might not be open, ignore error
				});
		}
	}
});

// Start observing changes
observer.observe(document.body, {
	childList: true,
	subtree: true,
});

console.log("LinkedIn job scraper content script loaded");
