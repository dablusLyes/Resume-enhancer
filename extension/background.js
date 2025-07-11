// Background script for side panel
chrome.runtime.onInstalled.addListener(() => {
	// Enable side panel on all tabs
	chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });
});
console.log("Side panel background script loaded");

// Handle action click to open side panel
chrome.action.onClicked.addListener((tab) => {
	chrome.sidePanel.open({ tabId: tab.id });
});

// Optional: Auto-open side panel when on LinkedIn job pages
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
	if (
		changeInfo.status === "complete" &&
		tab.url &&
		tab.url.includes("linkedin.com/jobs/view/")
	) {
		chrome.sidePanel.open({ tabId: sender.tab.id });

		// Optionally auto-open side panel on LinkedIn job pages
		// chrome.sidePanel.open({ tabId: tabId });
	}
});

// Handle messages from content script or side panel
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
	if (request.action === "openSidePanel") {
		console.log("Received message:", request, "from sender:", sender);

		chrome.sidePanel.open({ tabId: sender.tab.id });
	}
});
