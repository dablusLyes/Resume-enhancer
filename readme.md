# CVIMG: PDF Resume & LinkedIn Job Description Semantic Analyzer

CVIMG helps you compare your PDF resume to LinkedIn job descriptions using advanced semantic similarity, and overlays job descriptions invisibly onto your resume PDF. It includes a Chrome extension (side panel) and a Python backend.

---

## Features

- **Chrome Extension Side Panel**: Select your PDF resume, extract job descriptions from LinkedIn job pages, and view compatibility scores.
- **Semantic Evaluation**: Uses TF-IDF and Sentence Transformers for advanced semantic similarity. The compatibility score and interpretation are based on the advanced semantic similarity.
- **PDF Processing**: Overlays job description text invisibly onto your resume PDF.
- **Download & Cleanup**: Download processed PDFs and clean up server files automatically or manually.

---

## Prerequisites

- **Python 3.10+**
- **Node.js** (for extension development, if needed)
- **pip** (Python package manager)
- **Google Chrome** (for extension usage)

---

## Backend Setup

1. **Install Python dependencies:**

   ```bash
   pip install -r backend/requirements.txt
   ```

2. **Start the backend server:**

   ```bash
   cd backend
   python server.py
   ```

   The server runs at [http://localhost:5000](http://localhost:5000).

---

## Chrome Extension Setup

1. **Open Chrome and go to** `chrome://extensions`
2. **Enable "Developer mode"**
3. **Click "Load unpacked" and select the `extension` folder**

   - This loads the extension with the side panel.

4. **Pin the extension for easy access**

---

## Usage

1. **Start the backend server** (see above).
2. **Open LinkedIn and navigate to a job page.**
3. **Open the extension side panel:**
   - Click the extension icon or use the side panel shortcut.
4. **Select your PDF resume.**
5. **Click "Extract Job Description"** to scrape the job description from the current LinkedIn job page.
6. **Click "Process PDF"** to analyze compatibility and overlay the job description onto your resume.
7. **View the advanced semantic compatibility score and download the processed PDF.**

---

## Troubleshooting

- **Server Offline?**  
  Make sure the backend server is running (`python server.py`).
- **Job Description Not Extracted?**  
  Ensure you're on a LinkedIn job page and refresh if needed.
- **Service Worker Inactive?**  
  Reload the extension in Chrome.
- **Dependencies Missing?**  
  Install all Python packages listed in `backend/requirements.txt`.

---

## File Structure

```
backend/
  server.py
  process.py
  requirements.txt
  uploads/
  outputs/
extension/
  sidepanel.html
  sidepanel.js
  content-script.js
  manifest.json
  background.js
  icons/
```

---

## Development Notes

- **PDFs and job descriptions are temporarily stored in `backend/uploads/` and processed PDFs in `backend/outputs/`.**
- **Processed PDFs are available for download via the extension.**
- **Cleanup is automatic after download or can be triggered manually.**
- **Semantic evaluation now returns the advanced similarity score and interpretation.**

---

## License

MIT

---

## Authors

- [DABLA](https://github.com/dablusLyes)