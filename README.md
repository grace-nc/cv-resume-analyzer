# 🧠⚙️ AI-Powered CV/Resume Screening & Semantic Search

A multi-page **Streamlit web application** that integrates Machine Learning, Deep Learning, Sentence Transformers, and Google Gemini AI to streamline resume screening and semantic resume retrieval.

---

## 🚀 Key Features

### 1. 📝 CV Analyzer

- **PDF Text Extraction:** Automatically extracts text from uploaded PDF resumes.
- **Similarity Scoring:** Evaluates resume–Job Description alignment using Sentence Transformers and Cosine Similarity.
- **AI-Powered Evaluation:** Leverages Google Gemini AI (LLM) to generate detailed candidate evaluations, competency scores (1–5 scale), and overall match assessments.
- **Export Analysis Report:** Allows users to download the generated evaluation report as a `.txt` file.

### 2. 🔍 Semantic Search

- **Contextual Resume Retrieval:** Uses Sentence Transformers (`all-MiniLM-L6-v2`) to retrieve and rank relevant resumes based on contextual meaning rather than exact keyword matches.
- **Similarity Ranking:** Calculates cosine similarity between search queries and resume embeddings to identify the Top K most relevant resumes.
- **Resume Details:** Displays candidate IDs, job titles, similarity scores, and resume content.

---

## 🛠️ Project Directory Structure

```text
project-folder/
│
├── dataset/
│   └── resume-dataset-cleaned.csv
│
├── services/
│   ├── __init__.py
│   ├── pdf_extractor.py
│   ├── similarity_calculator_service.py
│   ├── semantic_search_service.py
│   └── llm_analyzer.py
│
├── .env
├── .gitignore
├── requirements.txt
├── app.py
└── README.md
```

---

## ⚙️ System Requirements & Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-project-folder>
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate the virtual environment:

**Windows:**
```bash
venv\Scripts\activate
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory of your project and add your Google Gemini API key:

```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

**Note:** Keep your `.env` file private. Do not commit or upload API keys to GitHub.

### 5. Prepare the Dataset

Place the cleaned dataset inside the `dataset/` folder using the filename:

`resume-dataset-cleaned.csv`

The dataset must contain the following columns:

| Column | Description |
|---|---|
| `id` | Unique resume identifier |
| `category` | Resume/job category |
| `job_title` | Job title associated with the resume |
| `text` | Extracted resume content |

---

## ▶️ Running the Application

Run the following command from the project root directory:

```bash
python -m streamlit run app.py
```

The application will be available in your browser at:

`http://localhost:8501`

---

## 🧰 Technologies & Libraries

| Technology |
|---|
| Python 
| Streamlit |
| Sentence Transformers |
| PyTorch | 
| Scikit-learn |
| Google Gemini AI | 
| PDFMiner | 
| Pandas | 
| Python-dotenv |

---
