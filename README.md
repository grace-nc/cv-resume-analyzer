# 🧠⚙️ AI-Powered CV/Resume Screening & Semantic Search Tools

A multi-page Streamlit web application that integrates Machine Learning, Sentence Transformers, and Google Gemini AI to streamline resume screening and advanced semantic search tasks.

---

## 🚀 Key Features

1. **📝 CV Analyzer**
   - **PDF Extraction**: Automatically extracts text from uploaded PDF CVs/Resumes.
   - **Machine Learning Similarity Scoring**: Calculates the semantic similarity score between a resume and a job description using Cosine Similarity.
   - **AI-Powered Evaluation (Google Gemini LLM)**: Generates a detailed evaluation report, average score (scale 1–5), and an overall assessment status.
   - **Export Report**: Downloadable text report of the analysis results.

2. **🔍 Semantic Search**
   - Context-based search using Sentence Transformers (`all-MiniLM-L6-v2`) to retrieve the most relevant resumes from a dataset based on contextual meaning rather than exact keyword matches.

---

## 🛠️ Project Directory Structure

```text
project-folder/
│
├── dataset/
│   └── resume-dataset-cleaned.csv      # Cleaned resume dataset
├── services/
│   ├── pdf_extractor.py                # PDF text extraction service
│   ├── similarity_calculator_service.py # ML similarity calculation service
│   ├── semantic_search_service.py      # Semantic search engine service
│   └── llm_analyzer.py                 # Google Gemini AI integration service
│
├── .env                                # Environment variables (API Key)
├── app.py                              # Main Streamlit application file
└── README.md                           # Project documentation
