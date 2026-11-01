# 🧠⚙️ AI-Powered CV/Resume Screening & Semantic Search Tools

A multi-page **Streamlit** web application that integrates Machine Learning, Sentence Transformers, and Google Gemini AI to streamline resume screening and advanced semantic search.

---

## 🚀 Key Features

1. **📝 CV Analyzer**
   - **PDF Text Extraction**: Automatically extracts text from uploaded PDF resumes.
   - **Machine Learning Similarity Scoring**: Calculates the semantic similarity score between a resume and a Job Description using Cosine Similarity.
   - **AI-Powered Evaluation (Google Gemini LLM)**: Generates detailed evaluation reports, average scores (1-5 scale), and overall match assessments.
   - **Export Report**: Allows downloading the complete analysis report as a `.txt` file.

2. **🔍 Semantic Search**
   - Powered by Sentence Transformers (`all-MiniLM-L6-v2`) to find the most relevant resumes based on contextual meaning rather than exact keyword matches.

---

## 🛠️ Project Directory Structure

project-folder/
│
├── dataset/
│   └── resume-dataset-cleaned.csv      # Cleaned resume dataset
├── services/
│   ├── pdf_extractor.py                # PDF text extraction utility
│   ├── similarity_calculator_service.py # ML similarity calculation service
│   ├── semantic_search_service.py      # Semantic search engine service
│   └── llm_analyzer.py                 # Google Gemini AI analysis service
│
├── .env                                # Environment variables (API Key)
├── app.py                              # Main Streamlit application file
└── README.md                           # Project documentation

---

## ⚙️ System Requirements & Installation


1. Clone or download this repository to your local machine.

2. Install the required dependencies:
   ```bash
   pip install streamlit pandas python-dotenv sentence-transformers scikit-learn PyPDF2 google-generativeai torch
   ```

3. Configure Environment Variables:
   Create a file named `.env` in the root directory of your project and add your Google Gemini API Key:
   ```env
   GEMINI_API_KEY=your_google_gemini_api_key_here
   ```

4. Prepare Your Dataset:
   Place your dataset file inside the `dataset/` folder named `resume-dataset-cleaned.csv` containing columns: `id`, `category`, `job_title`, and `text`.

---

## ▶️ Running the Application

Execute the following command in your terminal or command prompt within the project directory:

```bash
python -m streamlit run app.py
```

The application will launch automatically in your default browser at `http://localhost:8501`.

---

## 🧩 Tech Stack

- **Frontend & UI:** [Streamlit](https://streamlit.io/)
- **Machine Learning & Deep Learning:** [Sentence-Transformers](https://www.sbert.net/), Scikit-learn, PyTorch
- **LLM / Generative AI:** [Google Gemini API](https://ai.google.dev/)
- **Data Manipulation:** Pandas
