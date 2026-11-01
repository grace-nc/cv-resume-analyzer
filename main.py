
# Mini Project: Machine Learning and AI Powered CV/Resume Analyzer & Semantic Search

# 1. IMPORT LIBRARIES

import streamlit as st
import pandas as pd
import os

from dotenv import load_dotenv

from services.pdf_extractor import PDFExtractor
from services.similarity_calculator_service import SimilarityCalculator
from services.semantic_search_service import SemanticSearch

from services.llm_analyzer import (
    ResumeReportAnalyzer,
    GeminiQuotaExceededError,
    GeminiServiceUnavailableError
)


# 2. PAGE CONFIGURATION
st.set_page_config(
    page_title="AI Resume Analyzer & Semantic Search",
    page_icon="📝",
    layout="wide"
)


# 3. ENVIRONMENT SETUP


# Load environment variables
load_dotenv()

# Retrieve Gemini API Key
api_key = os.getenv("GEMINI_API_KEY")

# Validate API Key
if not api_key:
    st.error("GEMINI_API_KEY not found. Please check your .env file.")
    st.stop()


# 4. SESSION STATE

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False

if "resume" not in st.session_state:
    st.session_state.resume = ""

if "job_desc" not in st.session_state:
    st.session_state.job_desc = ""


# 5. LOAD DATA FOR SEMANTIC SEARCH

@st.cache_resource # agar tidak memuat ulang dataset dan model setiap kali rerun
def load_resume_data():

    # Load resume dataset
    df = pd.read_csv(
        "dataset/resume-dataset-cleaned.csv",
        dtype={"id": "string"}
    )

    # Validate required columns
    required_columns = [
        "id",
        "category",
        "job_title",
        "text"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Column '{column}' not found. "
                f"Available columns: {df.columns.tolist()}"
            )

    # Remove missing values
    df = df.dropna(
        subset=required_columns
    ).reset_index(drop=True)

    # Clean text columns
    text_columns = [
        "id",
        "category",
        "job_title",
        "text"
    ]

    for column in text_columns:
        df[column] = df[column].astype(str).str.strip()

    # Remove empty values
    for column in text_columns:
        df = df[
            df[column] != ""
        ]

    df = df.reset_index(drop=True)

    # Check whether dataset is empty
    if df.empty:
        raise ValueError("No valid resume data found in the dataset.")

    # Initialize Semantic Search
    search_engine = SemanticSearch()

    # Generate resume embeddings
    search_engine.fit(
        documents=df["text"].tolist(),
        ids=df["id"].tolist(),
        job_titles=df["job_title"].tolist()
    )

    return df, search_engine


# 6. SIDEBAR NAVIGATION

with st.sidebar:

    st.title("📌 Menu")
    st.divider()

    st.button(
        "🏠 Home",
        use_container_width=True,
        on_click=lambda: st.session_state.update(page="Home")
    )

    st.button(
        "📝 CV Analyzer",
        use_container_width=True,
        on_click=lambda: st.session_state.update(page="CV Analyzer")
    )

    st.button(
        "🔍 Semantic Search",
        use_container_width=True,
        on_click=lambda: st.session_state.update(page="Semantic Search")
    )


# 7. HOME PAGE

if st.session_state.page == "Home":

    st.title("🧠⚙️ AI-Powered CV/Resume Screening Tools")

    st.markdown("""
    ### Welcome!

    This application integrates Machine Learning and Artificial Intelligence (AI)
    to provide advanced professional tools:

    1. **CV Analyzer:** Evaluate how closely a candidate's resume matches a
       job description using Machine Learning-based similarity scoring and
       Google Gemini LLM/AI evaluation.

    2. **Semantic Search:** Search through a resume dataset by understanding
       contextual meaning using Machine Learning and Deep Learning-based frameworks.

    **To get started, select a button from the sidebar.**
    """)


# 8. CV ANALYZER PAGE

elif st.session_state.page == "CV Analyzer":

    st.title("📝 Resume Analyzer", text_alignment="center")

    st.header(
        "Analyze, Evaluate & Score CV/Resume",
        text_alignment="center"
    )

    st.subheader(
        body="Using Machine Learning-Based Similarity Scoring and AI-Powered Evaluation & Assessment",
        text_alignment="center"
    )

    st.divider()


    # RESUME INPUT FORM

    if not st.session_state.form_submitted:

        with st.form("form_container"):

            resume = st.file_uploader(
                label="Upload CV/Resume (Only in PDF Format)",
                type="pdf",
                accept_multiple_files=False
            )

            job_desc = st.text_area(
                label="Enter Job Description",
                placeholder="Enter the full job description here"
            )

            submitted = st.form_submit_button(
                "Analyze Resume",
                type="primary",
                use_container_width=True
            )

            if submitted:

                if job_desc.strip() and resume:

                    st.info(
                        "Both CV and Job Description are submitted. "
                        "Extracting Information..."
                    )

                    extracted = PDFExtractor.extract_text_from_pdf(resume)

                    if extracted.startswith("Error extracting"):

                        st.error(extracted)

                    elif not extracted.strip():

                        st.error(
                            "No readable text was found in the uploaded PDF."
                        )

                    else:

                        st.session_state.resume = extracted

                        st.session_state.job_desc = job_desc

                        st.session_state.form_submitted = True

                        st.rerun()

                else:

                    st.warning(
                        "Error. Please provide both CV/Resume (PDF) "
                        "and Job Description."
                    )


    # GENERATE RESUME SCORE

    if st.session_state.form_submitted:

        generate_score = st.status(
            "Analyzing resume and generating scores... Please wait."
        )


        # MACHINE LEARNING SIMILARITY SCORE
        

        similarity_calculator = SimilarityCalculator()

        similarity_score = similarity_calculator.calculate(
            st.session_state.resume,
            st.session_state.job_desc
        )


        # Create three columns
        col1, col2, col3 = st.columns(
            3,
            gap="medium",
            border=True,
            vertical_alignment="center"
        )


        
        # COLUMN 1: SIMILARITY SCORE

        with col1:

            st.markdown(
                f"""
                <h3 style='text-align: center;'>
                    Machine Learning-based Similarity Score:
                    <br>
                    {similarity_score:.2f} / 1
                    <br>
                    <span style='font-size: 22px; color: #4CAF50;'>
                        ({similarity_score * 100:.2f}%)
                    </span>
                </h3>
                """,
                unsafe_allow_html=True
            )


        # GEMINI LLM AI ANALYSIS

        analyzer = ResumeReportAnalyzer(api_key=api_key)

        try:

            report = analyzer.generate_report(
                st.session_state.resume,
                st.session_state.job_desc
            )

        except GeminiQuotaExceededError:

            generate_score.update(
                label="Analysis failed due to API free quota limit.",
                state="error",
                expanded=False
            )

            st.error(
                "⚠️ Gemini API free quota exceeded. "
                "You have reached your daily Free Tier limit. "
                "Please try again after your quota resets."
            )

            st.stop()


        except GeminiServiceUnavailableError:

            generate_score.update(
                label="Analysis failed due to Gemini server error.",
                state="error",
                expanded=False
            )

            st.error(
                "⚠️ Gemini server is currently unavailable. "
                "Please try again later."
            )

            st.stop()


        except Exception as e:

            generate_score.update(
                label="Analysis failed due to unexpected error.",
                state="error",
                expanded=False
            )

            st.error(f"⚠️ Unexpected error: {e}")

            st.stop()


        if not report:

            generate_score.update(
                label="Analysis failed due to unexpected error.",
                state="error",
                expanded=False
            )

            st.error(
                "⚠️ Gemini server is currently experiencing high demand. "
                "The server is temporarily unavailable. "
                "Please try again later."
            )

            st.stop()


        # EXTRACT AI SCORES

        report_ai_scores = analyzer.extract_scores(report)

        if report_ai_scores:

            avg_score_ai = sum(report_ai_scores) / len(report_ai_scores)

        else:

            avg_score_ai = None


        # COLUMN 2: AI SCORE

        with col2:

            if avg_score_ai is not None:

                st.markdown(
                    f"""
                    <h3 style='text-align: center;'>
                        AI LLM-based Score (Avg):
                        <br>
                        {avg_score_ai:.2f} / 5
                    </h3>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <h3 style='text-align: center;'>
                        AI LLM-based Score (Avg):
                        <br>
                        N/A
                    </h3>
                    """,
                    unsafe_allow_html=True
                )


        # COLUMN 3: OVERALL ASSESSMENT

        with col3:

            st.subheader(
                "Overall Assessment",
                text_alignment="center"
            )

            # Determine ML Similarity Status
            if similarity_score >= 0.8:
                similarity_status = "High"

            elif similarity_score >= 0.5:
                similarity_status = "Moderate"

            else:
                similarity_status = "Low"


            # Determine AI Score Status
            if avg_score_ai is None:
                ai_status = None

            elif avg_score_ai >= 4:
                ai_status = "High"

            elif avg_score_ai >= 3:
                ai_status = "Moderate"

            else:
                ai_status = "Low"


            # Determine Overall Assessment
            if ai_status is None:
                emoji = "⚪"
                status = "AI Assessment Unavailable"

            elif similarity_status == "High" and ai_status == "High":
                emoji = "🟢"
                status = "Strong Match"

            elif similarity_status == "Moderate" and ai_status == "Moderate":
                emoji = "🟠"
                status = "Moderate Match"

            elif similarity_status == "Low" and ai_status == "Low":
                emoji = "🔴"
                status = "Low Match"

            else:
                emoji = "⚪"
                status = "Mixed Results"


            st.markdown(
                f"""
                <div style='text-align: center;'>
                    Overall Assessment summarizes how well the resume
                    aligns with the job description based on semantic
                    similarity and AI evaluation scores.
                    <div style='font-size: 42px;'>{emoji}</div>
                    <p>{status}</p>
                </div>
                """,
                unsafe_allow_html=True
            )


        # UPDATE ANALYSIS STATUS
        generate_score.update(
            label="Successfully Generated Score!",
            state="complete",
            expanded=False
        )


        # AI-GENERATED REPORT

        st.divider()

        st.markdown(
            """
            <h2 style="text-align: center; margin-bottom: 5px;">
                AI-Generated Resume Analysis Report
            </h2>

            <p style="text-align: center; color: gray; font-size: 14px;">
                Resume evaluation based on the job description
            </p>

            <hr style="margin-top: 20px; margin-bottom: 25px;">
            """,
            unsafe_allow_html=True
        )

        with st.container():

            st.markdown(report)

        # DOWNLOAD REPORT
        st.divider()
        st.download_button(
            label="📥 Download Analysis Report",
            data=report,
            file_name="resume_analysis_report.txt",
            type="primary"
        )

        
        # RESET ANALYSIS -> if user wants to analyze another resume, reset the form and session state

        if st.button("Analyze Another Resume"):
            st.session_state.form_submitted = False
            st.session_state.resume = ""
            st.session_state.job_desc = ""
            st.rerun()


# 9. SEMANTIC SEARCH PAGE


elif st.session_state.page == "Semantic Search":

    st.title("🔍 Semantic Search")

    # CONTAINER: INTRODUCTION TO MINI APP

    with st.container(border=True):

        st.markdown("""
        This mini application demonstrates a **Semantic Search Engine**
        using Machine Learning and Deep Learning-based frameworks.

        Unlike traditional keyword-based search which focuses on matching exact terms, 
        Semantic Search understands the contextual meaning of words and sentences.
        """)


    
    # LOAD DATASET AND SEMANTIC SEARCH MODEL
    

    try:
        df_resumes, search_engine = load_resume_data()

    except Exception as e:

        st.error(f"Failed to load resume dataset: {e}")

        st.stop()


    # CONTAINER: SEARCH RESUME

    with st.container(border=True):

        st.subheader("🔎 Search Resume")

        query = st.text_input(
            "Enter your search query:",
            placeholder="e.g. I need someone to build my data infrastructure"
        )

        if query and query.strip():

            # Perform semantic search
            search_results = search_engine.search(
                query=query,
                top_k=5
            )

            st.subheader(f"Top Matches for: '{query}'")

            # Display search results
            for rank, result in enumerate(search_results, start=1):

                st.markdown(
                    f"### {rank}. {result['job_title']}"
                )

                st.markdown(
                    f"""
                    **Resume ID:** {result['id']}

                    **Similarity Score:** `{result['similarity_score'] * 100 :.2f}%`
                    """
                )

                # Display full resume dalam expandable section
                with st.expander("View Resume"):
                    st.write(result["resume"])
                st.divider()
