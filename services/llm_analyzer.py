from google import genai                            # API untuk mengirim prompt ke Google Gemini dan menerima hasil analisis
import re                                         # Regex : Mencari pola teks, agar skor dalam format score/5
from google.genai.errors import ClientError, ServerError

# error handling
class GeminiQuotaExceededError(Exception):
    pass

class GeminiServiceUnavailableError(Exception):
    pass

# Class untuk LLM analyzer dengan Gemini AI

class ResumeReportAnalyzer:
    def __init__(self, api_key : str, model = 'gemini-3.8-flash'):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    # Method untuk get_report
    def generate_report(self, resume: str, job_desc: str) -> str:
        # Prompting untuk Gemini
        prompt = f'''
        ## Context
        You are an AI Resume Analyzer. Evaluate how well a candidate's resume matches a Job Description (JD).

        Instructions:
        1. Identify the 5-10 most important requirements from the JD, covering relevant skills, experience, education, and qualifications.
        2. Evaluate each requirement using only explicit evidence from the resume. Do not invent information. Missing evidence means "Insufficient evidence," not confirmed lack of ability.
        3. Assign a score:
        - 5 = Excellent match
        - 4 = Good match
        - 3 = Partial match
        - 2 = Limited match
        - 1 = No demonstrated match
        4. Use 🟢 for scores 4-5, 🟠 for score 3, and 🔴 for scores 1-2.
        5. Keep each evaluation concise: one sentence for resume evidence and one sentence for the assessment.
        6. Avoid repetition, lengthy introductions, and irrelevant details.

        Resume:
        {resume}

        Job Description:
        {job_desc}

        Output format:

        ## 1. Resume-Job Description Match Analysis

        For each requirement, use:

        **[Emoji] [Score]/5 — [Requirement]**
        - **Resume Evidence:** [Specific evidence or "Insufficient evidence."]
        - **Evaluation:** [Brief justification for the score.]

        ## 2. Assessment Summary

        Provide 2-3 sentences describing the strongest matches and overall suitability. Do not calculate an overall match percentage.'''

        # Mengirimkan response ke Gemini API menggunakan client
        try:
            response = self.client.models.generate_content(
                        contents=prompt,
                        model=self.model
                        )
            return response.text
        
        except ClientError as e:

            if e.code == 429:
                raise GeminiQuotaExceededError(
                    "Gemini API quota exceeded. Please check your API usage."
                ) from e

            raise

        except ServerError as e:
            if e.code == 503:
                raise GeminiServiceUnavailableError(
                    "Gemini server is temporarily unavailable."
                ) from e

            raise

    # Method untuk extract score: x/5, ambil int x dari hasil report dari gemini AI
    @staticmethod
    def extract_scores(text: str) -> list[float]:
        # Regex : Regular expression pattern untuk menemukan pola score dalam hasil prompt dengan format float/5
        pattern = r'(\d+(?:\.\d+)?)/5'
        matches = re.findall(pattern, text)
        scores = [float(match) for match in matches] # casting menjadi float untuk memastikan score sudah dalam float
        return scores
