from pdfminer.high_level import extract_text      # Mengekstrak teks dari file PDF 

# 7. Class untuk PDFExtractor untuk ambil staticmethod

class PDFExtractor:
    @staticmethod
    def extract_text_from_pdf(uploaded_file) -> str:
        try:
            return extract_text(uploaded_file)
        except Exception as exc:
            return f"Failed to extract text from the PDF file : {str(exc)}."