from sklearn.metrics.pairwise import cosine_similarity     # Metrics of machine learning: Menghitung kemiripan antara dua embeddings (vektor numerik), pakai cosine math di mana kalau sudutnya antara 2 vektor mendekati nol berarti tingkat kemiripannya tinggi, Cosine similarity mengabaikan panjang teks dan fokus pada orientasi kata yang digunakan.
from sentence_transformers import SentenceTransformer      # NLP/Deep Learning : Mengubah teks CV dan JD menjadi representasi angka atau embeddings/vektor.
# hasil cosine similarity berkisar dari (-1) - 0 - 1, jika nilai 0 maka tidak ada kesamaan sama sekali (sudut 90), jika -1 maka sudut 180 derajat -> vektor berlawanan arah

# 8. Class  untuk mengkalkulasikan similaritas dengan cosine similarity
# text1 = resume, text2 = job description

class SimilarityCalculator:
    def __init__(self, model_name : str = 'sentence-transformers/all-mpnet-base-v2') :
    
        self.model = SentenceTransformer(model_name) # sentence transformer tipe all-mpnet-base-v2 punya kemampuan luar biasa untuk memahami teks dan mengubahnya menjadi angka dibanding model lainnya
        # encoding: mengubah dari text ke bentuk vektor numerik untuk semantic search dan similarity scoring
        # semantic search: sentence Transformers yang mengubah teks (kalimat atau paragraf pendek) menjadi angka vektor (disebut embeddings) berukuran 768 dimensi
        # pencarian Semantik (Semantic Search): Mencari data berdasarkan makna arti kata, bukan cuma kata kunci yang sama persis.
        # diantaranya - pengelompokan (Clustering): Mengelompokkan teks yang punya topik atau arti serupa ke dalam satu kluster.
        # - kemiripan kalimat (Sentence Similarity): Mengukur seberapa mirip arti dari dua kalimat yang berbeda.

    def calculate(self, text1: str, text2: str):
        # menghitung cosine similarity : membandingkan arah dua vektor. nilai yang lebih tinggi menunjukkan embeddings yang lebih mirip.
        embeddings1 = self.model.encode([text1]) # mengirim teks sebagai kumpulan input
        embeddings2 = self.model.encode([text2])
        cosine_score = cosine_similarity(embeddings1, embeddings2)[0][0] # hasil berupa np.array 2D
        return cosine_score