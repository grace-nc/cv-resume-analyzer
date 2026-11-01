import torch

from sentence_transformers import SentenceTransformer, util

class SemanticSearch:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        # Initialize Sentence Transformer model
        self.model = SentenceTransformer(model_name)

        # Initialize document data
        self.embeddings = None
        self.documents = None
        self.ids = None
        self.job_titles = None


    def fit(self, documents: list[str], ids: list, job_titles: list[str]
    ):

        # Validate input data
        if not documents:
            raise ValueError("Dataset cannot be empty.")

        if not (
            len(documents) == len(ids) == len(job_titles)
        ):
            raise ValueError(
                "Documents, IDs, and job titles must have the same length."
            )

        # Store document information
        self.documents = documents
        self.ids = ids
        self.job_titles = job_titles

        # Convert resume text into embeddings
        self.embeddings = self.model.encode(
            documents,
            convert_to_tensor=True,
            show_progress_bar=False
        )

        return self


    def search(self, query: str, top_k: int = 5):

        # Validate whether the model has been fitted
        if self.embeddings is None:
            raise ValueError(
                "Please fit the documents before performing semantic search."
            )

        # Validate search query
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")

        # Validate Top K
        if top_k < 1:
            raise ValueError("top_k must be greater than 0.")

        # Convert search query into embedding
        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True
        )

        # Calculate Cosine Similarity
        cosine_scores = util.cos_sim(
            query_embedding,
            self.embeddings
        )[0]

        # Retrieve Top K results, if top_k exceeds the number of documents, adjust it
        top_k = min(top_k, len(self.documents))

        results = torch.topk(
            cosine_scores,
            k=top_k
        )

        # Store search results
        search_results = []

        for score, idx in zip(
            results.values,
            results.indices
        ):

            idx = idx.item()

            search_results.append({
                "id": self.ids[idx],
                "job_title": self.job_titles[idx],
                "resume": self.documents[idx],
                "similarity_score": score.item()
                })

        return search_results