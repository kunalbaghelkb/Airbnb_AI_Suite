import os
import sys
import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from src.exception import CustomException
from src.logger import logging

class GenAIEngine:
    def __init__(self):
        # We will use 'all-MiniLM-L6-v2' model (Small and Fast)
        self.model_name = 'all-MiniLM-L6-v2'
        self.artifacts_path = 'artifacts'
        self.index_file = os.path.join(self.artifacts_path, 'faiss_index.bin')
        self.metadata_file = os.path.join(self.artifacts_path, 'metadata.pkl')
        
        # In-memory artifacts (Loaded on first use)
        self.index = None
        self.metadata = None
        self.encoder = None
        
    def _load_artifacts(self):
        '''Lazy loader for embedding model and search index'''
        try:
            if self.index is None:
                logging.info("Loading FAISS Index into memory...")
                self.index = faiss.read_index(self.index_file)
            
            if self.metadata is None:
                logging.info("Loading Metadata into memory...")
                with open(self.metadata_file, "rb") as f:
                    self.metadata = pickle.load(f)
            
            if self.encoder is None:
                logging.info("Loading SentenceTransformer model into memory (First search only)...")
                self.encoder = SentenceTransformer(self.model_name)
                
        except Exception as e:
            raise CustomException(f"Failed to load search artifacts: {str(e)}", sys)

    def create_vector_db(self):
        # ... (Method header stays the same, ensure it clears cache after build)
        try:
            logging.info("Starting Vector Database Creation...")
            
            # 1. Load Raw Data
            df = pd.read_csv('data/raw/listings.csv')
            df = df.dropna(subset=['name'])
            df_subset = df.copy()
            documents = df_subset['name'].tolist()
            
            # Store Metadata
            metadata = df_subset[['id', 'name', 'neighbourhood', 'price']].to_dict(orient='records')
            
            # 2. Generate Embeddings
            logging.info("Loading Embedding Model for training...")
            encoder = SentenceTransformer(self.model_name)
            embeddings = encoder.encode(documents)
            
            # 3. Build FAISS Index (Search Engine)
            dimension = embeddings.shape[1] # 384 dimensions
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
            
            # 4. Save Artifacts
            os.makedirs(self.artifacts_path, exist_ok=True)
            faiss.write_index(index, self.index_file)
            with open(self.metadata_file, "wb") as f:
                pickle.dump(metadata, f)
            
            # Clear in-memory cache to force reload on next search
            self.index = None
            self.metadata = None
            self.encoder = None
            
            logging.info("GenAI Artifacts Saved Successfully.")
            return "Vector DB Created Successfully!"
        
        except Exception as e:
            raise CustomException(e, sys)
            
    def search_listings(self, query, top_k=3):
        '''
        It takes user query and returns Best Matches (Instant from Memory)
        '''
        try:
            # Step 1: Ensure models are in memory
            self._load_artifacts()
            
            # Step 2: Create Vector from Query
            query_vector = self.encoder.encode([query])
            
            # Step 3: Do Search
            distances, indices = self.index.search(query_vector, top_k)
            
            results = []
            for i in range(top_k):
                idx = indices[0][i]
                if idx < len(self.metadata):
                    result = self.metadata[idx].copy()
                    result['score'] = float(distances[0][i])
                    results.append(result)
                
            return results
        except Exception as e:
            raise CustomException(e, sys)