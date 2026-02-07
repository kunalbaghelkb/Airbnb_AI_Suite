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
        # This converts text to 384 numbers list
        self.model_name = 'all-MiniLM-L6-v2'
        self.artifacts_path = 'artifacts'
        self.index_file = os.path.join(self.artifacts_path, 'faiss_index.bin')
        self.metadata_file = os.path.join(self.artifacts_path, 'metadata.pkl')
        
    def create_vector_db(self):
        '''
        This function reads listing names and creates Vector Database
        Only run this ONCE (Training Time)
        '''
        try:
            logging.info("Starting Vector Database Creation...")
            
            # 1. Load Raw Data
            df = pd.read_csv('data/raw/listings.csv')
            
            # Only take those rows where Name is available
            df = df.dropna(subset=['name'])
            
            df_subset = df.copy()
            
            documents = df_subset['name'].tolist()
            
            # Store Metadata (Price, Neighbourhood) so we can show details after search
            metadata = df_subset[['id', 'name', 'neighbourhood', 'price']].to_dict(orient='records')
            
            # 2. Generate Embeddings
            logging.info("Loading Embedding Model (This downloads ~80MB data)...")
            encoder = SentenceTransformer(self.model_name)
            embeddings = encoder.encode(documents)
            
            # 3. Build FAISS Index (Search Engine)
            dimension = embeddings.shape[1] # 384 dimensions
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
            
            logging.info(f"Vector Index Created with {index.ntotal} documents.")
            
            # 4. Save Artifacts
            os.makedirs(self.artifacts_path, exist_ok=True)
            
            # Save FAISS Index
            faiss.write_index(index, self.index_file)
            
            # Save Metadata & Model Info
            with open(self.metadata_file, "wb") as f:
                pickle.dump(metadata, f)
            
            logging.info("GenAI Artifacts Saved Successfully.")
            
            return "Vector DB Created Successfully!"
        
        except Exception as e:
            raise CustomException(e, sys)
            
    def search_listings(self, query, top_k=3):
        '''
        It takes user query and returns Best Matches
        '''
        try:
            # Load Index and Metadata
            index = faiss.read_index(self.index_file)
            with open(self.metadata_file, "rb") as f:
                metadata = pickle.load(f)
            
            # Load Encoder
            encoder = SentenceTransformer(self.model_name)
            
            # Create Vector from Query
            query_vector = encoder.encode([query])
            
            # Do Search (Distance Calculation)
            distances, indices = index.search(query_vector, top_k)
            
            results = []
            
            for i in range(top_k):
                idx = indices[0][i]
                result = metadata[idx]
                result['score'] = float(distances[0][i]) # Similarity Score
                results.append(result)
                
            return results
        except Exception as e:
            raise CustomException(e, sys)