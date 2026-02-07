# 🗽 Airbnb AI Suite: End-to-End ML & GenAI Project

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20Sklearn-green)
![GenAI](https://img.shields.io/badge/GenAI-RAG%20%7C%20FAISS-orange)
![Docker](https://img.shields.io/badge/Deployment-Docker%20%7C%20AWS-blueviolet)

## 📌 Project Overview
The **Airbnb AI Suite** is a production-grade machine learning application designed to solve two major problems in the Real Estate/Tourism domain:

1.  **Smart Price Prediction Engine:** Helps hosts/users estimate the correct rental price based on location, amenities, and market trends using advanced regression algorithms (XGBoost).
2.  **GenAI Recommendation System:** A **RAG (Retrieval Augmented Generation)** based search engine that allows users to search for properties using natural language (e.g., *"I need a peaceful apartment near Central Park for remote work"*).

---

## 🏗️ Architecture

The project follows a modular **MNC-Standard Pipeline** structure:

### 1. Data Pipeline (ETL)
* **Ingestion:** Reads raw data from CSV/Database.
* **Transformation:** Handles Missing Values, OneHotEncoding, Scaling, and Feature Engineering (Distance calculation from City Center).
* **Storage:** Saves processed artifacts (`preprocessor.pkl`).

### 2. Model Factory
* **Algorithms:** Trains multiple models (Random Forest, XGBoost, CatBoost, Gradient Boosting).
* **Hyperparameter Tuning:** Uses `GridSearchCV` to find the best parameters.
* **Selection:** Automatically selects the model with the best R2 Score (> 0.60).

### 3. GenAI Engine (RAG)
* **Vector DB:** Converts listing descriptions into Vector Embeddings using `Sentence-Transformers`.
* **Search:** Uses **FAISS (Facebook AI Similarity Search)** for semantic search capabilities.

---

## 🛠️ Tech Stack

* **Programming:** Python 3.11
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost, CatBoost
* **GenAI/NLP:** Sentence-Transformers (`all-MiniLM-L6-v2`), FAISS
* **Backend (Upcoming):** FastAPI / Flask
* **Deployment (Upcoming):** Docker, AWS EC2

---

## 📂 Project Structure

    Airbnb_AI_Suite/
    ├── artifacts/              # Stores Models & Preprocessors (Ignored in Git)
    ├── data/                   # Raw & Processed Data (Ignored in Git)
    ├── logs/                   # Execution Logs
    ├── notebooks/              # Jupyter Notebooks for EDA & Experiments
    ├── src/                    # Source Code
    │   ├── components/         # Core Modules
    │   │   ├── data_ingestion.py
    │   │   ├── data_transformation.py
    │   │   ├── model_trainer.py
    │   │   └── genai_engine.py # RAG Logic
    │   ├── pipeline/           # Pipelines
    │   │   ├── train_pipeline.py   # Triggers Ingestion -> Training
    │   │   └── predict_pipeline.py # Used for Inference (Web App)
    │   ├── utils.py            # Helper Functions
    │   ├── logger.py           # Logging Config
    │   └── exception.py        # Custom Exception Handling
    ├── test_prediction.py      # Script to test Price Prediction logic
    ├── test_genai.py           # Script to test RAG/Vector Search logic
    ├── app.py                  # API Entry Point (Upcoming)
    ├── requirements.txt        # Project Dependencies
    ├── Dockerfile              # Containerization
    └── README.md               # Documentation

---

## 🚀 How to Run Locally

1. Clone the Repository
    ```bash
    git clone [https://github.com/kunalbaghelkb/Airbnb_AI_Suite.git](https://github.com/kunalbaghelkb/Airbnb_AI_Suite.git) && cd Airbnb_AI_Suite

2. Create Virtual Environment
    ```bash
    python3.11 -m venv venv

3. Install Dependencies
    ```bash
    pip install -r requirements.txt

4. Run the Training Pipeline
This will ingest data, clean it, train models, and save the best `model.pkl`.
    ```bash
    python src/components/data_ingestion.py

5. Test Prediction
    ```bash
    python test_prediction.py

6. Test GenAI Search
    ```bash
    python test_genai.py

---

## 🔮 Future Roadmap
- [ ] **Frontend:** Develop a UI using HTML/CSS templates.
- [ ] **API:** Expose endpoints using FastAPI.
- [ ] **Dockerization:** Containerize the application.
- [ ] **CI/CD:** Setup GitHub Actions.
- [ ] **Cloud Deployment:** Deploy on AWS EC2 (Free Tier).

---

## 👨‍💻 Author
**Kunal Baghel**

*Aspiring Data Scientist & AI Engineer*

[LinkedIn](https://linkedin.com/in/kunalbaghelz) | [GitHub](http://github.com/kunalbaghelkb)
