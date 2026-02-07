from src.components.genai_engine import GenAIEngine

# 1. Initialize
genai = GenAIEngine()

# 2. Create Database (First Time Setup)
# print("Creating Vector Database... (Wait for download)")
# genai.create_vector_db()

# 3. User Query Search
query = "Apartment near park with view"
print(f"\n Searching for: '{query}'...\n")

results = genai.search_listings(query)

for res in results:
    print(f"🏠 Name: {res['name']}")
    print(f"📍 Area: {res['neighbourhood']}")
    print(f"💰 Price: ${res['price']}")
    print("-" * 30)