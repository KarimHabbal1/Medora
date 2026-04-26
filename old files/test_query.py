import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

print("Step 1: imports OK")
load_dotenv()
print("Step 2: dotenv OK")

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
print("Step 3: OpenAI client OK")

CHROMA_DIR = r"c:/Users/AUB/open source model/Medora/chroma_symptoms"
os.makedirs(CHROMA_DIR, exist_ok=True)
print("Step 4a: folder created OK")

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
print("Step 4b: Chroma client OK")

collection = chroma_client.get_or_create_collection("symptom_chunks")
print("Step 5: Collection OK, count =", collection.count())