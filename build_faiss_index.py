from pathlib import Path
import faiss
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from load_documents import load_and_split_document  # étape 1

# 🔹 Paramètres
PDF_PATH = "chapitre1.pdf"  # ton PDF
INDEX_PATH = "faiss_index"  # dossier où sera sauvegardé l'index
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def build_faiss_index():
    try:
        # 1️⃣ Charger et découper le document
        chunks = load_and_split_document(PDF_PATH, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        if not chunks:
            print("⚠️ Aucun chunk trouvé. Index non créé.")
            return None

        # 2️⃣ Générer les embeddings localement (gratuit)
        print("🧠 Génération des embeddings avec SentenceTransformers...")
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name=model_name)

        # 3️⃣ Créer l'index FAISS
        print("⚙️ Création de l'index FAISS...")
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # 4️⃣ Sauvegarder l'index localement
        index_dir = Path(INDEX_PATH)
        index_dir.mkdir(exist_ok=True)
        vectorstore.save_local(str(index_dir))

        print(f"✅ Index FAISS créé et sauvegardé dans '{INDEX_PATH}'")
        return vectorstore

    except Exception as e:
        print(f"❌ Erreur : {e}")
        return None

if __name__ == "__main__":
    build_faiss_index()
