# rag_pipeline/query_faiss.py

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

def load_faiss_index(index_path="faiss_index"):
    """
    Charge l'index FAISS sauvegardé localement.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore


def ask_question(query, index_path="faiss_index", top_k=3):
    """
    Recherche les chunks les plus pertinents pour la question.
    """
    vectorstore = load_faiss_index(index_path)
    docs = vectorstore.similarity_search(query, k=top_k)

    print(f"\n🔍 Question : {query}\n")
    print("🧠 Réponses possibles (extraits du document) :\n")
    for i, doc in enumerate(docs):
        print(f"--- Résultat {i+1} ---")
        print(doc.page_content[:500])  # on affiche les 500 premiers caractères
        print("\nMétadonnées :", doc.metadata)
        print("\n-------------------------\n")

if __name__ == "__main__":
    question = input("Pose ta question sur le document : ")
    ask_question(question)
