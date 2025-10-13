# rag_pipeline/load_documents.py

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_document(file_path, chunk_size=1000, chunk_overlap=200):
    """
    Charge un PDF et le découpe en petits chunks exploitables pour RAG.
    
    Paramètres :
        file_path (str)      : chemin vers le PDF
        chunk_size (int)     : taille maximale d’un chunk en caractères
        chunk_overlap (int)  : chevauchement entre chunks
    
    Retour :
        List[Document] : liste d'objets Document contenant le texte découpé et les métadonnées
    """
    # Vérifier que le fichier existe
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Charger le PDF
    try:
        loader = PyPDFLoader(str(path))
        documents = loader.load()
        if not documents:
            print(f"⚠️ Le document {file_path} est vide.")
            return []
    except Exception as e:
        print(f"Erreur lors du chargement du PDF : {e}")
        return []

    # Ajouter métadonnée source
    for doc in documents:
        doc.metadata["source"] = path.name

    # Découper en chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)

    # Ajouter un ID unique à chaque chunk
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    print(f"✅ Document '{path.name}' chargé et découpé en {len(chunks)} chunks.")
    return chunks

# Exemple de test
if __name__ == "__main__":
    # Chemin vers ton PDF
    pdf_path = "chapitre1.pdf"
    
    chunks = load_and_split_document(pdf_path, chunk_size=1000, chunk_overlap=200)

    # Afficher un aperçu du premier chunk
    if chunks:
        print("\n--- Aperçu du premier chunk ---")
        print(chunks[0].page_content[:300])  # 300 premiers caractères
        print("\n--- Métadonnées ---")
        print(chunks[0].metadata)
