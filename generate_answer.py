# rag_pipeline/generate_answer.py

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import pipeline


def load_faiss_index(index_path="faiss_index"):
    """
    Charge l'index FAISS sauvegardé localement.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore


def build_llm():
    """
    Initialise un modèle open-source depuis Hugging Face.
    """
    model_id = "google/flan-t5-base"  # tu peux tester "mistralai/Mistral-7B-Instruct" si tu veux plus de puissance
    pipe = pipeline("text2text-generation", model=model_id, max_new_tokens=512)
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


def ask_with_generation(query, index_path="faiss_index", top_k=3):
    """
    Pipeline complet : recherche + génération.
    """
    # 1️⃣ Charger l'index FAISS
    vectorstore = load_faiss_index(index_path)

    # 2️⃣ Trouver les passages les plus pertinents
    docs = vectorstore.similarity_search(query, k=top_k)
    context = "\n\n".join([doc.page_content for doc in docs])

    # 3️⃣ Construire le prompt
    template = """
    Tu es un assistant expert.
    En te basant UNIQUEMENT sur le contexte suivant, réponds à la question de manière claire et concise.

    CONTEXTE :
    {context}

    QUESTION :
    {question}

    RÉPONSE :
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # 4️⃣ Créer le modèle et combiner le prompt avec le modèle
    llm = build_llm()
    chain = prompt | llm  # ✅ Remplace LLMChain

    # 5️⃣ Générer la réponse
    response = chain.invoke({"context": context, "question": query})
    print("\n🧠 Réponse générée :\n")
    print(response)


if __name__ == "__main__":
    question = input("Pose ta question sur le document : ")
    ask_with_generation(question)
