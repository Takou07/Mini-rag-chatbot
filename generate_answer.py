# rag_pipeline/generate_answer.py

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import PromptTemplate, LLMChain
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

def load_faiss_index(index_path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore


def build_llm():
    """
    Charge un modèle open-source depuis Hugging Face.
    """
    model_id = "google/flan-t5-base"  # tu peux essayer mistralai/Mistral-7B si tu veux un modèle plus puissant
    pipe = pipeline("text2text-generation", model=model_id, max_new_tokens=512)
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


def ask_with_generation(query, index_path="faiss_index", top_k=3):
    # 1️⃣ Charger l'index FAISS
    vectorstore = load_faiss_index(index_path)

    # 2️⃣ Récupérer les passages pertinents
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

    # 4️⃣ Préparer le modèle et la chaîne
    llm = build_llm()
    chain = LLMChain(llm=llm, prompt=prompt)

    # 5️⃣ Générer la réponse
    response = chain.run(context=context, question=query)
    print("\n🧠 Réponse générée :\n")
    print(response)


if __name__ == "__main__":
    question = input("Pose ta question sur le document : ")
    ask_with_generation(question)
