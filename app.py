# app.py
import gradio as gr
from query_faiss import load_faiss_index
from langchain import LLMChain, PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# 🔹 Charger l'index FAISS
vectorstore = load_faiss_index("faiss_index")

# 🔹 Configurer le LLM (ici Flan-T5 en CPU/GPU)
text_gen_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",   # ou autre modèle HF
    device=-1                       # -1 pour CPU, 0+ pour GPU
)
llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

# 🔹 Prompt pour le LLM
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Tu es un assistant intelligent. Utilise uniquement le contexte fourni pour répondre à la question.
Ne fabrique pas d'informations. Si la réponse n'est pas dans le contexte, dis "Je ne sais pas".

Contexte:
{context}

Question: {question}
Réponse:
"""
)

chain = LLMChain(llm=llm, prompt=prompt_template)

# 🔹 Fonction qui répond aux questions
def chat_with_bot(message, chat_history):
    # 1️⃣ Récupérer les chunks pertinents
    docs = vectorstore.similarity_search(message, k=3)
    
    # 2️⃣ Construire le contexte
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 3️⃣ Générer la réponse avec le LLM
    response = chain.run(context=context, question=message)
    
    # 4️⃣ Mettre à jour l'historique du chat
    chat_history.append((message, response))
    return chat_history, ""

# 🔹 Interface Gradio
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Mini RAG Chatbot\nPose une question sur ton document PDF.")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Pose ta question", placeholder="Tape ta question ici...")
    clear = gr.Button("Effacer")

    msg.submit(chat_with_bot, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: [], None, chatbot)

demo.launch()
