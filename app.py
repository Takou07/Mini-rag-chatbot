# ✅ Version stable LangChain 2025

import gradio as gr
from query_faiss import load_faiss_index
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence  # sert à remplacer LLMChain
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

# Charger l'index FAISS
vectorstore = load_faiss_index("faiss_index")

# Configurer le LLM
text_gen_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)
llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

# Prompt
prompt = PromptTemplate(
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

# Remplacer LLMChain par un pipeline exécutable
chain = RunnableSequence(first=prompt, last=llm)

def chat_with_bot(message, chat_history):
    docs = vectorstore.similarity_search(message, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    response = chain.invoke({"context": context, "question": message})
    chat_history.append((message, response))
    return chat_history, ""

with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Mini RAG Chatbot\nPose une question sur ton document PDF.")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Pose ta question", placeholder="Tape ta question ici...")
    clear = gr.Button("Effacer")

    msg.submit(chat_with_bot, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: [], None, chatbot)

demo.launch()
