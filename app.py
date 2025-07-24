import gradio as gr
import asyncio
from RAG_Helper import RAGHelper

rag = RAGHelper(pdf_folder="pdfFiles")

async def init_rag():
    await rag.load_and_prepare(file_extensions=[".pdf", ".txt", ".docx"])
    rag.setup_retrieval_chain()
    rag.setup_qa_chain() 
# 啟動時載入知識庫

def ask_question(message, history):
    answer, _ = rag.ask(message)
    return answer

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="RAG 問答助手")
    msg = gr.Textbox(label="輸入你的問題", placeholder="請輸入問題...", show_label=False)
    clear = gr.Button("清除對話")

    def respond(user_message, chat_history):
        answer = ask_question(user_message, chat_history)
        chat_history.append((user_message, answer))
        return "", chat_history

    msg.submit(fn=respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    clear.click(lambda: [], None, chatbot, queue=False)

demo.launch(server_name="0.0.0.0", server_port=8080)
