import gradio as gr
from RAG_Helper import RAGHelper

rag = RAGHelper(pdf_folder="pdfFiles")

def init_rag():
    import asyncio
    asyncio.run(rag.load_and_prepare(file_extensions=[".pdf", ".txt", ".docx"]))
    rag.setup_retrieval_chain()
    return "✅ 知識庫載入完成"

def ask_question(query):
    answer, _ = rag.ask(query)
    return answer

with gr.Blocks() as demo:
    with gr.Row():
        load_btn = gr.Button("載入知識庫")
        status = gr.Textbox(label="狀態")

    with gr.Row():
        query = gr.Textbox(label="輸入問題")
        submit_btn = gr.Button("送出")
        answer = gr.Textbox(label="回答")

    load_btn.click(fn=init_rag, outputs=status)
    submit_btn.click(fn=ask_question, inputs=query, outputs=answer)

import os

if __name__ == "__main__":
    import gradio as gr
    iface = gr.Interface(fn=your_function, inputs=..., outputs=...)
    iface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 8080)))

