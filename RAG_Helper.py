import os
import glob
import re
import numpy as np
import openai
from pathlib import Path
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
)

class RAGHelper:
    def __init__(self, pdf_folder, chunk_size=300, chunk_overlap=50):
        self.pdf_folder = pdf_folder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore = None
        self.retrieval_chain = None

    def get_loader(self, path: str):
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            return PyPDFLoader(path)
        elif ext == ".txt":
            return TextLoader(path, encoding="utf-8")
        elif ext == ".docx":
            return UnstructuredWordDocumentLoader(path)
        elif ext == ".md":
            return UnstructuredMarkdownLoader(path)
        elif ext == ".csv":
            return CSVLoader(path)
        else:
            raise ValueError(f"\u4e0d\u652f\u63f4\u7684\u6a94\u6848\u985e\u578b: {ext}")

    async def load_any_file_async(self, path: str):
        loader = self.get_loader(path)
        if hasattr(loader, "alazy_load"):
            pages = []
            async for page in loader.alazy_load():
                pages.append(page)
            return pages
        else:
            return loader.load()

    def _semantic_split_documents(self, documents):
        def split_sentences(text):
            pattern = r"(?<=[\u3002\uff01\uff1f!?.;；\n])"
            return [s.strip() for s in re.split(pattern, text) if s.strip()]

        def get_embeddings(sentences, model="text-embedding-3-small"):
            response = openai.Embedding.create(input=sentences, model=model)
            return [r["embedding"] for r in response["data"]]

        def cosine_sim(a, b):
            a, b = np.array(a), np.array(b)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        def semantic_chunk(sentences, embeddings, threshold=0.88, max_sentences=8):
            chunks = []
            current = [sentences[0]]
            for i in range(1, len(sentences)):
                sim = cosine_sim(embeddings[i - 1], embeddings[i])
                if sim < threshold or len(current) >= max_sentences:
                    chunks.append("".join(current))
                    current = [sentences[i]]
                else:
                    current.append(sentences[i])
            if current:
                chunks.append("".join(current))
            return chunks

        all_chunks = []
        for doc in documents:
            text = doc.page_content
            meta = doc.metadata
            try:
                sentences = split_sentences(text)
                if len(sentences) < 2:
                    continue
                embeddings = get_embeddings(sentences)
                sem_chunks = semantic_chunk(sentences, embeddings)
                for i, chunk_text in enumerate(sem_chunks):
                    all_chunks.append(Document(
                        page_content=chunk_text,
                        metadata={**meta, "chunk_id": i}
                    ))
            except Exception as e:
                print(f"\u8a9e\u610f\u5207\u5272\u6642\u767c\u751f\u932f\u8aa4: {e}")
                continue
        print(f"\u2705 \u8a9e\u610f\u5207\u5272\u5b8c\u6210\uff0c\u5171\u5207\u51fa {len(all_chunks)} \u6bb5")
        return all_chunks

    def _build_vectorstore(self, documents):
        print(f"\u5efa\u7acb\u5411\u91cf\u8cc7\u6599\u5eab... \u5171 {len(documents)} \u500b\u6bb5\u843d")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = FAISS.from_documents(documents, embeddings)

    async def load_and_prepare(self, file_extensions=None):
        print("\u958b\u59cb\u8f09\u5165\u6a94\u6848...")
        if os.path.exists("my_faiss_index"):
            print("\u5df2\u50b3\u7d71\u5411\u91cf\u8cc7\u6599\u5eab\uff0c\u76f4\u63a5\u8f09\u5165...")
            self.vectorstore = FAISS.load_local(
                "my_faiss_index",
                OpenAIEmbeddings(model="text-embedding-3-small"),
                allow_dangerous_deserialization=True
            )
        else:
            if file_extensions is None:
                file_extensions = ['.pdf']

            all_chunks = []
            for ext in file_extensions:
                pattern = f"*{ext}"
                file_paths = glob.glob(os.path.join(self.pdf_folder, pattern))
                for path in file_paths:
                    try:
                        print(f"\u8b80\u53d6\u4e2d: {os.path.basename(path)}")
                        pages = await self.load_any_file_async(path)
                        chunks = self._semantic_split_documents(pages)
                        all_chunks.extend(chunks)
                        print(f" {os.path.basename(path)} \u5206\u5272\u5b8c\u6210\uff0c\u5171 {len(chunks)} \u6bb5")
                    except Exception as e:
                        print(f"\u8f09\u5165 {os.path.basename(path)} \u6642\u767c\u751f\u932f\u8aa4: {e}")
            if len(all_chunks) == 0:
                raise ValueError("\u6c92\u6709\u6210\u529f\u8f09\u5165\u4efb\u4f55\u6587\u4ef6")
            self._build_vectorstore(all_chunks)
            self.vectorstore.save_local("my_faiss_index")

    def setup_retrieval_chain(self):
        if not self.vectorstore:
            raise ValueError("\u8acb\u5148\u57f7\u884c load_and_prepare()")

        llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        system_prompt = (
            "\u4f60\u662f\u4e00\u500b\u554f\u7b54\u52a9\u624b\u3002\u57fa\u65bc\u4ee5\u4e0b\u63d0\u4f9b\u7684\u5167\u5bb9\u4f86\u56de\u7b54\u554f\u984c\u3002"
            "\u5982\u679c\u5167\u5bb9\u4e2d\u6c92\u6709\u76f8\u95dc\u8cc7\u8a0a\uff0c\u8acb\u8aaa\u300c\u6839\u64da\u63d0\u4f9b\u7684\u8cc7\u6599\u7121\u6cd5\u56de\u7b54\u9019\u500b\u554f\u984c\u300d\u3002"
            "\u8acb\u7528\u7e41\u9ad4\u4e2d\u6587\u56de\u7b54\u3002\n\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        self.retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)

    def ask(self, query):
        if not self.retrieval_chain:
            raise ValueError("\u8acb\u5148\u57f7\u884c setup_qa_chain()")
        try:
            result = self.retrieval_chain.invoke({"input": query})
            return result["answer"], result["context"]
        except Exception as e:
            if "max_tokens_per_request" in str(e):
                print("\u5167\u5bb9\u904e\u9577\uff0c\u5617\u8a66\u4f7f\u7528\u8f03\u77ed\u7684\u4e0a\u4e0b\u6587...")
                self.setup_retrieval_chain_with_shorter_context()
                result = self.retrieval_chain.invoke({"input": query})
                return result["answer"], result["context"]
            else:
                raise e

    def setup_retrieval_chain_with_shorter_context(self):
        if not self.vectorstore:
            raise ValueError("\u8acb\u5148\u57f7\u884c load_and_prepare()")

        llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        system_prompt = (
            "\u4f60\u662f\u4e00\u500b\u554f\u7b54\u52a9\u624b\u3002\u57fa\u65bc\u4ee5\u4e0b\u63d0\u4f9b\u7684\u5167\u5bb9\u4f86\u56de\u7b54\u554f\u984c\u3002"
            "\u5982\u679c\u5167\u5bb9\u4e2d\u6c92\u6709\u76f8\u95dc\u8cc7\u8a0a\uff0c\u8acb\u8aaa\u300c\u6839\u64da\u63d0\u4f9b\u7684\u8cc7\u6599\u7121\u6cd5\u56de\u7b54\u9019\u500b\u554f\u984c\u300d\u3002"
            "\u8acb\u7528\u7e41\u9ad4\u4e2d\u6587\u7c21\u6f54\u56de\u7b54\u3002\n\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        self.retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)
