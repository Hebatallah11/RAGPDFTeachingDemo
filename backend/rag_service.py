import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from backend.ocr_loader import load_pdf_with_ocr_fallback
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from backend.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    LLM_MODEL,
    TEMPERATURE,
    TOP_K,
)

load_dotenv()

vector_store = None


def load_pdf(file_path: str):
    loader = load_pdf_with_ocr_fallback(file_path)
    return loader.load()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def build_vector_store(chunks):
    global vector_store
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)
    return vector_store


def ingest_pdf(file_path: str) -> Dict[str, Any]:
    documents = load_pdf(file_path)
    chunks = split_documents(documents)
    build_vector_store(chunks)
    return {
        "pages": len(documents),
        "chunks": len(chunks),
        "message": "PDF indexed successfully",
    }


def get_retriever():
    global vector_store
    if vector_store is None:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": TOP_K,
            "score_threshold": 0.1,
        },
    )


def build_rag_prompt(question: str, retrieved_docs):
    context = "\n\n".join(
        [
            f"Source: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')}\n{doc.page_content}"
            for doc in retrieved_docs
        ]
    )
    return f"""You are a document-grounded assistant.
Answer the question using only the context below.
If the answer is not available in the context, say:
"I don't know based on the provided PDF."

Context:
{context}

Question:
{question}

Answer:"""


def ask_without_rag(question: str) -> str:
    llm = ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE)
    response = llm.invoke(question)
    return response.content


def ask_with_rag(question: str) -> Dict[str, Any]:
    global vector_store
    if vector_store is None:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    docs_and_scores = vector_store.similarity_search_with_score(question, k=TOP_K)
    retrieved_docs = [doc for doc, score in docs_and_scores]
    prompt = build_rag_prompt(question, retrieved_docs)

    llm = ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE)
    response = llm.invoke(prompt)

    chunks = []
    for doc, score in docs_and_scores:
        source = os.path.basename(doc.metadata.get("source", ""))
        page = doc.metadata.get("page", 0) + 1
        chunks.append(
            {
                "content": doc.page_content,
                "source": source,
                "page": page,
                "score": float(score),
            }
        )

    return {
        "answer": response.content,
        "chunks": chunks,
    }
