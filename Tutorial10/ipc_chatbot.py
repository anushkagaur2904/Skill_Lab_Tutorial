from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = Path(__file__).resolve().parent.parent / "data" / "ipc.pdf"


def build_vector_store():
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"IPC PDF not found at {PDF_PATH}. Download the official IPC document "
            "and save it as data/ipc.pdf."
        )

    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)


def answer_question(llm, vector_store, question, history):
    retrieved_docs = vector_store.similarity_search(question, k=4)
    context = "\n\n".join(
        f"[Page {doc.metadata.get('page', '?') + 1}]\n{doc.page_content}"
        for doc in retrieved_docs
    )

    previous = "\n".join(history[-4:]) or "No previous conversation."
    prompt = f"""
You are an educational chatbot for the Indian Penal Code (IPC).
Answer only from the retrieved IPC context below. If the answer is not present,
say that the information was not found in the retrieved document.
Do not invent section numbers or legal provisions. This is an academic demo,
not legal advice.

Previous conversation:
{previous}

Retrieved IPC context:
{context}

User question: {question}

Give a concise, clear answer. Mention relevant IPC section numbers when they
are present in the retrieved context.
"""
    response = llm.invoke(prompt)
    return response.content, retrieved_docs


def main():
    print("Loading IPC document and building the semantic vector store...")
    vector_store = build_vector_store()
    llm = ChatOllama(model="llama3.2", temperature=0)

    print("\nIPC Chatbot ready. Type 'exit' to stop.\n")
    history = []

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not question:
            continue

        try:
            answer, docs = answer_question(llm, vector_store, question, history)
            print(f"\nBot: {answer}\n")
            pages = sorted({doc.metadata.get("page", 0) + 1 for doc in docs})
            print(f"Retrieved pages: {pages}\n")
            history.extend([f"User: {question}", f"Bot: {answer}"])
        except Exception as exc:
            print(f"Error: {exc}\n")


if __name__ == "__main__":
    main()
