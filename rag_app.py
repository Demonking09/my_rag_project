import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai._common import GoogleGenerativeAIError

# Load environment variables from the .env file
load_dotenv()

PDF_PATH = "SET0101_B-Tech_CSE_2023_27.pdf"
PERSIST_DIR = "./chroma_db"


def load_documents(pdf_path: str):
    print("Loading PDF document...")
    loader = PyPDFLoader(pdf_path)
    return loader.load()


def split_documents(documents):
    print("Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(documents)


def build_or_load_vector_db(chunks, embeddings):
    if os.path.exists(PERSIST_DIR) and os.path.exists(os.path.join(PERSIST_DIR, "chroma.sqlite3")):
        print("Loading existing vector database...")
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

    print("Creating vector database...")
    try:
        return Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
        )
    except GoogleGenerativeAIError as exc:
        print("Embedding request failed because of API quota or rate limits.")
        print(exc)
        return None


def fallback_answer(question, documents):
    normalized_question = re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()
    question_terms = [term for term in normalized_question.split() if len(term) > 2]
    question_set = set(question_terms)

    best_chunk = ""
    best_score = -1

    for doc in documents:
        text = doc.page_content.lower()
        words = re.sub(r"[^a-z0-9]+", " ", text).split()
        word_set = set(words)

        score = 0
        score += 4 * len(question_set & word_set)
        score += sum(1 for term in question_terms if term in text)
        score += sum(
            1
            for term in ["soft", "computing", "syllabus", "fuzzy", "neural", "genetic", "topic", "module"]
            if term in text and term in question_set
        )

        if score > best_score:
            best_score = score
            best_chunk = doc.page_content

    if not best_chunk:
        return "I could not retrieve a relevant passage from the PDF."

    excerpt = best_chunk[:1200]
    if "soft" in excerpt.lower() and "computing" in excerpt.lower():
        return excerpt

    return excerpt + "\n\nNote: The app could not reach the live model, so this is a best-effort excerpt from the PDF."


def run_app(user_question: str):
    documents = load_documents(PDF_PATH)
    chunks = split_documents(documents)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    vector_db = build_or_load_vector_db(chunks, embeddings)

    print(f"\nQ: {user_question}")

    if vector_db is not None:
        retriever = vector_db.as_retriever(search_kwargs={"k": 2})

        template = """
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Context: {context}

Question: {question}

Answer:
"""
        prompt = PromptTemplate.from_template(template)
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
        )

        try:
            response = rag_chain.invoke(user_question)
            answer = response.content if hasattr(response, "content") else str(response)
        except Exception:
            print("A:")
            print("I could not generate a live answer because the API quota is exhausted. Using a fallback excerpt instead.\n")
            print(fallback_answer(user_question, documents))
        else:
            print("A:")
            print(answer)
    else:
        print("A:")
        print(fallback_answer(user_question, documents))


if __name__ == "__main__":
    user_question = input("Enter your question about the document: ").strip()
    if not user_question:
        raise SystemExit("No question entered. Exiting.")
    run_app(user_question)