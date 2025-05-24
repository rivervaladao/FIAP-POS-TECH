import os
import threading
import whisper
import time
from tqdm import tqdm
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex, Document, StorageContext, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from chromadb import PersistentClient
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Load environment variables from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("❌ ERROR: OPENAI_API_KEY is missing. Please check your .env file.")

os.environ["OPENAI_API_KEY"] = openai_api_key

# ChromaDB configuration
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "documents"

# Initialize ChromaDB client globally
chroma_client = PersistentClient(path=CHROMA_DB_PATH)
chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

# Configure LlamaIndex settings (only embedding model needed here)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

def transcribe_video_async(video_path, result_holder):
    model = whisper.load_model("base")
    result_holder["text"] = model.transcribe(video_path)["text"]

def transcribe_video(video_path: str) -> str:
    print(f"🔹 Transcribing video: {video_path}")
    result_holder = {}
    transcription_thread = threading.Thread(target=transcribe_video_async, args=(video_path, result_holder))
    transcription_thread.start()

    with tqdm(total=100, desc="Transcribing", bar_format="{l_bar}{bar} [{elapsed}<{remaining}]") as pbar:
        while transcription_thread.is_alive():
            time.sleep(0.5)
            pbar.update(5)

    transcription_thread.join()
    transcript_text = result_holder.get("text", "")

    if not transcript_text.strip():
        raise ValueError("❌ ERROR: No text extracted from the video. Check the video format and content.")

    print("\n✅ Transcription Complete!")
    return transcript_text

def dump_chroma_content_to_file(filename="chroma_content.txt"):
    print(f"📤 Dumping ChromaDB content to {filename}...")
    stored_docs = chroma_collection.count()

    if stored_docs == 0:
        print("❌ No documents found in ChromaDB to dump.")
        return

    results = chroma_collection.get(include=["documents"])
    documents = results["documents"] or []

    try:
        with open(filename, "w", encoding="utf-8") as f:
            for doc in documents:
                f.write(f"{doc}\n\n")
        print(f"✅ Successfully dumped {stored_docs} document contents to {filename}")
    except Exception as e:
        print(f"❌ Error dumping to file: {e}")

def store_documents_in_chroma(documents):
    print("🔹 Storing documents in ChromaDB...")
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"✅ Chunking complete! Total chunks: {len(nodes)}")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)

    print("✅ Documents Stored in ChromaDB!")
    return index

def load_index():
    print("🔹 Loading stored documents from ChromaDB...")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    try:
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
        print("✅ ChromaDB Index Loaded Successfully!")
        return index
    except Exception as e:
        print(f"❌ Error loading index: {e}")
        return None

def create_rag_chain():
    index = load_index()
    if index is None:
        return None

    # Set up LlamaIndex retriever
    retriever = index.as_retriever(similarity_top_k=5, verbose=True)

    # Wrap LlamaIndex retriever in a LangChain Runnable
    retriever_runnable = RunnableLambda(lambda query: [doc.node.text for doc in retriever.retrieve(query)])

    # Define prompt
    prompt = ChatPromptTemplate.from_template(
        """You are an AI assistant helping to answer questions about stored documents.
Use the following document context to answer the question as accurately as possible.
You can use the content of the stored documents.
If there is no clear answer, suggest some example questions that you can answer based on the data in the documents.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    )

    # Define LLM directly for LangChain compatibility
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Define RAG chain
    rag_chain = (
        {
            "context": retriever_runnable | (lambda texts: "\n\n".join(texts)),  # Retrieve and format context
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def query_document_content(query: str):
    print(f"🔹 Querying document content: {query}")
    
    rag_chain = create_rag_chain()
    if rag_chain is None:
        print("❌ No stored documents found. Please add documents first (type 'adv').")
        return

    try:
        response = rag_chain.invoke(query)
        print(f"✅ Response: {response}")
    except Exception as e:
        print(f"❌ Error during query: {e}")

def main():
    while True:
        user_input = input("\n🔎 Query (or adv, txt, exit): ").strip()

        if user_input.lower() == "exit":
            print("👋 Exiting program. Have a great day!")
            break

        elif user_input.lower() == "adv":
            path = input("📂 Enter the full path to a file or directory: ").strip()
            if not os.path.exists(path):
                print("❌ Error: Path not found. Please check the input.")
                continue

            try:
                if os.path.isfile(path) and path.lower().endswith(('.mp4', '.avi', '.mov')):
                    transcript_text = transcribe_video(path)
                    documents = [Document(text=transcript_text, metadata={"source": path})]
                else:
                    reader = SimpleDirectoryReader(
                        input_dir=path if os.path.isdir(path) else None,
                        input_files=[path] if os.path.isfile(path) else None,
                        recursive=True,
                        required_exts=[".pdf", ".txt", ".docx", ".md", ".ipynb", ".jpg", ".jpeg", ".png"]
                    )
                    documents = reader.load_data()
                    for doc in documents:
                        if "source" not in doc.metadata:
                            doc.metadata["source"] = path
                    if not documents:
                        print("❌ No supported documents found at the specified path.")
                        continue

                print(f"✅ Loaded {len(documents)} document(s) from {path}")
                store_documents_in_chroma(documents)
            except Exception as e:
                print(f"❌ Error processing path: {e}")

        elif user_input.lower() == "txt":
            filename = input("📝 Enter filename for text dump (default: chroma_content.txt): ").strip()
            if not filename:
                filename = "chroma_content.txt"
            dump_chroma_content_to_file(filename)

        else:
            query_document_content(user_input)

if __name__ == "__main__":
    main()