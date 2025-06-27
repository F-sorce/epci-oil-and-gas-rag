# import basics
import os
import time
import re
from dotenv import load_dotenv

# import pinecone
from pinecone import Pinecone, ServerlessSpec
from pinecone.openapi_support.exceptions import UnauthorizedException, PineconeApiException

# import langchain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

#documents
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import PyPDF2
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv() 

# Validate Pinecone API key
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY not found in environment variables. Please check your .env file.")

# Validate OpenAI API key
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

try:
    pc = Pinecone(api_key=pinecone_api_key)
    # Test the connection
    pc.list_indexes()
except UnauthorizedException as e:
    print("Error: Invalid Pinecone API key or malformed domain.")
    print("Please check your Pinecone API key in the .env file.")
    print("You can get your API key from: https://app.pinecone.io/")
    raise
except Exception as e:
    print(f"Error connecting to Pinecone: {str(e)}")
    raise

# initialize pinecone database
index_name = os.environ.get("PINECONE_INDEX_NAME", "langchain-rag-index")  # fallback to a valid name if env var not set

# Validate index name
if not re.match(r'^[a-z0-9-]+$', index_name):
    print(f"Invalid index name: {index_name}")
    print("Index name must consist of lowercase alphanumeric characters or hyphens")
    index_name = "langchain-rag-index"  # fallback to a valid name
    print(f"Using fallback index name: {index_name}")

# check whether index exists, and create if not
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    print(f"Creating new index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print("Waiting for index to be ready...")
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)
    print("Index is ready!")

# Wait for index to be fully ready
max_retries = 5
retry_delay = 2
for attempt in range(max_retries):
    try:
        index = pc.Index(index_name)
        # Test the index connection
        index.describe_index_stats()
        print("Successfully connected to index")
        break
    except Exception as e:
        if attempt < max_retries - 1:
            print(f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        else:
            print("Failed to connect to index after multiple attempts")
            raise

# initialize embeddings model + vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Process PDFs individually with error handling
documents = []
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=400,
    length_function=len,
    is_separator_regex=False,
)

def validate_pdf(file_path):
    """Validate PDF file and return True if valid, False otherwise."""
    try:
        with open(file_path, 'rb') as file:
            # Try to read the PDF
            reader = PyPDF2.PdfReader(file)
            # Check if PDF is encrypted
            if reader.is_encrypted:
                logger.warning(f"PDF is encrypted: {file_path}")
                return False
            # Check if PDF has pages
            if len(reader.pages) == 0:
                logger.warning(f"PDF has no pages: {file_path}")
                return False
            # Try to read first page to verify content
            reader.pages[0].extract_text()
            return True
    except Exception as e:
        logger.error(f"Error validating PDF {file_path}: {str(e)}")
        return False

# Get list of PDF files
pdf_files = [f for f in os.listdir("documents") if f.lower().endswith('.pdf')]

for pdf_file in pdf_files:
    file_path = os.path.join("documents", pdf_file)
    try:
        print(f"Processing {pdf_file}...")
        
        # Validate PDF before processing
        if not validate_pdf(file_path):
            print(f"Skipping {pdf_file} - PDF validation failed")
            continue
            
        loader = PyPDFLoader(file_path)
        raw_docs = loader.load()
        
        if not raw_docs:
            print(f"Warning: No content extracted from {pdf_file}")
            continue
            
        docs = text_splitter.split_documents(raw_docs)
        if not docs:
            print(f"Warning: No chunks created from {pdf_file}")
            continue
            
        documents.extend(docs)
        print(f"Successfully processed {pdf_file} - extracted {len(docs)} chunks")
    except Exception as e:
        print(f"Error processing {pdf_file}: {str(e)}")
        print("Please check if the PDF file is corrupted or try to repair it using a PDF repair tool.")
        continue

if not documents:
    print("No documents were successfully processed. Please check your PDF files.")
    exit(1)

print(f"Successfully processed {len(documents)} chunks from {len(pdf_files)} PDF files")

# Function to add documents in batches with retries
def add_documents_in_batches(documents, batch_size=100, max_retries=3):
    total_docs = len(documents)
    for i in range(0, total_docs, batch_size):
        batch = documents[i:i + batch_size]
        batch_ids = [f"id{j}" for j in range(i, min(i + batch_size, total_docs))]
        print(f"Adding batch {i//batch_size + 1} of {(total_docs + batch_size - 1)//batch_size}...")
        
        for attempt in range(max_retries):
            try:
                vector_store.add_documents(documents=batch, ids=batch_ids)
                print(f"Successfully added batch {i//batch_size + 1}")
                break
            except PineconeApiException as e:
                if "message length too large" in str(e):
                    if batch_size > 10:
                        print(f"Batch too large, reducing batch size to {batch_size//2}")
                        return add_documents_in_batches(documents, batch_size//2, max_retries)
                    else:
                        raise Exception("Even smallest batch size is too large. Please reduce chunk size.")
                elif attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Error adding batch {i//batch_size + 1}: {str(e)}")
                    raise

# add to database
print("Adding documents to Pinecone...")
try:
    add_documents_in_batches(documents)
    print("All documents successfully added to Pinecone!")
except Exception as e:
    print(f"Error adding documents to Pinecone: {str(e)}")
    raise