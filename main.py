from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("Loading the resume...")
loader = PyPDFLoader("MyResume.pdf")
resume_pages = loader.load()

print("Chunking the text...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,     # Each chunk will be around 500 characters long
    chunk_overlap=50    # Keep 50 characters of overlap so we don't cut a sentence in half
)

resume_chunks = text_splitter.split_documents(resume_pages)

print(f"Successfully split the resume into {len(resume_chunks)} chunks!")
print("Here is a look at the first chunk:")
print(resume_chunks[0].page_content)