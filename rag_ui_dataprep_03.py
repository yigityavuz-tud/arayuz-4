from rag_utils_03 import RAGPipeline

# Basic usage
pipeline = RAGPipeline(
    pdf_dir="path/to/pdfs",
    openai_api_key="your_openai_key",
    pinecone_api_key="your_pinecone_key",
    index_name="your_index_name"
)

# Run individual steps
pipeline.pdf_to_json()  # Convert PDFs to JSON
pipeline.preprocess_json()  # Clean and prepare JSON files
pipeline.index_documents()  # Index documents in Pinecone

# # Or run everything at once
# pipeline.run_full_pipeline()