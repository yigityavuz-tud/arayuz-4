from rag_utils_03 import RAGPipeline

# Basic usage
pipeline = RAGPipeline(
    pdf_dir="path/to/pdfs",
    openai_api_key="your_openai_key",
    pinecone_api_key="your_pinecone_key",
    index_name="your_index_name"
)

# Ask questions with different models
answer1 = pipeline.ask_question("Your question here", model="gpt-3.5-turbo")
answer2 = pipeline.ask_question("Your question here", model="deepseek-chat")

# Get statistics
stats = pipeline.get_statistics()
print(f"Total documents: {stats['total_chunks']}")