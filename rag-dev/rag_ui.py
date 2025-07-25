from rag_index_setup_02 import create_rag_system

# Create complete RAG system with cluster chunking
rag_system = create_rag_system(
    chunking_strategy='cluster'
)

# Query the system
answer = rag_system.query("2024 yılında Türkiye'degençler en çok hangi sorunlarla karşılaştı?")
print(answer)