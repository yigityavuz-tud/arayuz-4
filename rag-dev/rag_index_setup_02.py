import os
import glob
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple, Optional

# Standard imports
from pprint import pprint
from langchain import hub
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore


class RAGProcessor:
    """
    A modular RAG processor that supports different chunking strategies
    """
    
    def __init__(self, 
                 txt_directory: str = r"C:\Users\yigit\Desktop\Enterprises\arayuz-3\okumalar-txt",
                 embedding_model: str = "text-embedding-3-small",
                 llm_model: str = "gpt-3.5-turbo",
                 temperature: float = 0):
        """
        Initialize RAG processor
        
        Args:
            txt_directory: Directory containing .txt files to load
            embedding_model: OpenAI embedding model to use
            llm_model: OpenAI LLM model to use
            temperature: Temperature for LLM responses
        """
        # Load environment variables
        load_dotenv()
        self._setup_environment()
        
        self.txt_directory = txt_directory
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.temperature = temperature
        
        # Initialize components
        self.docs = None
        self.full_text = ""
        self.chunks = []
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        
        # Initialize embedding function and LLM
        self.embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
        self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
        self.llm = ChatOpenAI(model=self.llm_model, temperature=self.temperature)
        
    def _setup_environment(self):
        """Setup environment variables"""
        # Access the environment variables
        langchain_tracing_v2 = os.getenv('LANGCHAIN_TRACING_V2')
        langchain_endpoint = os.getenv('LANGCHAIN_ENDPOINT')
        langchain_api_key = os.getenv('LANGCHAIN_API_KEY')
        openai_api_key = os.getenv('OPENAI_API_KEY')
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        
        # Set environment variables
        if langchain_tracing_v2:
            os.environ['LANGCHAIN_TRACING_V2'] = langchain_tracing_v2
        if langchain_endpoint:
            os.environ['LANGCHAIN_ENDPOINT'] = langchain_endpoint
        if langchain_api_key:
            os.environ['LANGCHAIN_API_KEY'] = langchain_api_key
        if openai_api_key:
            os.environ['OPENAI_API_KEY'] = openai_api_key
        if pinecone_api_key:
            os.environ['PINECONE_API_KEY'] = pinecone_api_key
            
        self.index_name = os.getenv('PINECONE_INDEX_NAME')
    
    @staticmethod
    def openai_token_count(text: str) -> int:
        """Count tokens using tiktoken for OpenAI models"""
        import tiktoken
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(encoding.encode(text))
    
    def load_documents(self) -> List[Any]:
        """
        Load all .txt documents from the specified directory
        
        Returns:
            List of loaded documents
        """
        print(f"Loading documents from: {self.txt_directory}")
        
        # Use DirectoryLoader to load all .txt files
        loader = DirectoryLoader(
            self.txt_directory, 
            glob="*.txt", 
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        
        self.docs = loader.load()
        print(f"Loaded {len(self.docs)} documents")
        
        # Combine all document content for chunking analysis
        self.full_text = "\n\n".join([doc.page_content for doc in self.docs])
        print(f"Total text length: {len(self.full_text)} characters")
        
        return self.docs
    
    def chunk_text(self, 
                   strategy: str = 'cluster',
                   cluster_max_chunk_size: int = 200,
                   recursive_chunk_size: int = 400,
                   recursive_chunk_overlap: int = 50,
                   separators: List[str] = None) -> List[str]:
        """
        Chunk the loaded text using specified strategy
        
        Args:
            strategy: 'cluster' for ClusterSemanticChunker or 'recursive' for RecursiveTokenChunker
            cluster_max_chunk_size: Max chunk size for cluster chunking (tokens)
            recursive_chunk_size: Chunk size for recursive chunking (characters)
            recursive_chunk_overlap: Overlap for recursive chunking (characters)
            separators: Custom separators for recursive chunking
            
        Returns:
            List of text chunks
        """
        if not self.full_text:
            raise ValueError("No documents loaded. Call load_documents() first.")
        
        if separators is None:
            separators = ["\n\n", "\n", ".", "?", "!", " ", ""]
        
        if strategy == 'cluster':
            chunker = ClusterSemanticChunker(
                embedding_function=self.embedding_function, 
                max_chunk_size=cluster_max_chunk_size, 
                length_function=self.openai_token_count
            )
            self.chunks = chunker.split_text(self.full_text)
            
        elif strategy == 'recursive':
            chunker = RecursiveTokenChunker(
                chunk_size=recursive_chunk_size,
                chunk_overlap=recursive_chunk_overlap,
                length_function=len,
                separators=separators
            )
            self.chunks = chunker.split_text(self.full_text)
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'cluster' or 'recursive'")
        
        print(f"Created {len(self.chunks)} chunks using {strategy} strategy")
        return self.chunks
    
    def analyze_chunks(self, chunks: Optional[List[str]] = None, use_tokens: bool = False) -> Dict[str, float]:
        """
        Analyze chunk statistics (separate method for optional analysis)
        
        Args:
            chunks: List of chunks to analyze (uses self.chunks if None)
            use_tokens: Whether to analyze by tokens (True) or characters (False)
            
        Returns:
            Dictionary with chunk statistics
        """
        if chunks is None:
            chunks = self.chunks
            
        if not chunks:
            raise ValueError("No chunks to analyze")
        
        if use_tokens:
            lengths = [self.openai_token_count(chunk) for chunk in chunks]
            unit = "tokens"
        else:
            lengths = [len(chunk) for chunk in chunks]
            unit = "characters"
        
        stats = {
            'count': len(chunks),
            'average': sum(lengths) / len(lengths),
            'max': max(lengths),
            'min': min(lengths),
            'unit': unit
        }
        
        print(f"Number of chunks: {stats['count']}")
        print(f"Average {unit} per chunk: {stats['average']:.2f}")
        print(f"Max {unit} in a chunk: {stats['max']}")
        print(f"Min {unit} in a chunk: {stats['min']}")
        
        return stats
    
    def create_vectorstore(self, method_name: str = None) -> Any:
        """
        Create vector store from chunks
        
        Args:
            method_name: Name to include in metadata (optional)
            
        Returns:
            Created vectorstore
        """
        if not self.chunks:
            raise ValueError("No chunks available. Call chunk_text() first.")
        
        # Convert chunks to document format
        splits = []
        for i, chunk in enumerate(self.chunks):
            metadata = {'chunk_id': i}
            if method_name:
                metadata['method'] = method_name
            splits.append({
                'page_content': chunk,
                'metadata': metadata
            }) 
        
        print(f"Creating vector store with {len(splits)} chunks...")
        
        # Create vector store
        self.vectorstore = Pinecone.from_documents(
            documents=[Document(page_content=split['page_content'], metadata=split['metadata']) for split in splits],
            embedding=self.embeddings, 
            index_name=self.index_name
        )
        
        self.retriever = self.vectorstore.as_retriever()
        print("Vector store created successfully!")
        
        return self.vectorstore
    
    def setup_rag_chain(self, 
                        custom_template: Optional[str] = None) -> Any:
        """
        Setup the RAG chain for question answering
        
        Args:
            custom_template: Custom prompt template (optional)
            
        Returns:
            RAG chain
        """
        if not self.retriever:
            raise ValueError("No retriever available. Call create_vectorstore() first.")
        
        # Default template
        if custom_template is None:
            template = """Answer the question based only on the following context:
{context}

If you don't know the answer, just say that you don't know.

Question: {question}
"""
        else:
            template = custom_template
        
        prompt = ChatPromptTemplate.from_template(template)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Create RAG chain
        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return self.rag_chain
    
    def query(self, question: str) -> str:
        """
        Query the RAG system
        
        Args:
            question: Question to ask
            
        Returns:
            Answer from the RAG system
        """
        if not self.rag_chain:
            raise ValueError("RAG chain not setup. Call setup_rag_chain() first.")
        
        return self.rag_chain.invoke(question)
    
    def build_complete_rag_system(self,
                                  chunking_strategy: str = 'cluster',
                                  cluster_max_chunk_size: int = 200,
                                  recursive_chunk_size: int = 400,
                                  recursive_chunk_overlap: int = 0,
                                  custom_template: Optional[str] = None) -> Tuple[Any, Any, Any]:
        """
        Complete pipeline: load documents, chunk, create vectorstore, setup RAG chain
        
        Args:
            chunking_strategy: 'cluster' or 'recursive'
            cluster_max_chunk_size: Max chunk size for cluster chunking
            recursive_chunk_size: Chunk size for recursive chunking
            recursive_chunk_overlap: Overlap for recursive chunking
            custom_template: Custom prompt template
            
        Returns:
            Tuple of (vectorstore, retriever, rag_chain)
        """
        # Load documents
        self.load_documents()
        
        # Chunk text
        self.chunk_text(
            strategy=chunking_strategy,
            cluster_max_chunk_size=cluster_max_chunk_size,
            recursive_chunk_size=recursive_chunk_size,
            recursive_chunk_overlap=recursive_chunk_overlap
        )
        
        # Create vectorstore
        self.create_vectorstore(method_name=chunking_strategy)
        
        # Setup RAG chain
        self.setup_rag_chain(custom_template=custom_template)
        
        print("\n" + "="*50)
        print("RAG SYSTEM READY")
        print("="*50)
        print(f"✓ Loaded {len(self.docs)} documents")
        print(f"✓ Created {len(self.chunks)} chunks using {chunking_strategy} method")
        print(f"✓ Vector store indexed with {self.index_name}")
        print("✓ RAG chain ready for queries")
        
        return self.vectorstore, self.retriever, self.rag_chain


# Convenience function for quick setup
def create_rag_system(txt_directory: str = r"C:\Users\yigit\Desktop\Enterprises\arayuz-3\okumalar-txt",
                      chunking_strategy: str = 'cluster',
                      cluster_max_chunk_size: int = 200,
                      recursive_chunk_size: int = 400,
                      recursive_chunk_overlap: int = 0,
                      embedding_model: str = "text-embedding-3-small") -> RAGProcessor:
    """
    Convenience function to create a complete RAG system
    
    Args:
        txt_directory: Directory containing .txt files
        chunking_strategy: 'cluster' or 'recursive'
        cluster_max_chunk_size: Max chunk size for cluster chunking
        recursive_chunk_size: Chunk size for recursive chunking
        recursive_chunk_overlap: Overlap for recursive chunking
        embedding_model: OpenAI embedding model to use
        
    Returns:
        Configured RAGProcessor instance
    """
    rag_processor = RAGProcessor(
        txt_directory=txt_directory,
        embedding_model=embedding_model
    )
    
    rag_processor.build_complete_rag_system(
        chunking_strategy=chunking_strategy,
        cluster_max_chunk_size=cluster_max_chunk_size,
        recursive_chunk_size=recursive_chunk_size,
        recursive_chunk_overlap=recursive_chunk_overlap
    )
    
    return rag_processor