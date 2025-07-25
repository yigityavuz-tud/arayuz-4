"""
RAG Pipeline Module for PDF Processing and Question Answering

This module provides a complete pipeline for processing PDF documents and answering questions
using various AI models. The pipeline supports running individual steps or the entire process.

Usage:
    from rag_pipeline import RAGPipeline
    
    pipeline = RAGPipeline(
        pdf_dir="path/to/pdfs",
        openai_api_key="your_key",
        pinecone_api_key="your_key",
        index_name="your_index"
    )
    
    # Run individual steps
    pipeline.pdf_to_json()
    pipeline.preprocess_json()
    pipeline.index_documents()
    
    # Or run the entire pipeline
    pipeline.run_full_pipeline()
    
    # Ask questions
    answer = pipeline.ask_question("Your question here", model="gpt-3.5-turbo")
"""

import os
import json
import re
import glob
from pathlib import Path
from typing import List, Optional, Dict, Literal
from tqdm import tqdm
import requests

# Document processing
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json

# LangChain imports
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains import RetrievalQA


class DocumentProcessor:
    """Handles document processing utilities."""
    
    @staticmethod
    def enumerate_text_parts_with_position(text_elements, exclude_types=None, column_threshold=50):
        """Enumerate consecutive parts of a page in reading order, assigning a 'position' field to each."""
        if exclude_types is None:
            exclude_types = []
        exclude_types_no_footer = [t for t in exclude_types if t.lower() != 'footer']

        footer_elements = [
            elem for elem in text_elements
            if elem.get('type', '').lower() == 'footer'
        ]
        non_footer_elements = [
            elem for elem in text_elements
            if elem.get('type', '').lower() not in [t.lower() for t in exclude_types_no_footer + ['footer']]
        ]

        if not non_footer_elements and not footer_elements:
            return []

        rows = {}
        for element in non_footer_elements:
            coords = element.get('coordinates', [[0, 0]])
            if coords:
                x, y = coords[0]
                row_key = round(y / 20) * 20
                if row_key not in rows:
                    rows[row_key] = []
                rows[row_key].append(element)

        sorted_rows = sorted(rows.items())

        ordered_elements = []
        for row_y, row_elements in sorted_rows:
            row_elements.sort(key=lambda elem: elem.get('coordinates', [[0, 0]])[0][0])
            ordered_elements.extend(row_elements)

        enumerated_parts = []
        position = 0
        for element in ordered_elements:
            text = element.get('text', '').strip()
            if text:
                text = re.sub(r' +', ' ', text)
                part = dict(element)
                part['text'] = text
                part['position'] = position
                enumerated_parts.append(part)
                position += 1

        for i, element in enumerate(footer_elements):
            text = element.get('text', '').strip()
            if text:
                text = re.sub(r' +', ' ', text)
                part = dict(element)
                part['text'] = text
                part['position'] = position + i
                enumerated_parts.append(part)

        return enumerated_parts

    @staticmethod
    def wrap_metadata(item):
        """Wrap all fields except 'element_id' and 'text' into a 'metadata' dict."""
        if not isinstance(item, dict):
            return item
        new_item = {}
        if "element_id" in item:
            new_item["element_id"] = item["element_id"]
        if "text" in item:
            new_item["text"] = item["text"]
        
        metadata = {}
        for k, v in item.items():
            if k not in ("element_id", "text", "metadata"):
                metadata[k] = v
            elif k == "metadata" and isinstance(v, dict):
                for mk, mv in v.items():
                    metadata[mk] = mv
        new_item["metadata"] = metadata
        return new_item

    @staticmethod
    def remove_metadata_fields(metadata, fields_to_remove):
        """Remove specified fields from a metadata dict."""
        if not isinstance(metadata, dict):
            return metadata
        for field in fields_to_remove:
            metadata.pop(field, None)
        return metadata

    @staticmethod
    def convert_languages_field_in_metadata(metadata):
        """If 'languages' in metadata is a list of length 1, convert it to a string."""
        if isinstance(metadata, dict) and "languages" in metadata:
            if isinstance(metadata["languages"], list) and len(metadata["languages"]) == 1:
                metadata["languages"] = metadata["languages"][0]
        return metadata

    @staticmethod
    def replace_nulls(obj):
        """Recursively replace None values in dicts/lists with a space character."""
        if isinstance(obj, dict):
            return {k: DocumentProcessor.replace_nulls(v) if v is not None else " " for k, v in obj.items()}
        elif isinstance(obj, list):
            return [DocumentProcessor.replace_nulls(v) for v in obj]
        else:
            return obj if obj is not None else " "

    @staticmethod
    def can_merge_nodes(node1, node2):
        """Return True if node1 and node2 can be merged according to the rules."""
        if not (isinstance(node1, dict) and isinstance(node2, dict)):
            return False
        
        meta1 = node1.get("metadata", {})
        meta2 = node2.get("metadata", {})
        
        # Compare all metadata fields except 'position'
        meta1_without_pos = {k: v for k, v in meta1.items() if k != "position"}
        meta2_without_pos = {k: v for k, v in meta2.items() if k != "position"}
        
        if meta1_without_pos != meta2_without_pos:
            return False
        
        text1 = node1.get("text", "")
        text2 = node2.get("text", "")
        
        if not (isinstance(text1, str) and isinstance(text2, str)):
            return False
        
        if len(text1) + len(text2) < 200:
            return True
        return False

    @staticmethod
    def merge_nodes(node1, node2):
        """Merge node2 into node1, concatenating text and updating position to the lower of the two."""
        merged = dict(node1)
        merged["text"] = node1.get("text", "") + " " + node2.get("text", "")
        
        meta1 = node1.get("metadata", {})
        meta2 = node2.get("metadata", {})
        merged_meta = dict(meta1)
        
        pos1 = meta1.get("position", node1.get("position", None))
        pos2 = meta2.get("position", node2.get("position", None))
        
        if pos1 is not None and pos2 is not None:
            merged_meta["position"] = min(pos1, pos2)
        elif pos1 is not None:
            merged_meta["position"] = pos1
        elif pos2 is not None:
            merged_meta["position"] = pos2
            
        merged["metadata"] = merged_meta
        return merged


class RAGPipeline:
    """Main RAG Pipeline class for processing PDFs and answering questions."""
    
    def __init__(
        self,
        pdf_dir: str,
        openai_api_key: Optional[str] = None,
        pinecone_api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        deepseek_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        chunk_size: int = 300,
        chunk_overlap: int = 30
    ):
        """
        Initialize the RAG Pipeline.
        
        Args:
            pdf_dir: Directory containing PDF files
            openai_api_key: OpenAI API key
            pinecone_api_key: Pinecone API key
            index_name: Pinecone index name
            deepseek_api_key: DeepSeek API key (optional)
            google_api_key: Google API key (optional)
            chunk_size: Text chunk size for splitting
            chunk_overlap: Text chunk overlap
        """
        self.pdf_dir = pdf_dir
        self.json_dir = os.path.join(os.path.dirname(pdf_dir.rstrip("/\\")), "okumalar-json")
        self.prep_dir = os.path.join(os.path.dirname(pdf_dir.rstrip("/\\")), "okumalar-prep")
        
        # API keys
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.pinecone_api_key = pinecone_api_key or os.getenv('PINECONE_API_KEY')
        self.index_name = index_name or os.getenv('PINECONE_INDEX_NAME')
        self.deepseek_api_key = deepseek_api_key or os.getenv('DEEPSEEK_API_KEY')
        self.google_api_key = google_api_key or os.getenv('GOOGLE_API_KEY')
        
        # Text processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
        # Initialize components
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        
        # Create directories
        os.makedirs(self.json_dir, exist_ok=True)
        os.makedirs(self.prep_dir, exist_ok=True)

    def pdf_to_json(self, force_reprocess: bool = False) -> None:
        """
        Convert PDF files to JSON format using unstructured.
        
        Args:
            force_reprocess: If True, reprocess all PDFs even if JSON exists
        """
        print("🔄 Starting PDF to JSON conversion...")
        
        pdf_files = [f for f in os.listdir(self.pdf_dir) if f.lower().endswith(".pdf")]
        json_files = [f for f in os.listdir(self.json_dir) if f.lower().endswith('.json')]

        pdf_basenames = set(os.path.splitext(f)[0] for f in pdf_files)
        json_basenames = set(os.path.splitext(f)[0] for f in json_files)

        if force_reprocess:
            unprocessed_basenames = pdf_basenames
        else:
            unprocessed_basenames = pdf_basenames - json_basenames

        if not unprocessed_basenames:
            print("✅ All PDFs have been processed to JSON.")
            return

        print(f"📄 Processing {len(unprocessed_basenames)} PDF files...")
        
        for base_file_name in tqdm(sorted(unprocessed_basenames), desc="Converting PDFs"):
            pdf_path = os.path.join(self.pdf_dir, f"{base_file_name}.pdf")
            json_path = os.path.join(self.json_dir, f"{base_file_name}.json")
            
            try:
                print(f"Processing: {pdf_path}")
                elements = partition_pdf(
                    filename=pdf_path,
                    languages=["tur"],
                    strategy="fast",
                    infer_table_structure=True,
                )
                elements_to_json(elements=elements, filename=json_path)
                print(f"✅ Saved: {json_path}")
            except Exception as e:
                print(f"❌ Error processing {pdf_path}: {str(e)}")

    def preprocess_json(self, force_reprocess: bool = False) -> None:
        """
        Preprocess JSON files: clean, merge, split, and prepare for indexing.
        
        Args:
            force_reprocess: If True, reprocess all JSON files even if prep files exist
        """
        print("🔄 Starting JSON preprocessing...")
        
        json_files = [f for f in os.listdir(self.json_dir) if f.lower().endswith('.json')]
        
        if not json_files:
            print("❌ No JSON files found. Run pdf_to_json() first.")
            return

        processed_files = []
        
        for json_file in tqdm(json_files, desc="Preprocessing JSON"):
            json_path = os.path.join(self.json_dir, json_file)
            prep_json_path = os.path.join(self.prep_dir, json_file)
            
            # Skip if already processed and not forcing reprocess
            if not force_reprocess and os.path.exists(prep_json_path):
                continue
                
            try:
                print(f"\nProcessing JSON file: {json_file}")

                with open(json_path, "r", encoding="utf-8") as f:
                    json_data = json.load(f)

                # Process the data
                enumerated_json_data = DocumentProcessor.enumerate_text_parts_with_position(json_data)
                processed_data = self._process_json_data(enumerated_json_data)
                sorted_data = self._add_title_fields(processed_data)
                merged_data = self._merge_similar_nodes(sorted_data)
                split_json_data = self._split_text_content(merged_data)

                print(f"Number of nodes before split: {len(merged_data)}")
                print(f"Number of nodes after split: {len(split_json_data)}")

                # Save processed data
                with open(prep_json_path, "w", encoding="utf-8") as f:
                    json.dump(split_json_data, f, ensure_ascii=False, indent=2)
                
                processed_files.append(json_file)
                print(f"✅ Saved processed JSON to: {prep_json_path}")
                
            except Exception as e:
                print(f"❌ Error processing {json_file}: {str(e)}")

        if processed_files:
            print(f"✅ Successfully processed {len(processed_files)} JSON files")
        else:
            print("✅ All JSON files already preprocessed")

    def index_documents(self, force_reindex: bool = False) -> None:
        """
        Index processed documents into Pinecone vector store.
        
        Args:
            force_reindex: If True, recreate the entire index
        """
        print("🔄 Starting document indexing...")
        
        if not self.openai_api_key or not self.pinecone_api_key or not self.index_name:
            raise ValueError("OpenAI API key, Pinecone API key, and index name are required for indexing")

        prep_files = glob.glob(os.path.join(self.prep_dir, "*.json"))
        
        if not prep_files:
            print("❌ No preprocessed files found. Run preprocess_json() first.")
            return

        print(f"📚 Loading documents from {len(prep_files)} files...")
        
        all_documents = []
        for json_path in tqdm(prep_files, desc="Loading documents"):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    split_json_data = json.load(f)
                
                documents = [
                    Document(
                        page_content=element['text'],
                        metadata=element['metadata'] | {'element_id': element['element_id']}
                    )
                    for element in split_json_data
                    if element.get('text', '').strip()  # Only include non-empty text
                ]
                all_documents.extend(documents)
            except Exception as e:
                print(f"❌ Error loading {json_path}: {str(e)}")

        print(f"📄 Total documents to index: {len(all_documents)}")

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=self.openai_api_key
        )

        try:
            if force_reindex:
                print("🔄 Creating new vector store...")
                self.vectorstore = PineconeVectorStore.from_documents(
                    documents=all_documents,
                    embedding=self.embeddings,
                    index_name=self.index_name
                )
            else:
                print("🔄 Adding documents to existing vector store...")
                self.vectorstore = PineconeVectorStore(
                    index_name=self.index_name,
                    embedding=self.embeddings
                )
                # Add documents in batches to avoid rate limits
                batch_size = 100
                for i in tqdm(range(0, len(all_documents), batch_size), desc="Indexing batches"):
                    batch = all_documents[i:i+batch_size]
                    self.vectorstore.add_documents(batch)

            print("✅ Documents successfully indexed!")
            
        except Exception as e:
            print(f"❌ Error during indexing: {str(e)}")
            raise

    def setup_retriever(self) -> None:
        """Setup the self-query retriever."""
        if not self.vectorstore:
            if not self.openai_api_key or not self.pinecone_api_key or not self.index_name:
                raise ValueError("Vector store not initialized. Run index_documents() first or provide API keys.")
            
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=self.openai_api_key
            )
            self.vectorstore = PineconeVectorStore(
                index_name=self.index_name,
                embedding=self.embeddings
            )

        metadata_field_info = [
            AttributeInfo(
                name="filename",
                description="The name of the PDF document the text comes from",
                type="string",
            ),
            AttributeInfo(
                name="languages",
                description="The list of language codes used in the document (e.g. 'tur', 'eng')",
                type="list[string]",
            ),
            AttributeInfo(
                name="page_number",
                description="The page number within the PDF document",
                type="integer",
            ),
            AttributeInfo(
                name="type",
                description="The structural type of the text chunk (e.g. Title, Paragraph, List, Quote)",
                type="string",
            ),
            AttributeInfo(
                name="title",
                description="The section title of the chunk",
                type="string",
            ),
        ]

        document_content_description = "Sociological and political researches and analyses"
        
        llm = ChatOpenAI(temperature=0, openai_api_key=self.openai_api_key)

        self.retriever = SelfQueryRetriever.from_llm(
            llm=llm,
            vectorstore=self.vectorstore,
            document_contents="text",
            document_content_description=document_content_description,
            metadata_field_info=metadata_field_info,
            search_kwargs={"k": 5}
        )

    def ask_question(
        self,
        question: str,
        model: Literal["gpt-4", "gpt-3.5-turbo", "deepseek-chat", "gemini-2.5-pro"] = "gpt-3.5-turbo",
        context_k: int = 4
    ) -> str:
        """
        Ask a question and get an answer using the specified model.
        
        Args:
            question: The question to ask
            model: The model to use for answering
            context_k: Number of context documents to retrieve
            
        Returns:
            The answer string
        """
        print(f"🤔 Asking question with {model}...")
        
        # Setup retriever if not already done
        if not self.retriever:
            self.setup_retriever()

        # Retrieve relevant documents
        docs = self.vectorstore.as_retriever(search_kwargs={"k": context_k}).get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Prepare prompt
        prompt_template = """Aşağıda bazı metin parçaları verilmiştir. Bu bilgilere dayanarak son kullanıcı sorusuna tarafsız ve güvenli bir şekilde yanıt ver:

Metin parçaları:
{context}

Soru: {question}
Yanıt:"""

        if model in ["gpt-4", "gpt-3.5-turbo"]:
            return self._ask_openai(question, context, prompt_template, model)
        elif model == "deepseek-chat":
            return self._ask_deepseek(question, context, prompt_template)
        elif model == "gemini-2.5-pro":
            return self._ask_gemini(question, context, prompt_template)
        else:
            raise ValueError(f"Unsupported model: {model}")

    def run_full_pipeline(
        self,
        force_reprocess_pdf: bool = False,
        force_reprocess_json: bool = False,
        force_reindex: bool = False
    ) -> None:
        """
        Run the complete pipeline: PDF to JSON, preprocessing, and indexing.
        
        Args:
            force_reprocess_pdf: Force reprocessing of PDFs
            force_reprocess_json: Force reprocessing of JSON files
            force_reindex: Force reindexing of documents
        """
        print("🚀 Starting full RAG pipeline...")
        
        self.pdf_to_json(force_reprocess=force_reprocess_pdf)
        self.preprocess_json(force_reprocess=force_reprocess_json)
        self.index_documents(force_reindex=force_reindex)
        
        print("✅ Full pipeline completed successfully!")

    def get_statistics(self) -> Dict:
        """Get statistics about the processed files."""
        stats = {
            "pdf_files": 0,
            "json_files": 0,
            "prep_files": 0,
            "total_chunks": 0,
            "avg_chunk_length": 0,
            "min_chunk_length": 0,
            "max_chunk_length": 0
        }
        
        # Count files
        if os.path.exists(self.pdf_dir):
            stats["pdf_files"] = len([f for f in os.listdir(self.pdf_dir) if f.lower().endswith(".pdf")])
        
        if os.path.exists(self.json_dir):
            stats["json_files"] = len([f for f in os.listdir(self.json_dir) if f.lower().endswith(".json")])
        
        if os.path.exists(self.prep_dir):
            stats["prep_files"] = len([f for f in os.listdir(self.prep_dir) if f.lower().endswith(".json")])
            
            # Analyze chunks
            all_texts = []
            for json_file in glob.glob(os.path.join(self.prep_dir, "*.json")):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    all_texts.extend([element['text'] for element in data if 'text' in element])
                except:
                    continue
            
            if all_texts:
                lengths = [len(text) for text in all_texts]
                stats["total_chunks"] = len(lengths)
                stats["avg_chunk_length"] = sum(lengths) / len(lengths)
                stats["min_chunk_length"] = min(lengths)
                stats["max_chunk_length"] = max(lengths)
        
        return stats

    # Private helper methods
    def _process_json_data(self, enumerated_json_data):
        """Process and clean JSON data."""
        fields_to_remove = ["coordinates", "file_directory", "filetype", "last_modified"]
        processed_data = []
        
        if isinstance(enumerated_json_data, list):
            for item in enumerated_json_data:
                if isinstance(item, dict):
                    # Clean text
                    if "text" in item and isinstance(item["text"], str):
                        item["text"] = re.sub(r'(?<=[A-Za-zÇĞİÖŞÜçğıöşü])\- (?=[A-Za-zÇĞİÖŞÜçğıöşü])', '', item["text"])
                    
                    # Process metadata
                    wrapped = DocumentProcessor.wrap_metadata(item)
                    wrapped["metadata"] = DocumentProcessor.remove_metadata_fields(wrapped.get("metadata", {}), fields_to_remove)
                    wrapped["metadata"] = DocumentProcessor.convert_languages_field_in_metadata(wrapped.get("metadata", {}))
                    wrapped = DocumentProcessor.replace_nulls(wrapped)
                    processed_data.append(wrapped)
                else:
                    processed_data.append(DocumentProcessor.replace_nulls(item))
        
        return processed_data

    def _add_title_fields(self, processed_data):
        """Add title fields to processed data."""
        def get_position(x):
            if isinstance(x, dict):
                if "position" in x:
                    return x["position"]
                elif "metadata" in x and isinstance(x["metadata"], dict) and "position" in x["metadata"]:
                    return x["metadata"]["position"]
            return -1

        def get_type(x):
            if isinstance(x, dict):
                if "type" in x:
                    return x["type"]
                elif "metadata" in x and isinstance(x["metadata"], dict) and "type" in x["metadata"]:
                    return x["metadata"]["type"]
            return None

        def get_text(x):
            if isinstance(x, dict):
                return x.get("text", None)
            return None

        sorted_data = sorted(processed_data, key=get_position)
        position_to_title = {}
        last_title_text = None

        for item in sorted_data:
            item_type = get_type(item)
            item_text = get_text(item)
            item_position = get_position(item)
            
            if item_type == "Title" and item_text:
                last_title_text = item_text
            if item_position is not None:
                position_to_title[item_position] = last_title_text

        for item in sorted_data:
            item_position = get_position(item)
            title_val = position_to_title.get(item_position, None)
            
            if isinstance(item, dict):
                if "metadata" not in item or not isinstance(item["metadata"], dict):
                    item["metadata"] = {}
                item["metadata"]["title"] = title_val if title_val is not None else " "

        return sorted_data

    def _merge_similar_nodes(self, sorted_data):
        """Merge similar consecutive nodes."""
        merged_data = []
        i = 0
        while i < len(sorted_data):
            current = sorted_data[i]
            if (i + 1 < len(sorted_data) and 
                DocumentProcessor.can_merge_nodes(current, sorted_data[i + 1])):
                merged = DocumentProcessor.merge_nodes(current, sorted_data[i + 1])
                merged_data.append(merged)
                i += 2
            else:
                merged_data.append(current)
                i += 1
        return merged_data

    def _split_text_content(self, merged_data):
        """Split text content using the text splitter."""
        split_json_data = []
        for item in merged_data:
            if isinstance(item, dict) and "text" in item and isinstance(item["text"], str):
                splits = self.text_splitter.split_text(item["text"])
                for split_text in splits:
                    new_item = dict(item)
                    new_item["text"] = split_text
                    new_item = DocumentProcessor.replace_nulls(new_item)
                    split_json_data.append(new_item)
            else:
                split_json_data.append(DocumentProcessor.replace_nulls(item))
        return split_json_data

    def _ask_openai(self, question, context, prompt_template, model):
        """Ask question using OpenAI models."""
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model=model, temperature=0, openai_api_key=self.openai_api_key),
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt}
        )
        
        return qa_chain.run(question)

    def _ask_deepseek(self, question, context, prompt_template):
        """Ask question using DeepSeek API."""
        if not self.deepseek_api_key:
            raise ValueError("DeepSeek API key is required")
        
        prompt_text = prompt_template.format(context=context, question=question)
        
        headers = {
            "Authorization": f"Bearer {self.deepseek_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": prompt_text}
            ]
        }
        
        try:
            response = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise Exception(f"DeepSeek API error: {str(e)}")

    def _ask_gemini(self, question, context, prompt_template):
        """Ask question using Google Gemini API."""
        if not self.google_api_key:
            raise ValueError("Google API key is required")
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.google_api_key)
            
            model = genai.GenerativeModel("models/gemini-2.5-pro")
            prompt_text = prompt_template.format(context=context, question=question)
            
            response = model.generate_content(prompt_text)
            return response.text
        except ImportError:
            raise ImportError("google-generativeai package is required for Gemini support")
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")


# Example usage and CLI interface
def main():
    """Main function demonstrating pipeline usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Pipeline for PDF Processing and Q&A")
    parser.add_argument("--pdf-dir", required=True, help="Directory containing PDF files")
    parser.add_argument("--step", choices=["pdf2json", "preprocess", "index", "ask", "full"], 
                       default="full", help="Pipeline step to run")
    parser.add_argument("--question", help="Question to ask (required for 'ask' step)")
    parser.add_argument("--model", choices=["gpt-4", "gpt-3.5-turbo", "deepseek-chat", "gemini-2.5-pro"],
                       default="gpt-3.5-turbo", help="Model to use for answering")
    parser.add_argument("--force-reprocess", action="store_true", 
                       help="Force reprocessing of existing files")
    parser.add_argument("--stats", action="store_true", help="Show pipeline statistics")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = RAGPipeline(pdf_dir=args.pdf_dir)
    
    # Show statistics if requested
    if args.stats:
        stats = pipeline.get_statistics()
        print("\n📊 Pipeline Statistics:")
        print(f"PDF files: {stats['pdf_files']}")
        print(f"JSON files: {stats['json_files']}")
        print(f"Preprocessed files: {stats['prep_files']}")
        print(f"Total chunks: {stats['total_chunks']}")
        if stats['total_chunks'] > 0:
            print(f"Average chunk length: {stats['avg_chunk_length']:.2f} characters")
            print(f"Min chunk length: {stats['min_chunk_length']} characters")
            print(f"Max chunk length: {stats['max_chunk_length']} characters")
        print()
    
    # Execute requested step
    if args.step == "pdf2json":
        pipeline.pdf_to_json(force_reprocess=args.force_reprocess)
    elif args.step == "preprocess":
        pipeline.preprocess_json(force_reprocess=args.force_reprocess)
    elif args.step == "index":
        pipeline.index_documents(force_reindex=args.force_reprocess)
    elif args.step == "ask":
        if not args.question:
            print("❌ Question is required for 'ask' step")
            return
        answer = pipeline.ask_question(args.question, model=args.model)
        print(f"\n🤖 Answer ({args.model}):")
        print(answer)
    elif args.step == "full":
        pipeline.run_full_pipeline(
            force_reprocess_pdf=args.force_reprocess,
            force_reprocess_json=args.force_reprocess,
            force_reindex=args.force_reprocess
        )


if __name__ == "__main__":
    main()


# Example usage in code:
"""
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

# Or run everything at once
pipeline.run_full_pipeline()

# Ask questions with different models
answer1 = pipeline.ask_question("Your question here", model="gpt-3.5-turbo")
answer2 = pipeline.ask_question("Your question here", model="deepseek-chat")

# Get statistics
stats = pipeline.get_statistics()
print(f"Total documents: {stats['total_chunks']}")

# Force reprocessing
pipeline.run_full_pipeline(
    force_reprocess_pdf=True,
    force_reprocess_json=True,
    force_reindex=True
)
"""