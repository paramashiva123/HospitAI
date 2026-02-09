import os
import sqlite3
import hashlib
import asyncio
from collections import defaultdict
from typing import List
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.tools import tool
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import utility,connections,FieldSchema,CollectionSchema,DataType,Collection

os.environ["LITELLM_DISABLE_LOGGING"] = "true"
os.environ["LITELLM_LOG"] = "ERROR"
os.environ["OTEL_SDK_DISABLED"] = "true"

load_dotenv()

# CONFIGURATIONS

DATA_DIR = "data"
DB_PATH = "medical.db"

MILVUS_URI = "127.0.0.1"
MILVUS_PORT = "19530"

COLLECTION_NAME = "hospital_rag"

PARTITIONS = {
    "doctor_guide",
    "financial_data",
    "patient_data",
}

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 150
TOP_K = 10

llm = LLM(
    model="huggingface/Qwen/Qwen3-VL-30B-A3B-Instruct",
    temperature=0.2,
    max_tokens=512
)

# DEDUPLICATION

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_hash TEXT UNIQUE,
                source TEXT
            )
            """
        )
        conn.commit()


def document_exists(content_hash: str) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM documents WHERE content_hash = ?",
            (content_hash,),
        )
        return cur.fetchone() is not None


def register_document(content_hash: str, source: str):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO documents (content_hash, source) VALUES (?, ?)",
            (content_hash, source),
        )
        conn.commit()

def generate_content_hash(text : str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def get_or_create_event_loop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
    
get_or_create_event_loop()
# MILVUS SCHEMA

def create_collection():
    connections.connect(
        alias = "default",
        host = MILVUS_URI,
        port = MILVUS_PORT,
    )

    if utility.has_collection(COLLECTION_NAME):
        return Collection(COLLECTION_NAME)
    
    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True,
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=384,
        ),
        FieldSchema(
            name="content",
            dtype=DataType.VARCHAR,
            max_length=65535,
        ),
        FieldSchema(
            name="source",
            dtype=DataType.VARCHAR,
            max_length=512,
        ),
        FieldSchema(
            name="domain",
            dtype=DataType.VARCHAR,
            max_length=64,
        ),
    ]

    schema = CollectionSchema(
        fields=fields,
        description="Medical RAG for hospitals",
    )

    collection = Collection(
        name=COLLECTION_NAME,
        schema=schema,
    )

    collection.create_index(
        field_name="embedding",
        index_params={
            "index_type" : "HNSW",
            "metric_type" : "IP",
            "params" : {"M": 16, "efConstruction": 200},
        },
    )

    for partition in PARTITIONS:
        collection.create_partition(partition)
    return collection

# INGESTION

def ingest_documents(collection : Collection):
    init_db()

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
    )

    for domain in PARTITIONS:
        domain_path = os.path.join(DATA_DIR,domain)

        if not os.path.isdir(domain_path):
            continue

        print(f"\n [DOMAIN] {domain}")

        loader = DirectoryLoader(
            domain_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
        )

        pages = loader.load()
        pdf_map = defaultdict(list)

        for p in pages:
            pdf_map[p.metadata["source"]].append(p)

        for source , page_docs in pdf_map.items():
            full_text = "\n".join(p.page_content for p in page_docs)
            content_hash = generate_content_hash(full_text)

            if document_exists(content_hash):
                print(f"[SKIPPED] {source}")
                continue

            print(f"[INGEST] {source}")

            rows = []

            for page in page_docs:
                chunks = splitter.split_text(page.page_content)
                for i,chunk in enumerate(chunks):
                    rows.append({
                        "embedding" : embeddings.embed_query(chunk),
                        "content" : chunk,
                        "source" : source,
                        "domain" : domain,
                    })

            if rows:
                collection.insert(
                    [
                        [r["embedding"] for r in rows],
                        [r["content"] for r in rows],
                        [r["source"] for r in rows],
                        [r["domain"] for r in rows],
                    ],
                    partition_name=domain,
                )

            register_document(content_hash, source)
        
    collection.flush()
    collection.load()
        

# VECTORSTORE
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def _build_vectorstore():
    create_collection()
    ingest_documents(Collection(COLLECTION_NAME))

    return Milvus(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    connection_args={"host": MILVUS_URI, "port": MILVUS_PORT},
    vector_field="embedding",
    text_field="content",
    )


VECTOR_DB =_build_vectorstore()

def intent_classifier(query: str) -> str:
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1)

    template = """
    You’re a LLM that detects intent from user queries. Your task is to classify the user's intent based on their query. Below are the possible intents with brief descriptions. Use these to accurately determine the user's goal, and output only the intent topic.

    - doctor_guide: Inquiries about the treatments ,medicines ,diseases.

    - financial_data: Questions regarding financial data of hospitals.

    - patient_data: Queries related to patients info and their details.

    - Other: Choose this if the query doesn’t fall into any of the other intents.

    User Query: {query}

    Response:
    """
    prompt = PromptTemplate.from_template(template)


    chain = prompt | llm | StrOutputParser()

    response = chain.invoke(query)
    return response


@tool("retrieved_context")
def retrieved_context(query: str) -> str:
    "This Tool retrieves the relevant context the from Knowledge base"
    partition = intent_classifier(query)

    print(partition) 

    results = VECTOR_DB.similarity_search(
        query,
        k=5,
        expr=f"domain == '{partition}'"
    )
    if not results:
            results = VECTOR_DB.similarity_search(
            query,
            k=5
            )  

    return "\n\n".join(
        f"[{r.metadata.get('domain')} | source {r.metadata.get('source')}]\n"
        f"{r.page_content}"
        for r in results
    )

# CREW

@CrewBase
class RagCrew():
    """ Medical Rag crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"


    @agent
    def answerer(self) -> Agent:
        return Agent(
            config=self.agents_config['answerer'],
            tools=[retrieved_context],
            llm=llm,
            verbose=True
        )


    @task
    def answerer_task(self) -> Task:
        return Task(
            config=self.tasks_config['answerer_task'],
            output_key="final_answer"
            )

    @crew
    def crew(self) -> Crew:
        """Creates the RagCrew crew"""

        return Crew(
            agents=self.agents,
            tasks=[
                self.answerer_task()
                ],
            process=Process.sequential,
            verbose=True,
        )
