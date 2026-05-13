import hashlib
from src.utils.exceptions.custom_app_exception import CustomAppException  
from src.config.model_config import models
import re
from src.config.config import  config
from src.utils.helpers.logger_helper import logger
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
import fitz

class IngestionPipeline:
    def __init__(self):
        self.llm = models.llm
        self.embeddings = models.embeddings
        
    async def semantic_score(self,chunk_embedding,graph_query_embedding):
        return cosine_similarity(
        [chunk_embedding],
        [graph_query_embedding]
        )[0][0]

    async def structure_score(self,text):
        score = 0
        text_lower = text.lower()

        if any(w in text_lower for w in ["if", "must", "should", "eligible"]):
            score += 2

        if re.search(r'(\d+\.)|(- )|(\•)', text):
            score += 2

        if re.search(r'\d+', text):
            score += 1

        return score

    async def keyword_score(self,text):
        text_lower = text.lower()
        kw = sum(1 for k in config.base_keywords if k in text_lower)
        rel = sum(1 for r in config.relation_words if r in text_lower)
        return kw + rel

    async def score_chunk(self,text, embedding,graph_query_embedding):
        return (
        await self.keyword_score(text) +
        await self.structure_score(text) +
        (await self.semantic_score(embedding, graph_query_embedding) * 5)
        )


    async def select_top_chunks(
        self,
        chunks,
        embeddings_list,
        graph_query_embedding,
        top_k=10,

    ):

        scored = []

        for i, chunk in enumerate(chunks):

            score = await self.score_chunk(
                chunk,
                embeddings_list[i],
                graph_query_embedding=graph_query_embedding
            )

            scored.append(
                (
                    chunk,
                    embeddings_list[i],
                    score
                )
            )

        scored.sort(
            key=lambda x: x[2],
            reverse=True
        )

        return scored[:top_k]
    
    async def normalize_entity(self,text: str):

        ENTITY_NORMALIZATION = {
            "financial aid": "Scholarship",
            "scholarship scheme": "Scholarship",
            "aid": "Scholarship",
            "govt": "Government",
            "tn govt": "Tamil Nadu Government"
        }

        text = text.strip().lower()

        if text in ENTITY_NORMALIZATION:
            return ENTITY_NORMALIZATION[text]

        return text.title()
    
    async def generate_entity_id(self,label: str, name: str):

        clean = f"{label}_{name}".lower().strip()

        clean = re.sub(r'[^a-z0-9]+', '_', clean)

        return clean
    
    async def sanitize_label(self,label: str):

    # remove spaces/special chars
        label = re.sub(r'[^a-zA-Z0-9_]', '_', label)

        # Neo4j labels cannot start with number
        if label and label[0].isdigit():
            label = f"LABEL_{label}"

        return label

    async def sanitize_relationship(self,rel: str):

        rel = rel.upper()

        rel = re.sub(
            r'[^A-Z0-9_]',
            '_',
            rel
        )

        if rel and rel[0].isdigit():
            rel = f"REL_{rel}"

        return rel

    async def ingest_pdf(self,file_name:str,content:str,embeddings, transformer:LLMGraphTransformer, graph:Neo4jGraph,graph_query_embedding):
        try:

            # -------------------------------
            # 1. Read PDF and add document node
            # -------------------------------
            doc =  fitz.open(stream=content, filetype="pdf")

            full_text = "\n".join([page.get_text() for page in doc])

            doc_id = hashlib.md5(
                 file_name.encode()
                ).hexdigest()
            

            graph.query("""
                MERGE (d:Document {doc_id:$doc_id})

                SET d.name = $name,
                    d.uploaded_at = timestamp()
                """, {
                    "doc_id": doc_id,
                    "name": file_name
            })

            # -------------------------------
            # 2. Chunking
            # -------------------------------



            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            chunks = splitter.split_text(full_text)

            logger.info(f"Total chunks: {len(chunks)}")

            # -------------------------------
            # 3. Batch Embeddings (FAST)
            # -------------------------------
            vectors = embeddings.embed_documents(chunks)


            # -------------------------------
            # 4. Store chunk with doc in relationship
            # -------------------------------


            chunk_docs = []

            for i, chunk in enumerate(chunks):

                chunk_id = hashlib.md5(
                    chunk.encode()
                ).hexdigest()

                vector = vectors[i]

                # store ALL chunks
                graph.query("""
                MERGE (c:Chunk {chunk_id:$chunk_id})

                SET c.text = $text,
                    c.embedding = $embedding,
                    c.created_at = timestamp()

                WITH c

                MATCH (d:Document {doc_id:$doc_id})

                MERGE (d)-[:HAS_CHUNK]->(c)
                """, {
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "embedding": vector,
                    "doc_id": doc_id
                })

                chunk_docs.append(
                    {
                        "chunk_id": chunk_id,
                        "text": chunk,
                        "embedding": vector
                    }
                )


        

            # -------------------------------
            # 5. Smart Chunk Selection
            # -------------------------------



            selected_chunks = await self.select_top_chunks(chunks, vectors, top_k=10,graph_query_embedding=graph_query_embedding)

        

            # -------------------------------
            # 6. Create Docs for Graph Extraction
            # -------------------------------

            graph_documents_input = []

            for chunk, vector, score in selected_chunks:

                chunk_id = hashlib.md5(
                    chunk.encode()
                ).hexdigest()

                graph_documents_input.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "chunk_id": chunk_id,
                            "score": score
                        }
                    )
                )
            logger.info(f"Selected {len(graph_documents_input)} chunks for graph")

            # -------------------------------
            # 7. Graph Extraction (LLM)
            # -------------------------------
            graph_docs = transformer.convert_to_graph_documents(graph_documents_input)

            # -------------------------------
            # 8. Insert Entities + Relationships
            # -------------------------------

            for g_doc in graph_docs:

                chunk_id = g_doc.source.metadata["chunk_id"]

                # ---------------------------------------------
                # 9.Insert Entity Nodes
                # ---------------------------------------------

                for node in g_doc.nodes:

                    entity_name = await self.normalize_entity(
                        node.id
                    )

                    entity_label = await self.sanitize_label(node.type)

                    entity_id = await self.generate_entity_id(
                        entity_label,
                        entity_name
                    )

                    graph.query(f"""
                    MERGE (e:Entity:{entity_label}
                    {{entity_id:$entity_id}})

                    ON CREATE SET
                        e.name = $name,
                        e.created_at = timestamp()

                    WITH e

                    MATCH (c:Chunk {{chunk_id:$chunk_id}})

                    MERGE (c)-[:MENTIONS]->(e)
                    """, {
                        "entity_id": entity_id,
                        "name": entity_name,
                        "chunk_id": chunk_id
                    })

                # ---------------------------------------------
                # 10.Insert Relationships
                # ---------------------------------------------

                for rel in g_doc.relationships:

                    source_name = await self.normalize_entity(
                        rel.source.id
                    )

                    target_name = await self.normalize_entity(
                        rel.target.id
                    )

                    source_label = rel.source.type
                    target_label = rel.target.type

                    source_id = await self.generate_entity_id(
                        source_label,
                        source_name
                    )

                    target_id = await self.generate_entity_id(
                        target_label,
                        target_name
                    )

                    relation_type = await self.sanitize_relationship(rel.type)

                    graph.query(f"""
                    MATCH (a:Entity
                    {{entity_id:$source_id}})

                    MATCH (b:Entity
                    {{entity_id:$target_id}})

                    MERGE (a)-[:{relation_type}]->(b)
                    """, {
                        "source_id": source_id,
                        "target_id": target_id
                    })

            # -------------------------------
            # 11. Create Vector Index
            # -------------------------------
            graph.query("""
            CREATE VECTOR INDEX chunk_embedding_index
            IF NOT EXISTS
            FOR (c:Chunk)
            ON (c.embedding)

            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: 512,
                    `vector.similarity_function`: 'cosine'
                }
            }
            """)

            logger.info("Graph ingestion completed successfully")


            return {
                "status": "success",
                "document": file_name,
                "total_chunks": len(chunks),
                "graph_chunks": len(selected_chunks),
                "graph_documents": len(graph_docs)
            }
    
        except Exception as e:
            logger.error(f"Ingestion failed: {str(e)}", exc_info=True)
            raise CustomAppException(status_code=500, content=str(e), err_code="INGESTION_FAILED")


def get_ingestion_pipeline():
    return IngestionPipeline()
