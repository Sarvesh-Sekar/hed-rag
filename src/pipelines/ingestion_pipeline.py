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
        try:
            return cosine_similarity(
                [chunk_embedding],
                [graph_query_embedding]
            )[0][0]
        except Exception as e:
            logger.error(f"semantic_score failed: {e}", exc_info=True)
            # leaf helper: raise generic Exception to caller
            raise CustomAppException(status_code=500, content=str(e), err_code="SEMANTIC_SCORE_FAILED")

    async def structure_score(self,text):
        try:
            score = 0
            text_lower = text.lower()

            if any(w in text_lower for w in ["if", "must", "should", "eligible"]):
                score += 2

            if re.search(r'(\d+\.)|(- )|(\•)', text):
                score += 2

            if re.search(r'\d+', text):
                score += 1

            return score
        except Exception as e:
            logger.error(f"structure_score failed: {e}", exc_info=True)
            # leaf helper: raise generic Exception
            raise CustomAppException(status_code=500, content=str(e), err_code="STRUCTURE_SCORE_FAILED")

    async def keyword_score(self,text):
        try:
            text_lower = text.lower()
            kw = sum(1 for k in config.base_keywords if k in text_lower)
            rel = sum(1 for r in config.relation_words if r in text_lower)
            return kw + rel
        except Exception as e:
            logger.error(f"keyword_score failed: {e}", exc_info=True)
            # leaf helper: raise generic Exception
            raise CustomAppException(status_code=500, content=str(e), err_code="KEYWORD_SCORE_FAILED")

    async def score_chunk(self,text, embedding,graph_query_embedding):
        try:
            return (
                await self.keyword_score(text) +
                await self.structure_score(text) +
                (await self.semantic_score(embedding, graph_query_embedding) * 5)
            )
        except CustomAppException:
            # propagate existing domain errors
            raise
        except Exception as e:
            logger.error(f"score_chunk failed: {e}", exc_info=True)
            # raise CustomAppException so callers (like select_top_chunks/ingest) get a structured error
            raise CustomAppException(status_code=500, content=str(e), err_code="SCORE_CHUNK_FAILED")


    async def select_top_chunks(
        self,
        chunks,
        embeddings_list,
        graph_query_embedding,
        top_k=10,

    ):
        try:
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
        except CustomAppException:
            raise
        except Exception as e:
            logger.error(f"select_top_chunks failed: {e}", exc_info=True)
            raise CustomAppException(status_code=500, content=str(e), err_code="SELECT_TOP_CHUNKS_FAILED")
    
    async def normalize_entity(self,text: str):
        try:
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
        except Exception as e:
            logger.error(f"normalize_entity failed: {e}", exc_info=True)
            # leaf helper: re-raise generic Exception
            raise CustomAppException(status_code=500, content=str(e), err_code="NORMALIZE_ENTITY_FAILED")
    
    async def generate_entity_id(self,label: str, name: str):
        try:
            clean = f"{label}_{name}".lower().strip()
            clean = re.sub(r'[^a-z0-9]+', '_', clean)
            return clean
        except Exception as e:
            logger.error(f"generate_entity_id failed: {e}", exc_info=True)
            # leaf helper: re-raise generic Exception
            raise CustomAppException(status_code=500, content=str(e), err_code="GENERATE_ENTITY_ID_FAILED")
    
    async def sanitize_label(self,label: str):
        try:
            # remove spaces/special chars
            label = re.sub(r'[^a-zA-Z0-9_]', '_', label)

            # Neo4j labels cannot start with number
            if label and label[0].isdigit():
                label = f"LABEL_{label}"

            return label
        except Exception as e:
            logger.error(f"sanitize_label failed: {e}", exc_info=True)
            # leaf helper: re-raise generic Exception
            raise CustomAppException(status_code=500, content=str(e), err_code="SANITIZE_LABEL_FAILED")

    async def sanitize_relationship(self,rel: str):
        try:
            rel = rel.upper()

            rel = re.sub(
                r'[^A-Z0-9_]',
                '_',
                rel
            )

            if rel and rel[0].isdigit():
                rel = f"REL_{rel}"

            return rel
        except Exception as e:
            logger.error(f"sanitize_relationship failed: {e}", exc_info=True)
            # leaf helper: re-raise generic Exception
            raise CustomAppException(status_code=500, content=str(e), err_code="SANITIZE_REL_FAILED")

    async def ingest_pdf(self,file_name:str,content:str,embeddings, transformer:LLMGraphTransformer, graph:Neo4jGraph,graph_query_embedding):
        try:
            logger.info(f"Starting ingest_pdf for: {file_name}")

            # -------------------------------
            # 1. Read PDF and add document node
            # -------------------------------
            doc =  fitz.open(stream=content, filetype="pdf")

            full_text = "\n".join([page.get_text() for page in doc])

            logger.info(f"Read PDF pages: {len(doc)}; total_chars={len(full_text)}")

            doc_id = hashlib.md5(
                 file_name.encode()
                ).hexdigest()
            logger.info(f"Created document id: {doc_id}")
            

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
            logger.info(f"Chunked into {len(chunks)} chunks (chunk_size={splitter.chunk_size}, overlap={splitter.chunk_overlap})")


            # -------------------------------
            # 3. Batch Embeddings (FAST)
            # -------------------------------
            logger.info("Computing embeddings for chunks...")
            vectors = embeddings.embed_documents(chunks)
            logger.info(f"Computed embeddings for {len(vectors)} chunks")


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

            logger.info(f"Stored {len(chunk_docs)} chunks into graph for document {doc_id}")


        

            # -------------------------------
            # 5. Smart Chunk Selection
            # -------------------------------



            selected_chunks = await self.select_top_chunks(chunks, vectors, top_k=10,graph_query_embedding=graph_query_embedding)
            logger.info(f"Selected {len(selected_chunks)} top chunks for graph extraction")

        

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
            logger.info(f"Prepared {len(graph_documents_input)} Document objects for graph extraction")

            # -------------------------------
            # 7. Graph Extraction (LLM)
            # -------------------------------
            logger.info("Invoking graph transformer to extract nodes and relationships")
            graph_docs = transformer.convert_to_graph_documents(graph_documents_input)
            logger.info(f"Graph extraction returned {len(graph_docs)} graph documents")

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
                    total_relationships += 1

                total_entities += len(getattr(g_doc, "nodes", []))

            logger.info(f"Inserted entities and relationships into graph")

            # -------------------------------
            # 11. Create Vector Index
            # -------------------------------
            logger.info("Creating vector index (best-effort)")

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

            logger.info("Graph ingestion completed successfully for document %s", doc_id)


            return {
                "status": "success",
                "document": file_name,
                "total_chunks": len(chunks),
                "graph_chunks": len(selected_chunks),
                "graph_documents": len(graph_docs)
            }

        except CustomAppException:
            # propagate domain exceptions coming from child helpers
            raise
        except Exception as e:
            logger.error(f"Ingestion failed: {str(e)}", exc_info=True)
            raise CustomAppException(status_code=500, content=str(e), err_code="INGESTION_FAILED")


def get_ingestion_pipeline():
    return IngestionPipeline()
