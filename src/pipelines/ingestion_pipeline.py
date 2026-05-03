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
    def __init__(self, data_source):
        self.llm = models.llm
        self.embeddings = models.embeddings
        
    async def semantic_score(chunk_embedding,graph_query_embedding):
        return cosine_similarity(
        [chunk_embedding],
        [graph_query_embedding]
        )[0][0]

    async def structure_score(text):
        score = 0
        text_lower = text.lower()

        if any(w in text_lower for w in ["if", "must", "should", "eligible"]):
            score += 2

        if re.search(r'(\d+\.)|(- )|(\•)', text):
            score += 2

        if re.search(r'\d+', text):
            score += 1

        return score

    async def keyword_score(text):
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


    async def select_top_chunks(self,chunks, embeddings_list, top_k=10,graph_query_embedding=None):
        scored = []

        for i, chunk in enumerate(chunks):
            score = await self.score_chunk(chunk, embeddings_list[i],graph_query_embedding=graph_query_embedding)
            scored.append((chunk, embeddings_list[i], score))

        scored.sort(key=lambda x: x[2], reverse=True)

        selected = scored[:top_k]

        return selected

    async def ingest_pdf(self,content:str,embeddings, transformer:LLMGraphTransformer, graph:Neo4jGraph,graph_query_embedding):
        try:

            # -------------------------------
            # 1. Read PDF
            # -------------------------------
            doc =  await fitz.open(stream=content, filetype="pdf")

            full_text = "\n".join([page.get_text() for page in doc])

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
            embeddings_list = await embeddings.embed_documents(chunks)

            # -------------------------------
            # 4. Smart Chunk Selection
            # -------------------------------
            selected_chunks = await self.select_top_chunks(chunks, embeddings_list, top_k=10,graph_query_embedding=graph_query_embedding)

            selected_docs = [
                Document(page_content=chunk)
            for chunk, _, _ in selected_chunks
            ]

            logger.info(f"Selected {len(selected_docs)} chunks for graph")

            # -------------------------------
            # 5. Graph Extraction (LLM)
            # -------------------------------
            graph_docs = await transformer.convert_to_graph_documents(selected_docs)

            # -------------------------------
            # 6. Attach embeddings efficiently
            # -------------------------------
            chunk_to_embedding = {
            chunk: emb for chunk, emb, _ in selected_chunks
            }

            for g_doc in graph_docs:
                source_text = g_doc.source.page_content
                vector = chunk_to_embedding[source_text]

                for node in g_doc.nodes:
                    node.properties["content"] = source_text
                    node.properties["embedding"] = vector

            # -------------------------------
            # 7. Store in Neo4j (single write)
            # -------------------------------
            await graph.add_graph_documents(graph_docs)

            logger.info("Graph ingestion completed successfully")

            return {
            "status": "success",
            "total_chunks": len(chunks),
            "graph_chunks": len(selected_docs)
            }
        except Exception as e:
            logger.error(f"Ingestion failed: {str(e)}", exc_info=True)
            raise CustomAppException(status_code=500, content=str(e), err_code="INGESTION_FAILED")


def get_ingestion_pipeline():
    return IngestionPipeline()