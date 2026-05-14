from html import entities
from src.utils.exceptions.custom_app_exception import CustomAppException
from src.config.model_config import models
from src.config.config import config
from src.utils.helpers.logger_helper import logger
from langchain_neo4j import Neo4jGraph


class RetrievalPipeline:
    def __init__(self):
        self.models = models

    async def vector_search(self, embeddings, graph: Neo4jGraph, query: str, k: int = 5):
        try:
            logger.info("vector_search start: query=%s k=%d", query, k)
            query_embedding = embeddings.embed_query(query)

            result = graph.query("""
            CALL db.index.vector.queryNodes(
                'chunk_embedding_index',
                $k,
                $embedding
            )

            YIELD node, score

            RETURN
                node.chunk_id AS chunk_id,
                node.text AS text,
                score
            """, {"k": k, "embedding": query_embedding})

            logger.info("vector_search found %d results", len(result) if hasattr(result, '__len__') else 0)
            return result
        except Exception as e:
            logger.error("vector_search failed: %s", str(e), exc_info=True)
            raise CustomAppException(status_code=500, content=str(e), err_code="VECTOR_SEARCH_FAILED")


    async def extract_query_entities(self, llm, query):
        try:
            logger.info("extract_query_entities start: query=%s", query)
            prompt = f"""
        Extract important entities from the query.

        Allowed Entity Types:
        - Scheme
        - Benefit
        - Eligibility
        - Department
        - Institution
        - Course
        - IncomeLimit
        - StudentCategory

        Return ONLY comma separated entity names.

        Query:
        {query}
        """

            response = llm.invoke(prompt).content.strip()
            entities = [e.strip() for e in response.split(",") if e.strip()]
            logger.info("extract_query_entities produced %d entities", len(entities))
            return entities
        except Exception as e:
            logger.error("extract_query_entities failed: %s", str(e), exc_info=True)
            raise CustomAppException(status_code=500, content=str(e), err_code="EXTRACT_QUERY_ENTITIES_FAILED")


    async def find_related_entities(self, graph, entities):
        try:
            logger.info("find_related_entities start: n_entities=%d", len(entities) if entities else 0)
            collected = []
            for entity in entities:
                result = graph.query("""
            MATCH (e:Entity)

            WHERE toLower(e.name)
            CONTAINS toLower($entity)

            RETURN
                e.entity_id AS entity_id,
                e.name AS name,
                labels(e) AS labels
            LIMIT 10
            """, {"entity": entity})
                collected.extend(result)
            logger.info("find_related_entities found %d matches", len(collected))
            return collected
        except Exception as e:
            logger.error("find_related_entities failed: %s", str(e), exc_info=True)
            raise CustomAppException(status_code=500, content=str(e), err_code="FIND_RELATED_ENTITIES_FAILED")

    async def traverse_graph(self, graph, entity_ids, hops=2):
        try:
            logger.info("traverse_graph start: n_entity_ids=%d hops=%d", len(entity_ids) if entity_ids else 0, hops)
            graph_results = []
            for entity_id in entity_ids:
                result = graph.query(f"""
            MATCH (e:Entity {{entity_id:$entity_id}})
            -[r*1..{hops}]-               
            (related)

            UNWIND r as rel

            RETURN DISTINCT
                startNode(rel).name AS source,
                type(rel) AS relationship,
                endNode(rel).name AS target
            LIMIT 50
            """, {"entity_id": entity_id})
                graph_results.extend(result)
            logger.info("traverse_graph returned %d rows", len(graph_results))
            return graph_results
        except Exception as e:
            logger.error("traverse_graph failed: %s", str(e), exc_info=True)
            raise CustomAppException(status_code=500, content=str(e), err_code="TRAVERSE_GRAPH_FAILED")

    async def format_graph_context(self, graph_results):
        try:
            logger.info("format_graph_context start: n_rows=%d", len(graph_results) if graph_results else 0)
            lines = []
            for row in graph_results:
                source = row["source"]
                relation = row["relationship"]
                target = row["target"]
                line = f"{source} -[{relation}]-> {target}"
                lines.append(line)
            lines = list(set(lines))
            logger.info("format_graph_context returning %d unique lines", len(lines))
            return "\n".join(lines)
        except Exception as e:
            logger.error("format_graph_context failed: %s", str(e), exc_info=True)
            raise CustomAppException(status_code=500, content=str(e), err_code="FORMAT_GRAPH_CONTEXT_FAILED")

    async def entity_chunk_context(self, graph, entity_ids, limit=10):
        try:
            logger.info("entity_chunk_context start: n_entity_ids=%d limit=%d", len(entity_ids) if entity_ids else 0, limit)
            all_chunks = []
            for entity_id in entity_ids:
                result = graph.query("""
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)

            WHERE e.entity_id = $entity_id

            RETURN DISTINCT
                c.text AS text
            LIMIT $limit
            """, {"entity_id": entity_id, "limit": limit})
                all_chunks.extend([r["text"] for r in result])
            all_chunks = list(set(all_chunks))
            logger.info("entity_chunk_context returning %d unique chunks", len(all_chunks))
            return all_chunks
        except Exception as e:
            logger.error("entity_chunk_context failed: %s", str(e), exc_info=True)
            raise CustomAppException(status_code=500, content=str(e), err_code="ENTITY_CHUNK_CONTEXT_FAILED")

    async def hybrid_retrieval(self, query: str, k: int = 5):
        try:
            logger.info("hybrid_retrieval start: query=%s k=%d", query, k)

            # initialize models
            llm, embeddings = await self.models.initialize_models()
            logger.info("Models initialized for hybrid retrieval")

            # prepare graph client
            graph = Neo4jGraph(url=config.neo4j_url, username=config.neo4j_username, password=config.neo4j_password)
            logger.info("Neo4j graph client prepared")

            # vector search
            vector_results = await self.vector_search(embeddings, graph, query, k=k)
            vector_contexts = [r.get("text") for r in vector_results] if vector_results else []
            logger.info("Vector search returned %d contexts", len(vector_contexts))

            # extract entities from query
            query_entities = await self.extract_query_entities(llm, query)
            logger.info("Extracted %d entities from query", len(query_entities))

            # find related entities in graph
            graph_entities = await self.find_related_entities(graph, query_entities)
            logger.info("Found %d related entities in graph", len(graph_entities))

            entity_ids = [e["entity_id"] for e in graph_entities]

            # traverse graph
            graph_paths = await self.traverse_graph(graph, entity_ids, hops=2)
            logger.info("Graph traversal returned %d paths", len(graph_paths))

            # format graph context and gather entity chunks
            graph_context = await self.format_graph_context(graph_paths)
            entity_chunks = await self.entity_chunk_context(graph, entity_ids)
            logger.info("Collected %d entity_chunks", len(entity_chunks))

            # combine contexts
            all_contexts = list(set((vector_contexts or []) + (entity_chunks or [])))
            text_context = "\n\n".join(all_contexts[:10])

            # prepare final prompt
            raw_prompt = config.final_prompt
            clean_prompt = raw_prompt.encode().decode("unicode_escape")
            safe_graph_context = str(graph_context).replace("{", "{{").replace("}", "}}")
            safe_text_context = str(text_context).replace("{", "{{").replace("}", "}}")
            final_prompt = clean_prompt.format(query=query, graph_context=safe_graph_context, text_context=safe_text_context)

            logger.info("Invoking LLM for final answer")
            answer = llm.invoke(final_prompt).content

            result = {
                "answer": answer,
                "query_entities": query_entities,
                "matched_entities": graph_entities,
                "graph_context": graph_context,
                "vector_chunks": vector_contexts,
            }

            logger.info("hybrid_retrieval completed successfully")
            return result
        except CustomAppException:
            raise
        except Exception as e:
            logger.error("hybrid_retrieval pipeline failed: %s", str(e), exc_info=True)
            raise CustomAppException(status_code=500, content=str(e), err_code="HYBRID_RETRIEVE_FAILED")


def get_retrieval_pipeline():
    return RetrievalPipeline()
