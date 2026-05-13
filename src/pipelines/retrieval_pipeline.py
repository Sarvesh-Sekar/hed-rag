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
            """, {
                "k": k,
                "embedding": query_embedding
            })

            print(result,'vector search results')
            return result

        except Exception as e:
            logger.info(f"vector_search failed: {e}", exc_info=True)
            raise CustomAppException(status_code=500, content=str(e), err_code="VECTOR_SEARCH_FAILED")


    async def extract_query_entities(self,llm,query):

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

        entities = [
            e.strip()
            for e in response.split(",")
            if e.strip()
        ]

        return entities


    async def find_related_entities(self, graph, entities):

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
            """, {
                "entity": entity
            })

            collected.extend(result)

        return collected
    
    async def traverse_graph(self,graph,entity_ids, hops=2):

        graph_results = []

        for entity_id in entity_ids:
            
            # hops denotes entity can be from 1 to 2 hops
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
            """, {
                "entity_id": entity_id
            })

            graph_results.extend(result)

            print(graph_results,'graph traversal results')


        return graph_results

    async def format_graph_context(self, graph_results):

        lines = []

        for row in graph_results:

            source = row["source"]
            relation = row["relationship"]
            target = row["target"]

            line = (
                f"{source} "
                f"-[{relation}]-> "
                f"{target}"
            )

            lines.append(line)

        lines = list(set(lines))

        return "\n".join(lines)

    async def entity_chunk_context(self, graph, entity_ids, limit=10):

        all_chunks = []

        for entity_id in entity_ids:

            result = graph.query("""
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)

            WHERE e.entity_id = $entity_id

            RETURN DISTINCT
                c.text AS text
            LIMIT $limit
            """, {
                "entity_id": entity_id,
                "limit": limit
            })

            all_chunks.extend(
                [r["text"] for r in result]
            )

        # remove duplicates
        all_chunks = list(set(all_chunks))
        

        return all_chunks

    async def hybrid_retrieval(self, query: str, k: int = 5):
        try:
            # initialize models
            llm, embeddings = await self.models.initialize_models()

            # prepare graph
            graph = Neo4jGraph(
                url=config.neo4j_url,
                username=config.neo4j_username,
                password=config.neo4j_password,
            )


            vector_results = await self.vector_search(embeddings, graph, query, k=k)

            vector_contexts = [r["text"] for r in vector_results]

            query_entities = await self.extract_query_entities(llm,query)

            graph_entities = await self.find_related_entities(graph,query_entities)

            entity_ids = [e["entity_id"] for e in graph_entities]

            graph_paths = await self.traverse_graph(graph,entity_ids, hops=2)

            graph_context = await self.format_graph_context(graph_paths)

            entity_chunks = await self.entity_chunk_context(graph, entity_ids)

            all_contexts = list(set(vector_contexts + entity_chunks))
            text_context = "\n\n".join(all_contexts[:10])


            raw_prompt = config.final_prompt


            clean_prompt = raw_prompt.encode().decode('unicode_escape')


            safe_graph_context = str(graph_context).replace("{", "{{").replace("}", "}}")
            safe_text_context = str(text_context).replace("{", "{{").replace("}", "}}")


            final_prompt = clean_prompt.format(
                query=query,
                graph_context=safe_graph_context,
                text_context=safe_text_context
            )

            answer = llm.invoke(final_prompt).content
            

            result = {
                "answer": answer,
                "query_entities": query_entities,
                "matched_entities": graph_entities,
                "graph_context": graph_context,
                "vector_chunks": vector_contexts
            }

            return result

        except CustomAppException:
            raise
        except Exception as e:
            logger.info(f"hybrid_retrieval pipeline failed: {e}", exc_info=True)
            raise CustomAppException(status_code=500, content=str(e), err_code="HYBRID_RETRIEVE_FAILED")


def get_retrieval_pipeline():
    return RetrievalPipeline()
