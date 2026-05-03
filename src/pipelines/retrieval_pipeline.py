from src.utils.exceptions.custom_app_exception import CustomAppException
from src.config.model_config import models
from src.config.config import config
from src.utils.helpers.logger_helper import logger
from langchain_neo4j import Neo4jGraph


class RetrievalPipeline:
    def __init__(self):
        self.models = models

    async def generate_cypher(self, llm, query: str) -> str:
        try:
            prompt = f"""
                You are a Neo4j expert.

                Generate ONLY a Cypher query for the question below.

                Rules:
                - Do NOT explain.
                - Do NOT add markdown.
                - Use ONLY the given schema.
                - Do NOT invent labels or relationships.

                Schema:
                Nodes:
                - Madras(name)
                - QualityEducation(details)

                Relationships:
                - (QualityEducation)-[:SCHEME]->(Madras)

                Question: {query}
                """
            response = await llm.ainvoke(prompt)
            # Pick first generation text
            response = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        
            return (
                response.replace("```cypher", "")
                    .replace("```", "")
                    .strip()
            )
        except Exception as e:
            logger.info(f"generate_cypher failed: {e}", exc_info=True)
            raise CustomAppException(status_code=500, content=str(e), err_code="CYTHER_GENERATION_FAILED")

    async def run_cypher(self, graph: Neo4jGraph, cypher: str):
        try:
            print(cypher,'**#####')
            return graph.query(cypher)
        except Exception as e:
            logger.info(f"run_cypher failed: {e}", exc_info=True)
            raise CustomAppException(status_code=500, content=str(e), err_code="CYPHER_EXECUTION_FAILED")

    async def vector_search(self, embeddings, graph: Neo4jGraph, query: str, k: int = 5):
        try:
            query_embedding = embeddings.embed_query(query)
            result = graph.query(
                """
CALL db.index.vector.queryNodes('node_vector_index', $k, $embedding)
YIELD node, score
RETURN node.content AS content, score
""",
                {"k": k, "embedding": query_embedding},
            )
            return [r["content"] for r in result]
        except Exception as e:
            logger.info(f"vector_search failed: {e}", exc_info=True)
            raise CustomAppException(status_code=500, content=str(e), err_code="VECTOR_SEARCH_FAILED")

    async def graph_qa(self, llm, graph: Neo4jGraph, query: str):
        try:
            cypher = await self.generate_cypher(llm, query)
            data = await self.run_cypher(graph, cypher)

            # Debug print to inspect retrieved data
            logger.info(f"graph_qa data: {data}")

            final_prompt = f"""
Answer the question using the data below.

Question: {query}

Data:
{data}

Rules:
- Do NOT explain database queries
- Do NOT mention Cypher
- Give a clear, direct answer
- If no data found, say "No relevant information found"

Final Answer:
"""

            # Try async generation, fall back to ainvoke
            try:
                response = await llm.agenerate([final_prompt])
                answer = response.generations[0][0].text.strip()
            except Exception:
                answer = await llm.ainvoke(final_prompt)

            return answer
        except CustomAppException:
            raise
        except Exception as e:
            logger.info(f"graph_qa failed: {e}", exc_info=True)
            raise CustomAppException(status_code=500, content=str(e), err_code="GRAPH_QA_FAILED")

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

            # Try vector search but don't fail if it returns nothing
            try:
                contexts = await self.vector_search(embeddings, graph, query, k=k)
                print(contexts,'contexts')
            except CustomAppException:
                contexts = []

            # Run graph QA (uses generate_cypher + run_cypher)
            graph_answer = await self.graph_qa(llm, graph, query)

            # Build combined final prompt
            final_prompt = f"""
Question: {query}

Context:
{contexts}

Graph reasoning:
{graph_answer}

Provide final answer.
"""

            # Use async generation if available
            try:
                response = await llm.agenerate([final_prompt])
                final_answer = response.generations[0][0].text.strip()
            except Exception:
                # fallback to invoke-style API
                final_answer = await llm.ainvoke(final_prompt)

            return {"contexts": contexts, "graph_answer": graph_answer, "final_answer": final_answer}

        except CustomAppException:
            raise
        except Exception as e:
            logger.info(f"hybrid_retrieval pipeline failed: {e}", exc_info=True)
            raise CustomAppException(status_code=500, content=str(e), err_code="HYBRID_RETRIEVE_FAILED")


def get_retrieval_pipeline():
    return RetrievalPipeline()
