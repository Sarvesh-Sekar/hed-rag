import os
import json
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    neo4j_url: str
    neo4j_username: str
    neo4j_password: str
    llm_model_id: str
    embedding_model_id: str
    provider: str
    llm_temperature: float
    graph_query: str
    cypher_prompt:str
    final_prompt:str
    base_keywords: list[str]
    relation_words: list[str]
    max_tokens: int
    max_retries: int
    dimensions: int
    host: str
    port: int

    @classmethod
    def load(cls):
        # Local helper to parse list strings from .env
        def parse_list(env_key):
            val = os.getenv(env_key, "[]")
            try:
                return json.loads(val) # Assumes '["a", "b"]' format
            except json.JSONDecodeError:
                return [i.strip() for i in val.split(",") if i] # Fallback to CSV

        # Return an INSTANCE of the class
        return cls(
            neo4j_url=os.getenv("NEO4J_URL"),
            neo4j_username=os.getenv("NEO4J_USERNAME"),
            neo4j_password=os.getenv("NEO4J_PASSWORD"),
            llm_model_id=os.getenv("LLM_MODEL_ID"),
            embedding_model_id=os.getenv("EMBEDDING_MODEL_ID"),
            provider=os.getenv("PROVIDER"),
            cypher_prompt=os.getenv("CYPHER_PROMPT"),
            final_prompt=os.getenv("FINAL_PROMPT"),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE")),
            graph_query=os.getenv("GRAPH_QUERY", ""),
            base_keywords=parse_list("BASE_KEYWORDS"),
            relation_words=parse_list("RELATION_WORDS"),
            max_tokens=int(os.getenv("MAX_TOKENS")),
            max_retries=int(os.getenv("MAX_RETRIES")),
            dimensions=int(os.getenv("DIMENSIONS")),
            host=os.getenv("HOST"),
            port=int(os.getenv("PORT"))
        )


config = Config.load()
