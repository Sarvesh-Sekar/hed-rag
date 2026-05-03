from sample import GRAPH_QUERY
from src.utils.exceptions.custom_app_exception import CustomAppException
from fastapi import UploadFile,Depends
from src.utils.helpers.logger_helper import logger
from src.pipelines.ingestion_pipeline import IngestionPipeline,get_ingestion_pipeline
from src.config.model_config import models
from src.config.config import config
from src.models.chat_models import APIResponse
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph


class UploadService:
    def __init__(self, ingestion_pipeline:IngestionPipeline):
        self.ingestion_pipeline = ingestion_pipeline

    async def upload_file(self,file:UploadFile):
        try:
            logger.info(f"Processing file: {file.filename}")

            content = await file.read()

            logger.info(f"ingesting file: {file.filename}")
            graph = Neo4jGraph(
                url=config.neo4j_url,
                username=config.neo4j_username,
                password=config.neo4j_password
            )
            embeddings, llm = await models.initialize_models()
            transformer = LLMGraphTransformer(llm=llm)
            graph_query_embedding = embeddings.embed_query(config.graph_query)
            result = await self.ingestion_pipeline.ingest_pdf(content=content, embeddings=embeddings, transformer=transformer,graph=graph,graph_query_embedding=graph_query_embedding)


            return APIResponse(status="success", message="File uploaded and processed successfully", data=result)
        
        except CustomAppException as e:
            raise CustomAppException(status_code=e.status_code, content=e.detail, err_code=e.err_code)
        except Exception as e:
            raise CustomAppException(status_code=500, content=str(e), err_code="FILE_UPLOAD_ERROR")
        
def get_upload_service(ingestion_pipeline:IngestionPipeline = Depends(get_ingestion_pipeline)):
    return UploadService(ingestion_pipeline=ingestion_pipeline)