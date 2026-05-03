from src.utils.exceptions.custom_app_exception import CustomAppException
from src.pipelines.retrieval_pipeline import RetrievalPipeline, get_retrieval_pipeline
from src.config.model_config import models
from src.config.config import config
from src.utils.helpers.logger_helper import logger
from langchain_neo4j import Neo4jGraph
from fastapi import Depends
from src.models.chat_models import APIResponse


class RetrieveService:
    def __init__(self, pipeline: RetrievalPipeline):
        self.pipeline = pipeline

    async def hybrid_retrieval(self, query: str):
        try:
            result = await self.pipeline.hybrid_retrieval(query)
            return APIResponse(status="success", message="Retrieved successfully", data=result)

        except CustomAppException as e:
            # propagate known app exceptions
            raise CustomAppException(status_code=e.status_code, content=e.detail, err_code=e.err_code)
        except Exception as e:
            logger.error(f"retrieve_service.hybrid_retrieval failed: {e}", exc_info=True)
            raise CustomAppException(status_code=500, content=str(e), err_code="RETRIEVE_ERROR")


def get_retrieve_service(pipeline: RetrievalPipeline = Depends(get_retrieval_pipeline)):
    return RetrieveService(pipeline=pipeline)
