from src.utils.exceptions.custom_app_exception import CustomAppException
import os
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_aws.chat_models import ChatBedrockConverse
from src.config.config import config
from src.utils.helpers.logger_helper import logger

class ModelConfig:

    def __init__(self):
        self.llm  = None 
        self.embeddings = None
        self._initialized = False

    async def initialize_models(self):
        if not self._initialized:
            self.llm = ChatBedrockConverse(
                    model_id=config.llm_model_id,
                    temperature=config.llm_temperature,
                max_retries=config.max_retries,
                max_tokens=config.max_tokens,
                provider=config.provider
            )
            
            logger.info('LLM Object was created')
            
            self.embeddings = BedrockEmbeddings(
                model_id=config.embedding_model_id,
                model_kwargs={
                "dimensions": config.dimensions,
                "normalize": True
                    }
            )
            self._initialized = True

            logger.info('Created Embeddings LLM Object')

        return self.llm, self.embeddings

models = ModelConfig()