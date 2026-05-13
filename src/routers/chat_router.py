from fastapi import APIRouter , UploadFile,File
from fastapi.responses import JSONResponse
from src.utils.exceptions.custom_app_exception import CustomAppException
from src.models.chat_models import QueryRequest
from fastapi import Depends
from src.services.upload_service import UploadService, get_upload_service
from src.services.retrieve_service import RetrieveService, get_retrieve_service
from src.utils.helpers.logger_helper import logger


router = APIRouter(prefix="/api/admin", tags=["Admin Upload"])


@router.post("/upload")
async def upload_file(file: UploadFile = File(...), upload_service: UploadService = Depends(get_upload_service)):
    try:
        logger.info("Router triggered successfully")
        result = await upload_service.upload_file(file)
        return JSONResponse(content={"status": "success", "message": "File uploaded and processed successfully", "data": result.model_dump()})
    except CustomAppException as e:
        raise CustomAppException(status_code=e.status_code, content=e.detail, err_code=e.err_code)
    except Exception as e:
        raise CustomAppException(status_code=500, content=str(e), err_code="FILE_UPLOAD_ERROR")


@router.post("/retrieve")
async def retrieve(request: QueryRequest, retrieve_service: RetrieveService = Depends(get_retrieve_service)):
    try:
        result = await retrieve_service.hybrid_retrieval(request.query)
        return JSONResponse(content={"status": "success", "message": "Retrieved successfully", "data": result.model_dump()})
    except CustomAppException as e:
        raise CustomAppException(status_code=e.status_code, content=e.detail, err_code=e.err_code)
    except Exception as e:
        raise CustomAppException(status_code=500, content=str(e), err_code="RETRIEVE_ERROR")


    

                
