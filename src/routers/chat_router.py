from fastapi import FastAPIRouter, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from src.utils.exceptions.custom_app_exception import CustomAppException
from models.chat_models import UploadRequestModel
from fastapi import Depends
from src.services.upload_service import UploadService, get_upload_service

router = FastAPIRouter(prefix="/api/admin/", tags=["Admin Upload"])

@router.post("/upload")
async def upload_file(request: UploadRequestModel, upload_service: UploadService = Depends(get_upload_service)):
    try:
        result = await upload_service.upload_file(request.file)
        return JSONResponse(content={"status": "success", "message": "File uploaded and processed successfully", "data": result.model_dump()})
    except CustomAppException as e:
        raise CustomAppException(status_code=e.status_code, content=e.detail, err_code=e.err_code)
    except Exception as e:
        raise CustomAppException(status_code=500, content=str(e), err_code="FILE_UPLOAD_ERROR")


    

                
