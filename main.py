from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import yaml 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from src.utils.exceptions.custom_app_exception import CustomAppException
from src.config.config import config
from src.routers.chat_router import router

app = FastAPI(
    title="Higher Education Department chatbot",
    description="A chatbot that answers questions about government schemes for higher education.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)




def custom_openapi():
    
    if app.openapi_schema:
        return app.openapi_schema
    
    # Load your YAML file
    with open("openapi.yaml", "r") as f:
        schema = yaml.safe_load(f)
    
    app.openapi_schema = schema
    return app.openapi_schema

# Assign the custom function to the app
app.openapi = custom_openapi


app.include_router(router)

@app.exception_handler(CustomAppException)
async def custom_app_exception_handler(request, exc: CustomAppException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "err_code": exc.err_code,"status_code": exc.status_code}
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host=config.host, port=config.port, reload=True)