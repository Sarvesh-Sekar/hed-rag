from fastapi import HTTPException

class CustomAppException(HTTPException):
    def __init__(self, status_code: int, content: str,err_code:str):

        self.err_code = err_code
        super().__init__(status_code=status_code, detail=content)

    