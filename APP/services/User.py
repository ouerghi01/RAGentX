from pydantic import BaseModel


class User(BaseModel):
    username: str
    email: str | None = None
    his_job: str | None = None
    hashed_password: str