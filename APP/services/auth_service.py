
from dotenv import load_dotenv
import os
import jwt
from datetime import datetime, timedelta, timezone
from fastapi import Request, HTTPException
import time
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jwt.exceptions import InvalidTokenError
from passlib.context import CryptContext
from pydantic import BaseModel
from typing import Annotated
from fastapi import Request, Form,Response

load_dotenv()  # take environment variables from .env.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
SECRET_KEY = os.getenv("SEC_API", "default_secret_key_for_development_only")
if SECRET_KEY is None:
    raise ValueError("SEC_API environment variable must be set")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
class Token(BaseModel):
    access_token: str
    token_type: str
    the_user:str
    his_job:str
class TokenData(BaseModel):
    username: str | None = None


class User(BaseModel):
    username: str
    email: str | None = None
    his_job: str | None = None
    hashed_password: str

class AuthService:
     
    def __init__(self,cassandra_intra,agent):
         
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.cassandra_intra=cassandra_intra
        self.agent=agent

        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
    def verify_password(self,plain_password, hashed_password):
        return self.pwd_context.verify(plain_password, hashed_password)


    def get_password_hash(self,password):
        return self.pwd_context.hash(password)
    async def register_user(self, request: Request, username: str = Form(...), email: str = Form(...), password: str = Form(...), his_job: str = Form(...)):
            hashed_password = self.get_password_hash(password)
            user = User(username=username, email=email, his_job=his_job, hashed_password=hashed_password)
            self.cassandra_intra.insert_user(user.username, user.email, user.his_job, user.hashed_password)
            return user
    def get_user(self, username: str):
        user_found=self.cassandra_intra.find_user(username)
        if user_found:
            return User(
                username=user_found.username,
                email=user_found.email,
                his_job=user_found.his_job,
                hashed_password=user_found.password,
            )
        return None
    def authenticate_user(self, username: str, password: str):
        user = self.get_user( username)
        if not user:
            return False
        if not self.verify_password(password, user.hashed_password):
            return False
        return user
    async def login_for_access_token(self,response:Response,
            form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
        ) -> Token:
        user = self.authenticate_user( form_data.username, form_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = self.create_access_token(
            data={"sub": user.username}, expires_delta=access_token_expires
        )
        response.set_cookie(key="auth_token", value=access_token)
        the_user=await self.get_current_user(access_token)
        self.agent.chain=self.agent.retrieval_chain(the_user)
        return Token(access_token=access_token, token_type="bearer", the_user=the_user.username,his_job=the_user.his_job)

    def create_access_token(self,data: dict, expires_delta: timedelta | None = None):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    async def get_current_user(self,token: Annotated[str, Depends(oauth2_scheme)]):
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            if ((payload.get("exp")/60 < time.time()/60) or username is None):
                return None
            
            token_data = TokenData(username=username)
        except InvalidTokenError:
            return None
        user = self.get_user( username=token_data.username)
        if user is None:
            raise credentials_exception
        return user
    async def logout(self,request:Request,response:Response ):
        ttt=request.cookies.get("auth_token")
        if ttt is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        payload = jwt.decode(ttt, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        token_data = TokenData(username=username)
        user = self.get_user( username=token_data.username)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        payload["exp"] = datetime.now(timezone.utc).timestamp() - 1
        response.set_cookie(key="auth_token", value=jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM))
        return  {"detail": "Logged out"}
            

    


   