from pydantic import BaseModel, Field


class Answer(BaseModel):
    reponse: str = Field(description="The answer to the question")
class Answers(BaseModel):
    answers: list[Answer] = Field(description="List of answers to the questions")