from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    API_KEY: str
    DATABASE_URL: str
    DATA_SOURCE_URL: str  # or path to a CSV

    class Config:
        env_file = ".env"


def get_settings():
    return Settings()
