import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    app_name: str = "Hack Solana"
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"

    class Config:
        env_file = ".env"

settings = Settings()
