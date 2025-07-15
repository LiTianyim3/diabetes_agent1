import os
from dotenv import load_dotenv
load_dotenv()  # 读取 .env

# OpenAI
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")  # 默认用 gpt-3.5-turbo

# 知识图谱（可选）
KG_URI      = os.getenv("KG_URI")
KG_USER     = os.getenv("KG_USER")
KG_PASSWORD = os.getenv("KG_PASSWORD")
