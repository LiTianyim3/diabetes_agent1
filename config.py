import os
from dotenv import load_dotenv, find_dotenv
# 强制查找并加载 .env 文件，兼容各种启动路径
if not load_dotenv(find_dotenv(), override=True):
    print("[警告] 未找到 .env 文件，ZHIPU_API_KEY 可能无法加载！")

# OpenAI
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")  # 默认用 gpt-3.5-turbo

# 知识图谱（可选）
KG_URI      = os.getenv("KG_URI")
KG_USER     = os.getenv("KG_USER")
KG_PASSWORD = os.getenv("KG_PASSWORD")

# 智普AI
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
ZHIPU_API_BASE = os.getenv("ZHIPU_API_BASE", "https://open.bigmodel.cn/api/paas/v4")

class Config:
    """配置管理类"""
    
    _instance = None
    _config = {
        "model": {
            "graph-entity": {
                "search-key": "名称"
            },
            "embedding": {
                "model-path": "C:/Users/15717/.cache/modelscope/hub/",
                "model-name": "iic/nlp_corom_sentence-embedding_chinese-base",
                "model-version": "v1.1.0",
                "device": "cuda:0",
                "vector_store_path": "./data/vector_store"
            }
        },
        "Knowledge-base-path": "./data/knowledge_base",
        "zhipu": {
            "api_key": ZHIPU_API_KEY,
            "base_url": ZHIPU_API_BASE,
            "model": "glm-4-flash"
        }
    }
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get_with_nested_params(self, *keys):
        """获取嵌套配置参数"""
        result = self._config
        for key in keys:
            if isinstance(result, dict) and key in result:
                result = result[key]
            else:
                return None
        return result
    
    def get(self, key, default=None):
        """获取配置项"""
        return self._config.get(key, default)
    
    def set(self, key, value):
        """设置配置项"""
        self._config[key] = value
