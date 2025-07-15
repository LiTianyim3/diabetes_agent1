from langchain.llms.base import LLM
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from config import config  # 修改为绝对导入
import requests

load_dotenv()  # 确保.env被加载

class ZhipuLLM(LLM, BaseModel):
    """
    LangChain LLM wrapper for Zhipu API.
    """
    api_key: str      = os.getenv("ZHIPU_API_KEY") or getattr(config, "ZHIPU_API_KEY", None)
    base_url: str     = os.getenv("ZHIPU_API_BASE") or getattr(config, "ZHIPU_API_BASE", "https://open.bigmodel.cn/api/paas/v4")  # 去掉末尾斜杠
    model: str        = os.getenv("MODEL_NAME") or getattr(config, "MODEL_NAME", "glm-4-flash")
    temperature: float = 0.2
    max_tokens: int    = 512

    def _call(self, prompt: str, stop=None, **kwargs) -> str:
        kwargs.pop("functions", None)
        kwargs.pop("tools", None)
        kwargs.pop("stream", None)
        if not self.api_key:
            raise ValueError("未提供api_key，请通过参数或环境变量提供")
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        response = requests.post(url, headers=headers, json=data)
        try:
            response.raise_for_status()
        except Exception as e:
            print("Zhipu API error:", response.status_code, response.text)
            raise
        # 调试输出
        print("Zhipu API resp:", response.json())
        return response.json()["choices"][0]["message"]["content"]

    @property
    def _identifying_params(self):
        return {"model": self.model}

    @property
    def _llm_type(self):
        return "zhipu"
