"""
本地知识库的RAG检索模型类
"""
from model.model_base import Modelbase, ModelStatus
from config.config import Config
from client.zhipu_llm import ZhipuLLM

import os
import json
import shutil
import requests
import torch
from typing import List, Optional, Dict, Any
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    MHTMLLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.embeddings import Embeddings


class ZhipuEmbeddings(Embeddings):
    """智普AI在线嵌入模型"""
    
    def __init__(self, api_key: str, base_url: str = "https://open.bigmodel.cn/api/paas/v4"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = "embedding-3"  # 智普的嵌入模型
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表"""
        embeddings = []
        for text in texts:
            embedding = self.embed_query(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "input": text
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result["data"][0]["embedding"]
        except Exception as e:
            print(f"智普嵌入API调用失败: {e}")
            # 返回默认维度的零向量
            return [0.0] * 1024


class LocalModelScopeEmbeddings(Embeddings):
    """本地ModelScope嵌入模型"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._setup_model()
    
    def _setup_model(self):
        """设置本地模型"""
        try:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
            from modelscope.hub.snapshot_download import snapshot_download
            
            # 检查模型是否存在
            if not os.path.exists(self.model_path):
                print(f"正在下载模型到: {self.model_path}")
                model_dir = snapshot_download(
                    "iic/nlp_corom_sentence-embedding_chinese-base",
                    cache_dir=os.path.dirname(self.model_path),
                )
                print(f"模型下载完成: {model_dir}")
            
            # 初始化pipeline
            self.pipeline = pipeline(
                Tasks.sentence_embedding,
                model=self.model_path,
                device=0 if torch.cuda.is_available() else -1
            )
            print("本地嵌入模型初始化成功")
            
        except Exception as e:
            print(f"本地模型初始化失败: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表"""
        try:
            results = []
            for text in texts:
                result = self.pipeline(text)
                if isinstance(result, dict) and 'text_embedding' in result:
                    results.append(result['text_embedding'].tolist())
                else:
                    results.append(result.tolist())
            return results
        except Exception as e:
            print(f"文档嵌入失败: {e}")
            return [[0.0] * 1024 for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        try:
            result = self.pipeline(text)
            if isinstance(result, dict) and 'text_embedding' in result:
                return result['text_embedding'].tolist()
            else:
                return result.tolist()
        except Exception as e:
            print(f"查询嵌入失败: {e}")
            return [0.0] * 1024


class RetrieveModel(Modelbase):
    """RAG检索模型"""
    
    def __init__(self, use_online_embedding: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.use_online_embedding = use_online_embedding
        self.config = Config.get_instance()
        
        # 获取配置
        self._embedding_model_path = self.config.get_with_nested_params("model", "embedding", "model-path")
        self._embedding_model_name = self.config.get_with_nested_params("model", "embedding", "model-name")
        self._vector_store_path = self.config.get_with_nested_params("model", "embedding", "vector_store_path")
        self._data_path = self.config.get_with_nested_params("Knowledge-base-path")
        
        # 创建必要的目录
        for path in [self._vector_store_path, self._data_path]:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
        
        # 初始化嵌入模型
        self._setup_embedding_model()
        
        # 初始化文本分割器
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=100
        )
        
        # 存储检索器
        self._retriever: Optional[VectorStoreRetriever] = None
        self._user_retrievers: Dict[str, VectorStoreRetriever] = {}
        
        # 尝试加载已存在的向量库
        self._load_existing_vector_store()
    
    def _setup_embedding_model(self):
        """设置嵌入模型"""
        try:
            if self.use_online_embedding:
                # 使用智普在线嵌入
                zhipu_config = self.config.get("zhipu", {})
                api_key = zhipu_config.get("api_key")
                base_url = zhipu_config.get("base_url")
                
                if not api_key:
                    raise ValueError("智普API密钥未配置")
                
                self._embedding = ZhipuEmbeddings(api_key=api_key, base_url=base_url)
                print("使用智普在线嵌入模型")
            else:
                # 使用本地模型
                if not torch.cuda.is_available():
                    raise RuntimeError("本地模型需要GPU支持，但未检测到CUDA设备")
                
                model_path = os.path.join(self._embedding_model_path, self._embedding_model_name)
                self._embedding = LocalModelScopeEmbeddings(model_path)
                print("使用本地ModelScope嵌入模型")
                
            # 测试嵌入模型
            test_embedding = self._embedding.embed_query("测试文本")
            print(f"嵌入模型测试成功，向量维度: {len(test_embedding)}")
            
        except Exception as e:
            print(f"嵌入模型初始化失败: {e}")
            print("尝试使用智普在线嵌入作为备用方案...")
            
            # 备用方案：使用智普在线嵌入
            try:
                zhipu_config = self.config.get("zhipu", {})
                api_key = zhipu_config.get("api_key")
                base_url = zhipu_config.get("base_url")
                
                if api_key:
                    self._embedding = ZhipuEmbeddings(api_key=api_key, base_url=base_url)
                    self.use_online_embedding = True
                    print("已切换到智普在线嵌入模型")
                else:
                    raise ValueError("无法初始化任何嵌入模型")
            except Exception as e2:
                print(f"备用方案也失败: {e2}")
                raise
    
    def _load_existing_vector_store(self):
        """加载已存在的向量库"""
        try:
            index_file = os.path.join(self._vector_store_path, "index.faiss")
            if os.path.exists(index_file):
                print("正在加载已存在的向量库...")
                vectorstore = FAISS.load_local(self._vector_store_path, self._embedding, allow_dangerous_deserialization=True)
                self._retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
                self._model_status = ModelStatus.READY
                print("向量库加载成功")
            else:
                print("未找到已存在的向量库，需要调用build()方法创建")
        except Exception as e:
            print(f"加载向量库失败: {e}")
            print("需要调用build()方法重新创建")
    
    def build(self):
        """构建向量库"""
        print("开始构建向量库...")
        self._model_status = ModelStatus.LOADING
        
        try:
            # 加载所有文档
            docs = self._load_all_documents()
            
            if not docs:
                print("警告: 未找到任何文档")
                self._model_status = ModelStatus.FAILED
                return
            
            print(f"总共找到 {len(docs)} 个文档")
            
            # 分割文档
            splits = self._text_splitter.split_documents(docs)
            print(f"文档分割后共 {len(splits)} 个片段")
            
            # 创建向量库
            print("正在创建向量库...")
            vectorstore = FAISS.from_documents(documents=splits, embedding=self._embedding)
            
            # 保存向量库
            vectorstore.save_local(self._vector_store_path)
            print(f"向量库已保存到: {self._vector_store_path}")
            
            # 创建检索器
            self._retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
            self._model_status = ModelStatus.READY
            print("向量库构建完成")
            
        except Exception as e:
            print(f"构建向量库失败: {e}")
            self._model_status = ModelStatus.FAILED
            raise
    
    def _load_all_documents(self) -> List[Document]:
        """加载所有文档"""
        docs = []
        
        # 定义文档加载器
        loaders = [
            (DirectoryLoader, "**/*.pdf", PyPDFLoader, {}),
            (DirectoryLoader, "**/*.docx", UnstructuredWordDocumentLoader, {}),
            (DirectoryLoader, "**/*.txt", TextLoader, {"autodetect_encoding": True}),
            (DirectoryLoader, "**/*.csv", CSVLoader, {"autodetect_encoding": True}),
            (DirectoryLoader, "**/*.html", UnstructuredHTMLLoader, {}),
            (DirectoryLoader, "**/*.mhtml", MHTMLLoader, {}),
            (DirectoryLoader, "**/*.md", UnstructuredMarkdownLoader, {}),
        ]
        
        for loader_class, glob_pattern, file_loader_class, loader_kwargs in loaders:
            try:
                loader = loader_class(
                    self._data_path,
                    glob=glob_pattern,
                    loader_cls=file_loader_class,
                    silent_errors=True,
                    loader_kwargs=loader_kwargs,
                    use_multithreading=True,
                )
                file_docs = loader.load()
                docs.extend(file_docs)
                print(f"已加载 {len(file_docs)} 个 {glob_pattern} 文件")
            except Exception as e:
                print(f"加载 {glob_pattern} 文件时出错: {e}")
        
        return docs
    
    @property
    def retriever(self) -> Optional[VectorStoreRetriever]:
        """获取检索器"""
        if self._model_status == ModelStatus.FAILED:
            print("模型状态为失败，尝试重新构建...")
            self.build()
        return self._retriever
    
    def search(self, query: str, k: int = 6) -> List[Document]:
        """检索相关文档"""
        if not self.retriever:
            print("检索器未初始化")
            return []
        
        try:
            docs = self.retriever.get_relevant_documents(query)
            return docs[:k]
        except Exception as e:
            print(f"检索失败: {e}")
            return []
    
    def build_user_vector_store(self, user_id: str):
        """为用户构建专属向量库"""
        user_data_path = os.path.join("user_data", user_id)
        if not os.path.exists(user_data_path):
            print(f"用户文件夹 {user_data_path} 不存在")
            return
        
        try:
            # 清理旧的向量库
            if user_id in self._user_retrievers:
                del self._user_retrievers[user_id]
                print(f"用户 {user_id} 的旧向量库已删除")
            
            # 加载用户文档
            docs = self._load_user_documents(user_data_path)
            
            if not docs:
                print(f"用户 {user_id} 文件夹中没有找到文档")
                return
            
            # 分割文档
            splits = self._text_splitter.split_documents(docs)
            
            # 创建用户向量库
            vectorstore = FAISS.from_documents(documents=splits, embedding=self._embedding)
            user_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
            
            # 存储用户检索器
            self._user_retrievers[user_id] = user_retriever
            print(f"用户 {user_id} 的向量库已构建完成")
            
        except Exception as e:
            print(f"构建用户 {user_id} 向量库时出错: {e}")
    
    def _load_user_documents(self, user_data_path: str) -> List[Document]:
        """加载用户文档"""
        docs = []
        
        # 定义文档加载器
        loaders = [
            (DirectoryLoader, "**/*.pdf", PyPDFLoader, {}),
            (DirectoryLoader, "**/*.docx", UnstructuredWordDocumentLoader, {}),
            (DirectoryLoader, "**/*.txt", TextLoader, {"autodetect_encoding": True}),
            (DirectoryLoader, "**/*.csv", CSVLoader, {"autodetect_encoding": True}),
            (DirectoryLoader, "**/*.html", UnstructuredHTMLLoader, {}),
            (DirectoryLoader, "**/*.mhtml", MHTMLLoader, {}),
            (DirectoryLoader, "**/*.md", UnstructuredMarkdownLoader, {}),
        ]
        
        for loader_class, glob_pattern, file_loader_class, loader_kwargs in loaders:
            try:
                loader = loader_class(
                    user_data_path,
                    glob=glob_pattern,
                    loader_cls=file_loader_class,
                    silent_errors=True,
                    loader_kwargs=loader_kwargs,
                    use_multithreading=True,
                )
                file_docs = loader.load()
                docs.extend(file_docs)
            except Exception as e:
                print(f"加载用户 {glob_pattern} 文件时出错: {e}")
        
        return docs
    
    def get_user_retriever(self, user_id: str) -> Optional[VectorStoreRetriever]:
        """获取用户专属检索器"""
        return self._user_retrievers.get(user_id)
    
    def upload_user_file(self, user_id: str, file_path: str, file_content: bytes):
        """上传用户文件"""
        user_data_path = os.path.join("user_data", user_id)
        os.makedirs(user_data_path, exist_ok=True)
        
        file_name = os.path.basename(file_path)
        save_path = os.path.join(user_data_path, file_name)
        
        with open(save_path, "wb") as f:
            f.write(file_content)
        
        print(f"文件 {file_name} 已成功上传到用户 {user_id} 的文件夹")
        return save_path
    
    def list_uploaded_files(self, user_id: str) -> List[str]:
        """列出用户上传的文件"""
        user_data_path = os.path.join("user_data", user_id)
        if not os.path.exists(user_data_path):
            return []
        
        files = os.listdir(user_data_path)
        return files
    
    def delete_uploaded_file(self, user_id: str, filename: Optional[str] = None):
        """删除用户上传的文件"""
        user_data_path = os.path.join("user_data", user_id)
        if not os.path.exists(user_data_path):
            print(f"用户文件夹 {user_data_path} 不存在")
            return
        
        if filename:
            file_path = os.path.join(user_data_path, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"文件 {filename} 已成功删除")
            else:
                print(f"文件 {filename} 不存在")
        else:
            # 清空文件夹
            for file in os.listdir(user_data_path):
                file_path = os.path.join(user_data_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"用户 {user_id} 文件夹已清空")
    
    def get_rag_context(self, query: str, user_id: Optional[str] = None) -> str:
        """获取RAG上下文"""
        context_parts = []
        
        # 从全局知识库检索
        global_docs = self.search(query)
        if global_docs:
            context_parts.append("=== 全局知识库 ===")
            for i, doc in enumerate(global_docs, 1):
                context_parts.append(f"{i}. {doc.page_content[:500]}...")
        
        # 从用户专属知识库检索
        if user_id:
            user_retriever = self.get_user_retriever(user_id)
            if user_retriever:
                try:
                    user_docs = user_retriever.get_relevant_documents(query)
                    if user_docs:
                        context_parts.append("\n=== 用户专属知识库 ===")
                        for i, doc in enumerate(user_docs, 1):
                            context_parts.append(f"{i}. {doc.page_content[:500]}...")
                except Exception as e:
                    print(f"检索用户知识库失败: {e}")
        
        return "\n".join(context_parts) if context_parts else "未找到相关信息"


# 全局实例
INSTANCE = None

def get_retrieve_model(use_online_embedding: bool = True) -> RetrieveModel:
    """获取检索模型实例"""
    global INSTANCE
    if INSTANCE is None:
        INSTANCE = RetrieveModel(use_online_embedding=use_online_embedding)
    return INSTANCE