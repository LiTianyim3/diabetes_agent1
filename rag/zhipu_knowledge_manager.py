"""
智普AI知识库管理器
"""
import os
import json
import hashlib
import time
from typing import List, Dict, Any, Optional
from zhipuai import ZhipuAI
from config.config import Config
from datetime import datetime


class ZhipuKnowledgeManager:
    """智普AI知识库管理器"""
    
    def __init__(self):
        self.config = Config.get_instance()
        zhipu_config = self.config.get("zhipu", {})
        self.client = ZhipuAI(api_key=zhipu_config.get("api_key"))
        
        # 本地知识库信息存储
        self.knowledge_info_file = "./data/knowledge_info.json"
        self.knowledge_base_path = self.config.get_with_nested_params("Knowledge-base-path")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(self.knowledge_info_file), exist_ok=True)
        
        # 加载或创建知识库信息
        self.knowledge_info = self._load_knowledge_info()
    
    def _load_knowledge_info(self) -> Dict[str, Any]:
        """加载本地知识库信息"""
        if os.path.exists(self.knowledge_info_file):
            try:
                with open(self.knowledge_info_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载知识库信息失败: {e}")
                return {}
        return {}
    
    def _save_knowledge_info(self):
        """保存本地知识库信息"""
        try:
            with open(self.knowledge_info_file, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_info, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存知识库信息失败: {e}")
    
    def _get_file_hash(self, file_path: str) -> str:
        """计算文件哈希值"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def create_knowledge_base(self, name: str, description: str = "") -> Optional[str]:
        """创建智普AI知识库"""
        try:
            result = self.client.knowledge.create(
                embedding_id=3,  # 使用embedding-3模型
                name=name,
                description=description or f"糖尿病智能问答系统知识库 - {name}"
            )
            
            knowledge_id = result.id
            
            # 保存知识库信息
            self.knowledge_info[knowledge_id] = {
                "name": name,
                "description": description,
                "created_at": datetime.now().isoformat(),
                "files": {},
                "status": "active"
            }
            self._save_knowledge_info()
            
            print(f"✓ 知识库创建成功: {name} (ID: {knowledge_id})")
            return knowledge_id
            
        except Exception as e:
            print(f"✗ 创建知识库失败: {e}")
            return None
    
    def upload_file_to_knowledge_base(self, knowledge_id: str, file_path: str) -> bool:
        """上传文件到智普AI知识库"""
        try:
            if not os.path.exists(file_path):
                print(f"✗ 文件不存在: {file_path}")
                return False
            
            # 检查文件大小 (50MB限制)
            file_size = os.path.getsize(file_path)
            if file_size > 50 * 1024 * 1024:
                print(f"✗ 文件过大: {file_path} ({file_size / 1024 / 1024:.1f}MB > 50MB)")
                return False
            
            # 计算文件哈希
            file_hash = self._get_file_hash(file_path)
            filename = os.path.basename(file_path)
            
            # 检查是否已经上传过相同文件
            if knowledge_id in self.knowledge_info:
                existing_files = self.knowledge_info[knowledge_id].get("files", {})
                for existing_file, info in existing_files.items():
                    if info.get("hash") == file_hash:
                        print(f"✓ 文件已存在，跳过上传: {filename}")
                        return True
            
            # 上传文件
            with open(file_path, "rb") as f:
                resp = self.client.knowledge.document.create(
                    file=f,
                    purpose="retrieval",
                    knowledge_id=knowledge_id
                )
            
            # 调试：打印响应对象信息
            print(f"调试信息 - 响应对象类型: {type(resp)}")
            print(f"调试信息 - 响应对象属性: {dir(resp)}")
            if hasattr(resp, '__dict__'):
                print(f"调试信息 - 响应对象内容: {resp.__dict__}")
            
            # 获取文档ID - 兼容不同的响应格式
            document_id = None
            if hasattr(resp, 'id'):
                document_id = resp.id
            elif hasattr(resp, 'document_id'):
                document_id = resp.document_id
            elif hasattr(resp, 'file_id'):
                document_id = resp.file_id
            elif isinstance(resp, dict):
                document_id = resp.get('id') or resp.get('document_id') or resp.get('file_id')
            else:
                # 如果无法获取ID，生成一个临时ID
                document_id = f"temp_{int(time.time() * 1000)}_{filename}"
                print(f"⚠️  无法获取文档ID，使用临时ID: {document_id}")
            
            if document_id:
                print(f"✓ 获取到文档ID: {document_id}")
            else:
                print("⚠️  未获取到文档ID，但继续处理")
                document_id = f"unknown_{int(time.time() * 1000)}"
            
            # 保存文件信息
            if knowledge_id not in self.knowledge_info:
                self.knowledge_info[knowledge_id] = {"files": {}}
            
            self.knowledge_info[knowledge_id]["files"][filename] = {
                "document_id": document_id,
                "hash": file_hash,
                "size": file_size,
                "uploaded_at": datetime.now().isoformat(),
                "status": "uploaded"
            }
            self._save_knowledge_info()
            
            print(f"✓ 文件上传成功: {filename} (文档ID: {document_id})")
            return True
            
        except Exception as e:
            print(f"✗ 上传文件失败: {file_path} - {e}")
            return False
    
    def upload_directory_to_knowledge_base(self, knowledge_id: str, directory_path: str) -> Dict[str, bool]:
        """上传目录下所有支持的文件到知识库"""
        results = {}
        
        if not os.path.exists(directory_path):
            print(f"✗ 目录不存在: {directory_path}")
            return results
        
        # 支持的文件类型
        supported_extensions = {'.doc', '.docx', '.pdf', '.xlsx', '.txt', '.csv'}
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(filename)
                if ext.lower() in supported_extensions:
                    success = self.upload_file_to_knowledge_base(knowledge_id, file_path)
                    results[filename] = success
                else:
                    print(f"⚠️  跳过不支持的文件类型: {filename}")
                    results[filename] = False
        
        return results
    
    def get_knowledge_base_list(self) -> List[Dict[str, Any]]:
        """获取知识库列表"""
        knowledge_list = []
        for knowledge_id, info in self.knowledge_info.items():
            knowledge_list.append({
                "id": knowledge_id,
                "name": info.get("name", "Unknown"),
                "description": info.get("description", ""),
                "created_at": info.get("created_at", ""),
                "files_count": len(info.get("files", {})),
                "status": info.get("status", "unknown")
            })
        return knowledge_list
    
    def get_knowledge_base_info(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """获取知识库详细信息"""
        return self.knowledge_info.get(knowledge_id)
    
    def chat_with_knowledge_base(self, knowledge_id: str, question: str, 
                                model: str = "glm-4", stream: bool = False) -> str:
        """使用知识库进行对话"""
        try:
            # 自定义提示模板
            prompt_template = """从文档
\"\"\"
{{knowledge}}
\"\"\"
中找问题
\"\"\"
{{question}}
\"\"\"
的答案，找到答案就仅使用文档语句回答问题，找不到答案就用自身知识回答并且告诉用户该信息不是来自文档。
不要复述问题，直接开始回答。"""
            
            messages = [
                {"role": "user", "content": question}
            ]
            
            tools = [
                {
                    "type": "retrieval",
                    "retrieval": {
                        "knowledge_id": knowledge_id,
                        "prompt_template": prompt_template
                    }
                }
            ]
            
            if stream:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    stream=True
                )
                
                full_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        print(content, end="", flush=True)
                print()  # 换行
                return full_response
            else:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools
                )
                
                return response.choices[0].message.content
                
        except Exception as e:
            print(f"✗ 对话失败: {e}")
            return f"对话失败: {str(e)}"
    
    def delete_knowledge_base(self, knowledge_id: str) -> bool:
        """删除知识库"""
        try:
            # 这里需要调用智普AI的删除API（如果有的话）
            # 目前先从本地信息中删除
            if knowledge_id in self.knowledge_info:
                del self.knowledge_info[knowledge_id]
                self._save_knowledge_info()
                print(f"✓ 知识库已从本地删除: {knowledge_id}")
                return True
            else:
                print(f"✗ 知识库不存在: {knowledge_id}")
                return False
                
        except Exception as e:
            print(f"✗ 删除知识库失败: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        total_knowledge_bases = len(self.knowledge_info)
        total_files = sum(len(info.get("files", {})) for info in self.knowledge_info.values())
        total_size = 0
        
        for info in self.knowledge_info.values():
            for file_info in info.get("files", {}).values():
                total_size += file_info.get("size", 0)
        
        return {
            "total_knowledge_bases": total_knowledge_bases,
            "total_files": total_files,
            "total_size": total_size,
            "total_size_mb": total_size / 1024 / 1024
        }


# 全局实例
_zhipu_kb_manager = None

def get_zhipu_knowledge_manager() -> ZhipuKnowledgeManager:
    """获取智普知识库管理器实例"""
    global _zhipu_kb_manager
    if _zhipu_kb_manager is None:
        _zhipu_kb_manager = ZhipuKnowledgeManager()
    return _zhipu_kb_manager
