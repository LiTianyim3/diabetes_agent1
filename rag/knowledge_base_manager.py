"""
知识库管理工具
"""
import os
import shutil
from typing import List, Dict, Any
from rag.index_diabetes1 import get_retrieve_model
from config.config import Config


class KnowledgeBaseManager:
    """知识库管理器"""
    
    def __init__(self):
        self.config = Config.get_instance()
        self.retrieve_model = get_retrieve_model()
        self.knowledge_base_path = self.config.get_with_nested_params("Knowledge-base-path")
        self.vector_store_path = self.config.get_with_nested_params("model", "embedding", "vector_store_path")
        
        # 确保目录存在
        os.makedirs(self.knowledge_base_path, exist_ok=True)
        os.makedirs(self.vector_store_path, exist_ok=True)
    
    def add_documents(self, file_paths: List[str]) -> bool:
        """添加文档到知识库"""
        try:
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    print(f"文件不存在: {file_path}")
                    continue
                
                # 复制文件到知识库目录
                filename = os.path.basename(file_path)
                dest_path = os.path.join(self.knowledge_base_path, filename)
                
                # 如果文件已存在，添加编号
                counter = 1
                base_name, ext = os.path.splitext(filename)
                while os.path.exists(dest_path):
                    new_filename = f"{base_name}_{counter}{ext}"
                    dest_path = os.path.join(self.knowledge_base_path, new_filename)
                    counter += 1
                
                shutil.copy2(file_path, dest_path)
                print(f"文档已添加: {dest_path}")
            
            # 重新构建向量库
            print("重新构建向量库...")
            self.retrieve_model.build()
            return True
            
        except Exception as e:
            print(f"添加文档失败: {e}")
            return False
    
    def remove_documents(self, filenames: List[str]) -> bool:
        """从知识库中移除文档"""
        try:
            for filename in filenames:
                file_path = os.path.join(self.knowledge_base_path, filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"文档已移除: {filename}")
                else:
                    print(f"文档不存在: {filename}")
            
            # 重新构建向量库
            print("重新构建向量库...")
            self.retrieve_model.build()
            return True
            
        except Exception as e:
            print(f"移除文档失败: {e}")
            return False
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """列出知识库中的所有文档"""
        documents = []
        
        try:
            for filename in os.listdir(self.knowledge_base_path):
                file_path = os.path.join(self.knowledge_base_path, filename)
                if os.path.isfile(file_path):
                    stat = os.stat(file_path)
                    documents.append({
                        "filename": filename,
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                        "path": file_path
                    })
            
            return documents
            
        except Exception as e:
            print(f"列出文档失败: {e}")
            return []
    
    def rebuild_vector_store(self) -> bool:
        """重新构建向量库"""
        try:
            print("开始重新构建向量库...")
            self.retrieve_model.build()
            return True
        except Exception as e:
            print(f"重新构建向量库失败: {e}")
            return False
    
    def clear_vector_store(self) -> bool:
        """清空向量库"""
        try:
            if os.path.exists(self.vector_store_path):
                shutil.rmtree(self.vector_store_path)
                os.makedirs(self.vector_store_path, exist_ok=True)
                print("向量库已清空")
                return True
            return False
        except Exception as e:
            print(f"清空向量库失败: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        try:
            documents = self.list_documents()
            total_size = sum(doc["size"] for doc in documents)
            
            # 按文件类型统计
            file_types = {}
            for doc in documents:
                ext = os.path.splitext(doc["filename"])[1].lower()
                file_types[ext] = file_types.get(ext, 0) + 1
            
            # 检查向量库状态
            vector_store_exists = os.path.exists(os.path.join(self.vector_store_path, "index.faiss"))
            
            return {
                "total_documents": len(documents),
                "total_size": total_size,
                "file_types": file_types,
                "vector_store_exists": vector_store_exists,
                "model_status": self.retrieve_model.model_status.value,
                "knowledge_base_path": self.knowledge_base_path,
                "vector_store_path": self.vector_store_path
            }
            
        except Exception as e:
            print(f"获取统计信息失败: {e}")
            return {}
    
    def search_documents(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """搜索文档"""
        try:
            docs = self.retrieve_model.search(query, k)
            results = []
            
            for doc in docs:
                results.append({
                    "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "metadata": doc.metadata,
                    "source": doc.metadata.get("source", "unknown")
                })
            
            return results
            
        except Exception as e:
            print(f"搜索文档失败: {e}")
            return []
    
    def validate_knowledge_base(self) -> Dict[str, Any]:
        """验证知识库完整性"""
        issues = []
        
        # 检查目录存在
        if not os.path.exists(self.knowledge_base_path):
            issues.append("知识库目录不存在")
        
        if not os.path.exists(self.vector_store_path):
            issues.append("向量库目录不存在")
        
        # 检查向量库文件
        index_file = os.path.join(self.vector_store_path, "index.faiss")
        pkl_file = os.path.join(self.vector_store_path, "index.pkl")
        
        if not os.path.exists(index_file):
            issues.append("向量库索引文件不存在")
        
        if not os.path.exists(pkl_file):
            issues.append("向量库配置文件不存在")
        
        # 检查模型状态
        if not self.retrieve_model.is_ready():
            issues.append("检索模型未就绪")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "recommendations": self._get_recommendations(issues)
        }
    
    def _get_recommendations(self, issues: List[str]) -> List[str]:
        """根据问题获取建议"""
        recommendations = []
        
        if "知识库目录不存在" in issues:
            recommendations.append("请创建知识库目录并添加文档")
        
        if "向量库索引文件不存在" in issues or "向量库配置文件不存在" in issues:
            recommendations.append("请运行 rebuild_vector_store() 重建向量库")
        
        if "检索模型未就绪" in issues:
            recommendations.append("请检查模型配置并重新初始化")
        
        return recommendations


# 全局实例
_kb_manager = None

def get_knowledge_base_manager() -> KnowledgeBaseManager:
    """获取知识库管理器实例"""
    global _kb_manager
    if _kb_manager is None:
        _kb_manager = KnowledgeBaseManager()
    return _kb_manager
