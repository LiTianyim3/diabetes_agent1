"""
智普AI知识库测试脚本
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.zhipu_knowledge_manager import get_zhipu_knowledge_manager
from config.config import Config
import time


def test_zhipu_knowledge_base():
    """测试智普AI知识库功能"""
    print("=" * 60)
    print("智普AI知识库测试")
    print("=" * 60)
    
    # 获取知识库管理器
    print("\n1. 初始化知识库管理器...")
    try:
        kb_manager = get_zhipu_knowledge_manager()
        print("✓ 知识库管理器初始化成功")
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        return False
    
    # 创建知识库
    print("\n2. 创建知识库...")
    knowledge_id = kb_manager.create_knowledge_base(
        name="糖尿病智能问答知识库",
        description="包含糖尿病相关知识、意图检测等内容的综合知识库"
    )
    
    if not knowledge_id:
        print("✗ 知识库创建失败")
        return False
    
    print(f"✓ 知识库创建成功，ID: {knowledge_id}")
    
    # 上传知识库文件
    print("\n3. 上传知识库文件...")
    config = Config.get_instance()
    knowledge_base_path = config.get_with_nested_params("Knowledge-base-path")
    
    # 上传目录中的所有文件
    upload_results = kb_manager.upload_directory_to_knowledge_base(
        knowledge_id, 
        knowledge_base_path
    )
    
    print(f"上传结果:")
    for filename, success in upload_results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {filename}")
    
    # 等待文件处理完成
    print("\n4. 等待文件处理...")
    time.sleep(5)  # 等待5秒让文件处理完成
    
    # 获取知识库信息
    print("\n5. 获取知识库信息...")
    kb_info = kb_manager.get_knowledge_base_info(knowledge_id)
    if kb_info:
        print(f"知识库名称: {kb_info['name']}")
        print(f"创建时间: {kb_info['created_at']}")
        print(f"文件数量: {len(kb_info['files'])}")
        print("文件列表:")
        for filename, file_info in kb_info['files'].items():
            print(f"  - {filename} (大小: {file_info['size']} bytes)")
    
    # 获取统计信息
    print("\n6. 获取统计信息...")
    stats = kb_manager.get_statistics()
    print(f"总知识库数量: {stats['total_knowledge_bases']}")
    print(f"总文件数量: {stats['total_files']}")
    print(f"总大小: {stats['total_size_mb']:.2f} MB")
    
    # 测试对话功能
    print("\n7. 测试知识库对话...")
    test_questions = [
        "糖尿病的症状有哪些？",
        "空腹血糖≥7.0mmol/L（空腹指至少 8 小时未进食）；随机血糖≥11.1mmol/L，且伴有糖尿病典型症状（如多饮、多尿、多食、体重下降等）；口服葡萄糖耐量试验（OGTT）中，餐后 2 小时血糖≥11.1mmol/L。糖化血红蛋白（HbA1c）≥6.5%是糖尿病吗，是什么类型的糖尿病？"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n问题 {i}: {question}")
        print("-" * 40)
        
        try:
            # 使用流式输出
            answer = kb_manager.chat_with_knowledge_base(
                knowledge_id=knowledge_id,
                question=question,
                stream=True
            )
            print(f"\n回答长度: {len(answer)} 字符")
            
        except Exception as e:
            print(f"✗ 对话失败: {e}")
        
        print("-" * 40)
    
    # 列出所有知识库
    print("\n8. 列出所有知识库...")
    knowledge_list = kb_manager.get_knowledge_base_list()
    for kb in knowledge_list:
        print(f"- {kb['name']} (ID: {kb['id']}, 文件数: {kb['files_count']})")
    
    print("\n" + "=" * 60)
    print("智普AI知识库测试完成")
    print("=" * 60)
    
    return True


def test_single_question():
    """测试单个问题"""
    print("=" * 60)
    print("单个问题测试")
    print("=" * 60)
    
    # 这里需要手动输入你的知识库ID
    knowledge_id = input("请输入知识库ID: ").strip()
    
    if not knowledge_id:
        print("✗ 知识库ID不能为空")
        return
    
    kb_manager = get_zhipu_knowledge_manager()
    
    while True:
        question = input("\n请输入问题 (输入 'quit' 退出): ").strip()
        if question.lower() == 'quit':
            break
        
        if question:
            print(f"\n问题: {question}")
            print("-" * 40)
            
            try:
                answer = kb_manager.chat_with_knowledge_base(
                    knowledge_id=knowledge_id,
                    question=question,
                    stream=True
                )
                print(f"\n回答长度: {len(answer)} 字符")
                
            except Exception as e:
                print(f"✗ 对话失败: {e}")
            
            print("-" * 40)


def main():
    """主函数"""
    print("智普AI知识库测试工具")
    print("1. 完整测试 (创建知识库、上传文件、测试对话)")
    print("2. 单个问题测试 (需要已有知识库ID)")
    
    choice = input("请选择测试模式 (1/2): ").strip()
    
    if choice == "1":
        success = test_zhipu_knowledge_base()
        if success:
            print("\n✓ 测试完成！")
        else:
            print("\n✗ 测试失败！")
    elif choice == "2":
        test_single_question()
    else:
        print("✗ 无效选择")


if __name__ == "__main__":
    main()
