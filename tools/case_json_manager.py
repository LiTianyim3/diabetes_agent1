# json文件的创建
import os
import json
import datetime

def save_case_json(case_content: str, name: str = None, data_dir: str = None) -> str:
    """
    保存病例为 JSON 文件，内容格式为 {"病例内容": ...}
    - 如果 name 不为空，则以 name.json 命名（已存在则覆盖）
    - 否则以 case_时间戳.json 命名
    返回保存的文件路径
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    if name and name.strip():
        fname = f"{name.strip()}.json"
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"case_{ts}.json"
        fpath = os.path.join(data_dir, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump({"病例内容": case_content}, f, ensure_ascii=False, indent=2)
    return fpath
