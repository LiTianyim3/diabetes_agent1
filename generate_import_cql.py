#!/usr/bin/env python3
# generate_import_cql.py

import json

INPUT_JSON = "sample.json"
OUTPUT_CQL = "import_data.cql"

# 1. 读取 JSON
with open(INPUT_JSON, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. 收集实体和关系
nodes = {}      # id -> (type, name)
relations = []  # list of (from, to, type)
for para in data.get('paragraphs', []):
    for sent in para.get('sentences', []):
        for ent in sent.get('entities', []):
            eid = ent['entity_id']
            if eid not in nodes:
                nodes[eid] = (ent['entity_type'], ent['entity'])
        for rel in sent.get('relations', []):
            relations.append((rel['head_entity_id'],
                              rel['tail_entity_id'],
                              rel['relation_type']))

# 3. 写入 CQL 脚本
with open(OUTPUT_CQL, 'w', encoding='utf-8') as out:
    # 3.1 约束
    out.write("// —— 创建唯一约束 ——\n")
    for _, (typ, _) in nodes.items():
        out.write(f"CREATE CONSTRAINT IF NOT EXISTS ON (n:{typ}) ASSERT n.id IS UNIQUE;\n")
    out.write("\n")

    # 3.2 实体 MERGE
    out.write("// —— MERGE 所有节点 ——\n")
    for eid, (typ, name) in nodes.items():
        # 转义单引号
        safe_name = name.replace("'", "\\'")
        out.write(f"MERGE (n:{typ} {{id: '{eid}', name: '{safe_name}'}});\n")
    out.write("\n")

    # 3.3 关系 MERGE
    out.write("// —— MERGE 所有关系 ——\n")
    for frm, to, rtype in relations:
        out.write(f"""
MATCH (a {{id: '{frm}'}}), (b {{id: '{to}'}})
MERGE (a)-[r:{rtype}]->(b);
""")
print(f"已生成 Cypher 脚本：{OUTPUT_CQL}（含 {len(nodes)} 个节点和 {len(relations)} 条关系）")
