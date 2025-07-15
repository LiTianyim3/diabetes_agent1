def score_severity(data: dict) -> str:
    hba1c = data.get("hba1c", 0)
    if hba1c < 7.0:
        return "轻度"
    if hba1c <= 9.0:
        return "中度"
    return "重度"
