import re
header_val = "#### Điều 11. Hỗ trợ mặt bằng sản xuất"
dieu_pattern = re.compile(r'(?i)^\s*(?:#+\s*)?(Điều\s+\d+[a-zA-Z]*)(?:\.|:|\s|$)')
match = dieu_pattern.search(header_val)
if match:
    print("MATCH:", match.group(0))
else:
    print("NO MATCH")
    
dieu_pattern2 = re.compile(r'(?i)(Điều\s+\d+[a-zA-Z]*)(?:\.|:|\s|$)')
match2 = dieu_pattern2.search(header_val)
if match2:
    print("MATCH2:", match2.group(1))

