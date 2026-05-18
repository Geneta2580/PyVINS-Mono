import gtsam

# 检索所有包含 "Smart" 或 "smart" 的属性和类名
smart_apis = [name for name in dir(gtsam) if 'smart' in name.lower()]

print("找到的 smart 相关 API 数量:", len(smart_apis))
for api in sorted(smart_apis):
    print(f" - {api}")