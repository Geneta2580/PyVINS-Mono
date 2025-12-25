import gtsam
import gtsam_unstable
import numpy as np

def test_factor_access_v3():
    graph = gtsam.NonlinearFactorGraph()
    
    # 构造符号和因子
    X1 = gtsam.symbol('x', 1)
    L1 = gtsam.symbol('l', 1)
    K = gtsam.Cal3_S2(500, 500, 0, 320, 240)
    inv_depth_factor3a = gtsam_unstable.InvDepthFactorVariant3a(
    X1, L1, 
    (2, 1), K, gtsam.noiseModel.Isotropic.Sigma(2, 1.0), gtsam.Pose3())
    graph.add(inv_depth_factor3a)

    # 1. 尝试获取图大小 (如果 len() 报错，就用 .size())
    try:
        size = len(graph)
    except TypeError:
        size = graph.size()
    print(f"图大小: {size}")

    for i in range(size):
        factor = graph.at(i)
        if factor is None: continue
            
        # 2. 获取 Keys 并处理
        # factor.keys() 在新版通常返回 list，list 没有 .at()
        keys = factor.keys()
        print(f"Keys 类型: {type(keys)}")
        
        # 使用 Python 最通用的迭代方式
        key_list = []
        for k in keys:
            key_list.append(gtsam.DefaultKeyFormatter(k))
        print(f"索引 {i} 的 Keys: {key_list}")

        # 3. 类型检查
        print(f"类名: {type(factor).__name__}")
        # 推荐使用 isinstance 判定，它比字符串匹配更可靠
        if isinstance(factor, gtsam_unstable.InvDepthFactorVariant3a):
            print("  [确认] 这是一个逆深度因子3a")
        elif isinstance(factor, gtsam_unstable.InvDepthFactorVariant3b):
            print("  [确认] 这是一个逆深度因子3b")
        else:
            print("  [确认] 这是一个其他因子")

if __name__ == "__main__":
    test_factor_access_v3()