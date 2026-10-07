#!/usr/bin/env python3
import subprocess
import sys
import re

def run_evo_rpe(gt_file, est_file, pose_relation, delta):
    """
    调用 evo_rpe 计算特定距离下的误差，并提取 mean 值
    """
    cmd = [
        "evo_rpe", "tum", gt_file, est_file,
        "-r", pose_relation,
        "-d", str(delta),
        "-u", "m",
        "--t_max_diff", "0.05",
        "--all_pairs",
        "--silent" # 隐藏画图等输出，保持干净
    ]
    
    try:
        # 执行命令并捕获标准输出
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
        
        # 使用正则表达式精确提取 mean 的数值
        match = re.search(r"mean\s+([0-9.]+)", output)
        if match:
            return float(match.group(1))
        else:
            return None
    except subprocess.CalledProcessError:
        # 如果距离超出轨迹总长度，evo 会报错，这里捕获异常并跳过
        return None

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python3 {sys.argv[0]} <ground_truth.txt> <estimated.txt>")
        sys.exit(1)
        
    gt_file = sys.argv[1]
    est_file = sys.argv[2]
    
    # KITTI官方评估标准设定的距离区间
    distances = [100, 200, 300, 400, 500, 600, 700, 800]
    
    total_trans_err = 0.0
    total_rot_err = 0.0
    valid_counts = 0
    
    print("-" * 65)
    print(" KITTI Metrics Evaluation for TUM Format using evo")
    print("-" * 65)
    print(f"{'Dist (m)':<10} | {'Trans Error (%)':<20} | {'Rot Error (deg/m)':<20}")
    print("-" * 65)
    
    for d in distances:
        # 获取平移误差的 mean 值 (evo输出单位为 m)
        trans_mean = run_evo_rpe(gt_file, est_file, "trans_part", d)
        
        # 获取旋转误差的 mean 值 (evo输出单位为 deg)
        rot_mean = run_evo_rpe(gt_file, est_file, "angle_deg", d)
        
        if trans_mean is not None and rot_mean is not None:
            # 核心换算逻辑
            trans_pct = (trans_mean / d) * 100  # 转换为百分比
            rot_deg_m = rot_mean / d            # 转换为 deg/m
            
            print(f"{d:<10} | {trans_pct:<20.4f} | {rot_deg_m:<20.6f}")
            
            total_trans_err += trans_pct
            total_rot_err += rot_deg_m
            valid_counts += 1
        else:
            # 兼容处理：如果轨迹只有 350m 长，400m-800m 的区间会自动显示 N/A
            print(f"{d:<10} | {'N/A (Trajectory short)':<20} | {'N/A':<20}")
            
    print("-" * 65)
    
    # 计算所有有效距离的综合平均值
    if valid_counts > 0:
        avg_trans = total_trans_err / valid_counts
        avg_rot = total_rot_err / valid_counts
        avg_rot_100m = avg_rot * 100 # KITTI 论文中常见的 deg/100m 指标
        
        print("=> OVERALL RESULT (Averaged over valid distances):")
        print(f"   Final Translation Error : {avg_trans:.4f} %")
        print(f"   Final Rotation Error    : {avg_rot:.6f} deg/m ({avg_rot_100m:.4f} deg/100m)")
    else:
        print("=> ERROR: Trajectory is too short to compute any KITTI metrics (needs at least 100m).")
    print("-" * 65)
