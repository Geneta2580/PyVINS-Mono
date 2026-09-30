import numpy as np
from scipy.spatial.transform import Rotation
import gtsam

def pose_matrix_to_tum_format(pose_matrix):
    """Converts a 4x4 SE(3) pose matrix to a TUM trajectory format string components."""
    t = pose_matrix[:3, 3]
    q = Rotation.from_matrix(pose_matrix[:3, :3]).as_quat() # q is [x, y, z, w]
    return t[0], t[1], t[2], q[0], q[1], q[2], q[3]

def skew_symmetric(v):
    """构建向量的反对称矩阵 (3x3)"""
    return np.array([
        [ 0,    -v[2],  v[1]],
        [ v[2],  0,    -v[0]],
        [-v[1],  v[0],  0   ]
    ])

# 仅用于初始化计算预积分相对零偏的雅可比函数
def calculate_preintegration_and_jacobian(measurements, start_time, initial_bias_gyro):
    if not measurements:
        return None, None, None, None

    # 【修正1】: 明确初始化所有累加变量的数据类型为 float
    delta_R_mat = np.eye(3, dtype=float)
    delta_V_vec = np.zeros(3, dtype=float)
    delta_P_vec = np.zeros(3, dtype=float)
    J_R_bg = np.zeros((3, 3), dtype=float)
    
    # 确保零偏也是正确的类型
    initial_bias_gyro = np.asarray(initial_bias_gyro, dtype=float)
    
    last_ts = start_time
    for ts, data in measurements:
        dt = ts - last_ts
        if dt <= 0:
            last_ts = ts
            continue

        # 【修正2】: 强制将输入数据转换为 float 类型的 Numpy 数组
        accel = np.asarray(data.accel, dtype=float)
        gyro = np.asarray(data.gyro, dtype=float)

        gyro_corrected = gyro - initial_bias_gyro
        delta_R_step = Rotation.from_rotvec(gyro_corrected * dt).as_matrix()

        # 更新雅可比矩阵
        J_R_bg = delta_R_step.T @ J_R_bg - np.eye(3) * dt

        # 更新预积分增量
        accel_body = delta_R_mat @ accel
        
        # 现在所有的运算都在统一的 float64 类型下进行，不会再有 dtype='O' 的问题
        delta_P_vec += delta_V_vec * dt + 0.5 * accel_body * dt**2
        delta_V_vec += accel_body * dt
        delta_R_mat = delta_R_mat @ delta_R_step
        
        last_ts = ts

    return delta_R_mat, delta_V_vec, delta_P_vec, J_R_bg

def caculate_rotation_matrix_from_two_vectors(vec1, vec2):
    # 计算旋转轴（叉积并归一化）
    axis = np.cross(vec1, vec2)
    # 处理 vec1 和 vec2 平行或反平行的情况
    if np.linalg.norm(axis) < 1e-6:

        # 向量平行（旋转0度）或反平行（旋转180度）
        if np.dot(vec1, vec2) > 0:
            # 平行，旋转0度
            R0 = np.eye(3)
        else:
            # 反平行，旋转180度。选择任意一个与 vec1 垂直的轴
            temp_axis = np.cross(vec1, np.array([1, 0, 0]))
            if np.linalg.norm(temp_axis) < 1e-6:
                temp_axis = np.cross(vec1, np.array([0, 1, 0]))
            axis = temp_axis / np.linalg.norm(temp_axis)
            angle = np.pi
    else:
        axis = axis / np.linalg.norm(axis)

        # 计算旋转角（点积）
        dot_product = np.dot(vec1, vec2)

        # 确保点积在 [-1, 1] 范围内以避免浮点误差
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle = np.arccos(dot_product)

    # 构造旋转向量
    rot_vec = axis * angle
    r = Rotation.from_rotvec(rot_vec)
    R0 = r.as_matrix()
    return R0

# # 堆叠雅可比块，计算无landmark的Hessian矩阵和b向量
# def build_structureless_hessian(
#     kf_states,          # dict: { kf_gtsam_id: gtsam.Pose3 (即 T_w_b) }
#     lm_states,          # dict: { lm_gtsam_id: np.array([x, y, z]) (即 P_w) }
#     observations,       # list: [ (kf_gtsam_id, lm_gtsam_id, np.array([u, v])) ... ]
#     K,                  # 相机内参矩阵 3x3
#     T_bc,               # IMU到相机的外参 4x4
#     noise_sigma=1.5):   # 像素噪声 (用于构建信息矩阵 Omega)
#     # 建立 ID 到矩阵索引的映射表 (为了组装大矩阵)
#     kf_ids = sorted(list(kf_states.keys()))
#     lm_ids = sorted(list(lm_states.keys()))
    
#     N = len(kf_ids) # 位姿数量
#     M = len(lm_ids) # 路标数量
    
#     kf_id_to_idx = {kf: i for i, kf in enumerate(kf_ids)}
#     lm_id_to_idx = {lm: i for i, lm in enumerate(lm_ids)}

#     # 初始化全局大矩阵块
#     # 注意：为了极致速度，H_ll 其实不需要建 3M x 3M 的大矩阵，
#     # 只需要存 M 个 3x3 的小矩阵即可，这里为了公式直观先写成大矩阵。
#     H_pp = np.zeros((6 * N, 6 * N))
#     H_pl = np.zeros((6 * N, 3 * M))
#     H_ll = np.zeros((3 * M, 3 * M))
    
#     b_p = np.zeros((6 * N, 1))
#     b_l = np.zeros((3 * M, 1))

#     # 信息矩阵 (测量噪声协方差的逆)
#     Omega = np.eye(2) * (1.0 / (noise_sigma ** 2))

#     # ================= 步骤 1：遍历观测，累加 Hessian 和 梯度 =================
#     for kf_id, lm_id, pt_2d in observations:
#         i = kf_id_to_idx[kf_id]
#         j = lm_id_to_idx[lm_id]
        
#         T_wb = kf_states[kf_id]
#         P_w = lm_states[lm_id]
        
#         # [核心数学]：计算重投影误差 e (2x1)
#         # e = pt_2d - pi(T_wb, T_bc, P_w)
#         # E: 重投影误差对位姿的雅可比(2x6)
#         # F: 重投影误差对路标的雅可比(2x3)
#         e, E, F = compute_reprojection_error_and_jacobians(T_wb, T_bc, P_w, pt_2d, K)
        
#         # 累加到大矩阵中
#         # 这里本质就是一个J.T @ Omega @ J的累加
#         # 注意不要先算大J然后J.T @ J，因为雅可比中包含很多0无关项
#         # E_T * Omega * E
#         H_pp[i*6:(i+1)*6, i*6:(i+1)*6] += E.T @ Omega @ E
        
#         # F_T * Omega * F
#         H_ll[j*3:(j+1)*3, j*3:(j+1)*3] += F.T @ Omega @ F
        
#         # E_T * Omega * F
#         H_pl[i*6:(i+1)*6, j*3:(j+1)*3] += E.T @ Omega @ F
        
#         # 梯度的累加
#         b_p[i*6:(i+1)*6, :] += E.T @ Omega @ e
#         b_l[j*3:(j+1)*3, :] += F.T @ Omega @ e

#     # ================= 步骤 2：执行舒尔补 (Schur Complement) =================
#     # 因为 H_ll 是对角的，我们可以光速求逆
#     H_ll_inv = np.zeros_like(H_ll)
#     for j in range(M):
#         H_ll_block = H_ll[j*3:(j+1)*3, j*3:(j+1)*3]
#         # 加上微小阻尼防止奇异 (Levenberg-Marquardt 思想)
#         H_ll_block += np.eye(3) * 1e-6 
#         H_ll_inv[j*3:(j+1)*3, j*3:(j+1)*3] = np.linalg.inv(H_ll_block)

#     # 降维打击！一行代码完成边缘化
#     H_marg = H_pp - H_pl @ H_ll_inv @ H_pl.T
#     b_marg = b_p - H_pl @ H_ll_inv @ b_l

#     # 返回结果供 GTSAM 使用，同时返回 H_ll_inv、H_pl 和 b_l 用于后续的 retract
#     return H_marg, b_marg, H_ll_inv, H_pl, b_l, kf_ids, lm_ids

# def compute_reprojection_error_and_jacobians(T_wb_gtsam, T_bc_mat, P_w, pt_2d, K):
#     """
#     计算重投影误差 e，以及对位姿的雅可比 E 和对路标的雅可比 F
#     """
#     # 1. 提取或转换矩阵 (4x4)
#     # 兼容传入的是 gtsam.Pose3 或是 numpy array
#     T_wb = T_wb_gtsam.matrix() if hasattr(T_wb_gtsam, 'matrix') else T_wb_gtsam
#     T_bc = T_bc_mat
    
#     # 2. 坐标系转换：世界系 (World) -> 载体系 (Body) -> 相机系 (Camera)
#     # P_b = T_bw * P_w
#     T_bw = np.linalg.inv(T_wb)
#     P_w_homo = np.append(P_w, 1.0)
#     P_b_homo = T_bw @ P_w_homo
#     P_b = P_b_homo[:3]
    
#     # P_c = T_cb * P_b
#     T_cb = np.linalg.inv(T_bc)
#     P_c_homo = T_cb @ P_b_homo
#     P_c = P_c_homo[:3]
    
#     Xc, Yc, Zc = P_c[0], P_c[1], P_c[2]
    
#     # 深度检查，防止点跑到相机后面导致奇异
#     if Zc < 1e-5:
#         # 在实际工程中，此处应抛出异常或返回特殊值标记该点无效
#         Zc = 1e-5 

#     # 3. 相机投影模型
#     fx, fy = K[0, 0], K[1, 1]
#     cx, cy = K[0, 2], K[1, 2]
    
#     u_pred = fx * Xc / Zc + cx
#     v_pred = fy * Yc / Zc + cy
#     pt_2d_pred = np.array([u_pred, v_pred])
    
#     # === 计算误差 (2x1) ===
#     # 定义为：观测值 - 预测值
#     e = (pt_2d - pt_2d_pred).reshape(2, 1)
    
#     # ================= 核心雅可比计算 =================
    
#     # 4. 投影雅可比: 像素坐标对相机系下三维点 P_c 的偏导数 (2x3)
#     # J_proj = d(pt_2d) / d(P_c)
#     J_proj = np.array([
#         [fx / Zc,  0,        -fx * Xc / (Zc ** 2)],
#         [0,        fy / Zc,  -fy * Yc / (Zc ** 2)]
#     ])
    
#     # === 计算雅可比 F (对路标点 P_w 的偏导，2x3) ===
#     # 链式法则：d(e)/d(P_w) = d(e)/d(pt_2d_pred) * d(pt_2d_pred)/d(P_c) * d(P_c)/d(P_w)
#     # d(e)/d(pt_2d_pred) = -I
#     # d(P_c)/d(P_w) = R_cw (相机到世界旋转矩阵的逆)
#     T_cw = T_cb @ T_bw
#     R_cw = T_cw[:3, :3]
    
#     F = - J_proj @ R_cw
    
#     # === 计算雅可比 E (对载体位姿 T_wb 的偏导，2x6) ===
#     # 假设使用右乘扰动模型 (与 GTSAM 默认的 Pose3 retract 一致)：
#     # T_wb_new = T_wb * Exp(delta_xi)
#     # 扰动向量 delta_xi 的顺序为 [旋转 omega (3D), 平移 v (3D)]
    
#     # d(P_b)/d(delta_xi) = [ (P_b)^^, -I_3x3 ]  (这是由李代数扰动模型推导出的标准结论)
#     J_Pb_pose = np.hstack((skew_symmetric(P_b), -np.eye(3))) # 3x6
    
#     # d(P_c)/d(P_b) = R_cb
#     R_cb = T_cb[:3, :3]
    
#     # 链式法则组装：d(e)/d(delta_xi) = - J_proj * R_cb * J_Pb_pose
#     E = - J_proj @ R_cb @ J_Pb_pose
    
#     return e, E, F

def compute_reprojection_error_and_jacobians(T_wb, T_bc, P_w, pt_2d, K, noise_sigma=2.0, huber_k=1.345):
    """
    纯数学解析求解重投影误差及雅可比，完美对齐 GTSAM SE(3) 扰动模型
    """
    # 1. 坐标变换：将世界系点 P_w 变换到相机系 P_c
    T_wc = T_wb.compose(T_bc)
    P_c = T_wc.transformTo(gtsam.Point3(*P_w) if isinstance(P_w, np.ndarray) else P_w)
    Xc, Yc, Zc = P_c[0], P_c[1], P_c[2]
    
    # 💥 手性防火墙：如果点在相机背面或极近，直接拒绝
    if Zc < 0.1:
        return None, None, None, None
        
    # 2. 投影模型：计算预测的像素坐标
    fx, fy = K.fx(), K.fy()
    u = fx * (Xc / Zc) + K.px()
    v = fy * (Yc / Zc) + K.py()
    
    # 误差向量 e = 观测值 - 预测值
    e = np.array([pt_2d[0] - u, pt_2d[1] - v]).reshape(2, 1)
    
    # 3. 鲁棒核 (Huber) 权重计算
    error_norm = np.linalg.norm(e) / noise_sigma
    weight = 1.0
    if error_norm > huber_k:
        weight = huber_k / error_norm # 误差太大时降低权重，抵抗外点
        
    # 4. 雅可比解析求导
    # 4.1 投影方程对相机系坐标 P_c 的雅可比 (2x3)
    H_proj = np.array([
        [fx / Zc, 0.0, -fx * Xc / (Zc * Zc)],
        [0.0, fy / Zc, -fy * Yc / (Zc * Zc)]
    ])
    
    # 4.2 误差对 3D 点 P_w 的雅可比 F (2x3)
    # P_c = R_cw * P_w + t_cw  =>  d(P_c)/d(P_w) = R_cw
    R_cw = T_wc.inverse().rotation().matrix()
    F = - (H_proj @ R_cw) # 负号是因为 e = meas - proj
    
    # 4.3 误差对 位姿 T_wb 的雅可比 E (2x6)
    # 遵循 GTSAM 扰动顺序 [rx, ry, rz, tx, ty, tz]
    P_b = T_wb.transformTo(P_w)
    R_cb = T_bc.inverse().rotation().matrix()
        
    # 根据链式法则：dP_c / d(SE3) = R_cb * [skew(P_b), -I]
    H_pose3 = np.hstack([R_cb @ skew_symmetric(P_b), -R_cb])
    E = - (H_proj @ H_pose3)
    
    return e, E, F, weight

def build_structureless_hessian(
    kf_states, lm_states, observations, K, T_bc, noise_sigma=2.0):
    
    # 1. 预统计观测次数，过滤废点
    lm_obs_count = {}
    for kf_id, lm_id, pt_2d in observations:
        lm_obs_count[lm_id] = lm_obs_count.get(lm_id, 0) + 1
        
    truly_valid_lm_ids = [l for l in lm_states.keys() if lm_obs_count.get(l, 0) >= 2]
    kf_ids = sorted(list(kf_states.keys()))
    
    N = len(kf_ids) 
    M = len(truly_valid_lm_ids)
    
    kf_id_to_idx = {kf: i for i, kf in enumerate(kf_ids)}
    lm_id_to_idx = {lm: i for i, lm in enumerate(truly_valid_lm_ids)}

    H_pp = np.zeros((6 * N, 6 * N))
    H_pl = np.zeros((6 * N, 3 * M))
    H_ll = np.zeros((3 * M, 3 * M))
    b_p = np.zeros((6 * N, 1))
    b_l = np.zeros((3 * M, 1))
    
    total_error = 0.0 # 记录真实的平方误差

    # 基础信息矩阵
    Base_Omega = np.eye(2) * (1.0 / (noise_sigma ** 2))

    # ================= 步骤 1：遍历观测，累加 =================
    for kf_id, lm_id, pt_2d in observations:
        if lm_id not in lm_id_to_idx:
            continue # 跳过次数不足的点
            
        i, j = kf_id_to_idx[kf_id], lm_id_to_idx[lm_id]
        
        # 解析提取雅可比
        res = compute_reprojection_error_and_jacobians(
            kf_states[kf_id], T_bc, lm_states[lm_id], pt_2d, K, noise_sigma)
            
        if res[0] is None:
            continue # 手性异常，安全跳过
            
        e, E, F, weight = res
        
        # 动态应用 Huber 权重
        Omega = weight * Base_Omega
        
        H_pp[i*6:(i+1)*6, i*6:(i+1)*6] += E.T @ Omega @ E
        H_ll[j*3:(j+1)*3, j*3:(j+1)*3] += F.T @ Omega @ F
        H_pl[i*6:(i+1)*6, j*3:(j+1)*3] += E.T @ Omega @ F
        
        b_p[i*6:(i+1)*6, :] += E.T @ Omega @ e
        b_l[j*3:(j+1)*3, :] += F.T @ Omega @ e
        
        # 记录真实的 Huber 误差 (rho)
        if weight == 1.0:
            total_error += 0.5 * (e.T @ Base_Omega @ e)[0, 0]
        else:
            # Huber 核的真实误差值
            huber_k = 1.345
            err_norm = np.linalg.norm(e) / noise_sigma
            total_error += huber_k * err_norm - 0.5 * (huber_k ** 2)

    # ================= 步骤 2：执行舒尔补 =================
    H_ll_inv = np.zeros_like(H_ll)
    for j in range(M):
        H_ll_block = H_ll[j*3:(j+1)*3, j*3:(j+1)*3]
        # LM 微小阻尼防止矩阵崩溃
        H_ll_inv[j*3:(j+1)*3, j*3:(j+1)*3] = np.linalg.inv(H_ll_block + np.eye(3) * 1e-5)

    # 降维！
    H_marg = H_pp - H_pl @ H_ll_inv @ H_pl.T
    b_marg = b_p - H_pl @ H_ll_inv @ b_l
    
    # 强制对称与正定保护
    H_marg = 0.5 * (H_marg + H_marg.T)
    H_marg += np.eye(6 * N) * 1e-6 

    return H_marg, b_marg, H_ll_inv, H_pl, b_l, total_error, truly_valid_lm_ids