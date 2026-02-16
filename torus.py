import numpy as np
import time
import curses

def get_camera_euler_angles(t, alpha, omega=1.0):
    """
    计算在大圆轨道运动的摄像机欧拉角
    t: 时间
    alpha: 轨道倾角 (弧度)
    omega: 角速度
    返回: (phi, theta, gamma) 以弧度为单位
    """
    
    # 1. 计算当前在大圆上的相位角
    psi = omega * t
    
    # 2. 定义轨道平面坐标系到世界坐标系的变换矩阵
    # 绕 X 轴旋转 alpha
    R_alpha = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha),  np.cos(alpha)]
    ])
    
    # 3. 计算在轨道平面内的局部坐标基 (Local Frame)
    # 局部 x 轴: 切线方向 (Tangent)
    x_orbit = np.array([-np.sin(psi), np.cos(psi), 0])
    # 局部 z 轴: 指向球心方向 (Look-at center, -Normal)
    z_orbit = np.array([-np.cos(psi), -np.sin(psi), 0])
    # 局部 y 轴: 副法线方向 (Binormal)
    y_orbit = np.cross(z_orbit, x_orbit)
    
    # 4. 将轨道坐标基变换到世界坐标系
    x_world = R_alpha @ x_orbit
    y_world = R_alpha @ y_orbit
    z_world = R_alpha @ z_orbit
    
    # 5. 构建旋转矩阵 R = [x | y | z]
    R = np.stack([x_world, y_world, z_world], axis=1)
    
    # 6. 从旋转矩阵解析欧拉角 (Z-Y-X 顺序: yaw, pitch, roll)
    # 注意：根据不同的库（如 Unity 或 Blender），公式可能略有不同
    phi = np.arctan2(R[1, 0], R[0, 0])  # Yaw (Z)
    theta = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))  # Pitch (Y)
    gamma = np.arctan2(R[2, 1], R[2, 2])  # Roll (X)
    
    return phi, theta, gamma

class TorusRenderer:
    def __init__(self, width, height, tor_R=3.0, tor_r=0.8, camera_distance=8.0, focus_distance=6.0):
        self.width = width
        self.height = height
        self.gradient = " .:-=+*%#@"
        self.grad_len = len(self.gradient)
        self.tor_R = tor_R
        self.tor_r = tor_r
        self.camera_distance = camera_distance
        self.focus_distance = focus_distance

    def solve_quartic_strict(self, ro, rd):
        Ra2 = self.tor_R * self.tor_R
        ra2 = self.tor_r * self.tor_r
        INF = 1e20

        # --- 步骤 1: 基础系数计算 ---
        m = np.sum(ro * ro, axis=-1)
        n = np.sum(ro * rd, axis=-1)
        k = (m + Ra2 - ra2) * 0.5
        
        k3 = n
        k2 = n*n - Ra2*(rd[..., 0]**2 + rd[..., 1]**2) + k
        k1 = n*k - Ra2*(ro[..., 0]*rd[..., 0] + ro[..., 1]*rd[..., 1])
        k0 = k*k - Ra2*(ro[..., 0]**2 + ro[..., 1]**2)

        # --- 步骤 2: po 变换处理 ---
        # 严格复现: if abs(k3*(k3*k3 - k2) + k1) < 0.01
        po_mask = np.abs(k3 * (k3 * k3 - k2) + k1) < 0.01
        
        # 复制系数用于变换
        pk3, pk2, pk1, pk0 = k3.copy(), k2.copy(), k1.copy(), k0.copy()
        
        # 对满足条件的像素进行变换
        pk1[po_mask] = k3[po_mask]
        pk3[po_mask] = k1[po_mask]
        inv_k0 = 1.0 / k0[po_mask]
        pk1[po_mask] *= inv_k0
        pk2[po_mask] *= inv_k0
        pk3[po_mask] *= inv_k0
        pk0[po_mask] = inv_k0 # 注意：原代码此处 k0 变为 1/k0

        # --- 步骤 3: 构造三次方程系数 c0, c1, c2 ---
        c2 = (pk2 * 2.0 - 3.0 * pk3 * pk3) / 3.0
        c1 = pk3 * (pk3 * pk3 - pk2) + pk1
        c0 = (pk3 * (pk3 * (c2*3.0 + 2.0 * pk2) - 8.0 * pk1) + 4.0 * pk0) / 3.0
        c1 *= 2.0

        Q = c2 * c2 + c0
        R = c2**3 - 3.0 * c2 * c0 + c1 * c1
        h = R * R - Q**3

        t_final = np.full(k3.shape, INF)

        # --- 步骤 4: 分支处理 h >= 0 ---
        mask_h_pos = h >= 0
        if np.any(mask_h_pos):
            sub_h = np.sqrt(h[mask_h_pos])
            sub_R = R[mask_h_pos]
            v = np.sign(sub_R + sub_h) * np.abs(sub_R + sub_h)**(1.0/3.0)
            u = np.sign(sub_R - sub_h) * np.abs(sub_R - sub_h)**(1.0/3.0)
            
            sx = (v + u) + 4.0 * c2[mask_h_pos]
            sy = (v - u) * np.sqrt(3.0)
            
            lens = np.sqrt(sx*sx + sy*sy)
            y_val = np.sqrt(0.5 * (lens + sx))
            x_val = np.where(np.abs(y_val) > 1e-10, 0.5 * sy / y_val, 0.0)
            
            denom = x_val*x_val + y_val*y_val
            r_val = np.where(np.abs(denom) > 1e-10, 2.0 * c1[mask_h_pos] / denom, 0.0)
            
            t1 = x_val - r_val - pk3[mask_h_pos]
            t2 = -x_val - r_val - pk3[mask_h_pos]
            
            # po 处理：如果 po_mask 为 True，t = 2.0 / t
            sub_po = po_mask[mask_h_pos]
            t1 = np.where(sub_po, np.where(np.abs(t1) > 1e-10, 2.0 / t1, INF), t1)
            t2 = np.where(sub_po, np.where(np.abs(t2) > 1e-10, 2.0 / t2, INF), t2)
            
            res = np.full(t1.shape, INF)
            res = np.where(t1 > 0.0, np.minimum(res, t1), res)
            res = np.where(t2 > 0.0, np.minimum(res, t2), res)
            t_final[mask_h_pos] = res

        # --- 步骤 5: 分支处理 h < 0 ---
        mask_h_neg = h < 0
        if np.any(mask_h_neg):
            sub_Q = Q[mask_h_neg]
            sub_R = R[mask_h_neg]
            sQ = np.sqrt(sub_Q)
            
            # 取出该分支对应的参数
            sub_c1 = c1[mask_h_neg]
            sub_c2 = c2[mask_h_neg]
            sub_pk3 = pk3[mask_h_neg]
            sub_po = po_mask[mask_h_neg]

            angle = np.arccos(np.clip(-sub_R / (sub_Q * sQ), -1.0, 1.0))
            w = sQ * np.cos(angle / 3.0)
            d2 = -(w + sub_c2)
            
            # 只有 d2 >= 0 才有实根
            valid_d2 = d2 >= 0
            if np.any(valid_d2):
                d1 = np.sqrt(d2)
                denom1 = np.where(np.abs(d1) > 1e-10, sub_c1 / d1, 0.0)
                h1 = np.sqrt(np.maximum(0.0, w - 2.0*sub_c2 + denom1))
                h2 = np.sqrt(np.maximum(0.0, w - 2.0*sub_c2 - denom1))
                
                roots = np.stack([
                    -d1 - h1 - sub_pk3,
                    -d1 + h1 - sub_pk3,
                     d1 - h2 - sub_pk3,
                     d1 + h2 - sub_pk3
                ]) # Shape (4, num_neg_pixels)

                # 处理 po 倒数逻辑
                if np.any(sub_po):
                    # 只有 sub_po 为 True 的列需要求倒数
                    mask_roots_po = np.broadcast_to(sub_po, roots.shape)
                    roots = np.where(mask_roots_po, 
                                     np.where(np.abs(roots) > 1e-10, 2.0/roots, INF), 
                                     roots)

                roots[roots <= 0.0] = INF
                t_final[mask_h_neg] = np.min(roots, axis=0)

        return np.where(t_final < INF, t_final, -1.0)

    def get_normal(self, pos):
        """严格复现 tor_normal_fast"""
        Ra2 = self.tor_R**2
        ra2 = self.tor_r**2
        d2 = np.sum(pos*pos, axis=-1, keepdims=True)
        
        nx = pos[..., 0:1] * (d2 - ra2 - Ra2)
        ny = pos[..., 1:2] * (d2 - ra2 - Ra2)
        nz = pos[..., 2:3] * (d2 - ra2 + Ra2) # z分量符号不同
        
        n = np.concatenate([nx, ny, nz], axis=-1)
        norm = np.linalg.norm(n, axis=-1, keepdims=True)
        return np.where(norm > 1e-10, n/norm, np.array([0.0, 0.0, 1.0]))

    def render_frame(self, theta, phi, gamma):
        # 生成射线方向 (RD)
        y, x = np.ogrid[:self.height, :self.width]
        normalize_length = min(self.height, self.width * 0.5)
        ny = (y - self.height/2) / normalize_length * 8.0
        nx = (x - self.width/2) / normalize_length * 4.0
        
        rx = np.full((self.height, self.width), nx)
        ry = np.full((self.height, self.width), ny)
        rz = np.full((self.height, self.width), -self.focus_distance)
        rlen = np.sqrt(rx*rx + ry*ry + rz*rz)
        rd = np.stack([rx/rlen, ry/rlen, rz/rlen], axis=-1)

        # 旋转逻辑 (Roll -> Pitch -> Yaw)
        sθ, cθ = np.sin(theta), np.cos(theta)
        sφ, cφ = np.sin(phi), np.cos(phi)
        sγ, cγ = np.sin(gamma), np.cos(gamma)

        # 应用旋转矩阵
        _rx, _ry = rd[..., 0].copy(), rd[..., 1].copy()
        rd[..., 0] = _rx * cγ - _ry * sγ
        rd[..., 1] = _rx * sγ + _ry * cγ
        _rz, _ry = rd[..., 2].copy(), rd[..., 1].copy()
        rd[..., 2] = _rz * cθ - _ry * sθ
        rd[..., 1] = _rz * sθ + _ry * cθ
        _rx, _rz = rd[..., 0].copy(), rd[..., 2].copy()
        rd[..., 0] = _rx * cφ + _rz * sφ
        rd[..., 2] = -_rx * sφ + _rz * cφ

        # 相机位置
        ro_single = np.array([cθ * sφ, sθ, cθ * cφ]) * self.camera_distance
        ro = np.broadcast_to(ro_single, rd.shape)

        # 核心求交
        t = self.solve_quartic_strict(ro, rd)
        
        # 光照计算 (Lambert + Specular)
        hit_mask = t > 0
        intensity = np.zeros((self.height, self.width))
        
        if np.any(hit_mask):
            hit_pos = ro[hit_mask] + rd[hit_mask] * t[hit_mask][:, np.newaxis]
            nox, noy, noz = self.get_normal(hit_pos).T
            
            lx, ly, lz = -0.577, 0.577, -0.577
            # intensity = max(-(nox * lx + noy * ly + noz * lz), 0.1)
            diffuse = np.maximum(-(nox*lx + noy*ly + noz*lz), 0.1)
            
            # 高光
            rx_hit, ry_hit, rz_hit = rd[hit_mask].T
            hx, hy, hz = -rx_hit - lx, -ry_hit - ly, -rz_hit - lz
            hlen = np.sqrt(hx*hx + hy*hy + hz*hz)
            
            # 只有 hlen > 1e-10 时计算高光
            spec_mask = hlen > 1e-10
            spec = np.zeros_like(diffuse)
            if np.any(spec_mask):
                hx /= hlen; hy /= hlen; hz /= hlen
                spec_val = np.maximum(nox*hx + noy*hy + noz*hz, 0.0)
                spec[spec_mask] = spec_val[spec_mask] ** 16.0 * 0.3
            
            intensity[hit_mask] = np.clip(diffuse + spec, 0.0, 1.0)

        # 字符映射
        char_indices = (intensity * (self.grad_len - 1)).round().astype(int)
        grad_map = np.array(list(self.gradient))
        return ["".join(row) for row in grad_map[char_indices]]


class Torus:
    def __init__(self, stdstr, height, width):
        self.stdstr = stdstr
        self.height = height
        self.width = width
        
        self.torus_renderer = TorusRenderer(self.width, self.height)
        
        self.time = 0
        self.speed = 0.5
        self.alpha = 1.3
        
        self.lines = []
        self.update(0)
        
    def update(self, dt):
        self.time += dt
        
        phi, theta, gamma = get_camera_euler_angles(self.time, self.alpha, self.speed)
        
        self.lines = self.torus_renderer.render_frame(theta, phi, gamma)
        
    def draw(self):
        
        phistr, thetastr, gammastr = tuple(map(lambda x : (("+" if x > 0 else "") + str(round(x, 2)))[::-1].zfill(5)[::-1], get_camera_euler_angles(self.time, self.alpha, self.speed)))
            
        text = f"Camera: θ = {thetastr}, φ = {phistr}, γ = {gammastr}"
        for i in range(1,self.height-1):
            self.stdstr.addstr(i, 0, self.lines[i], curses.color_pair(1))    
        self.stdstr.addstr(self.height-3, self.width - len(text) - 1, text)