import curses
import random
import math

class Firework:
    """烟花类，管理单个烟花的发射和爆炸"""
    
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # 初始位置：底部随机位置
        self.x = random.randint(2, screen_width - 3)
        self.y = screen_height - 1
        
        # 目标高度和当前位置
        self.target_y = random.randint(screen_height // 4, screen_height // 2)
        self.current_y = self.y
        
        # 速度和加速度
        self.speed = random.uniform(0.8, 1.5)
        self.gravity = 0.05
        
        # 颜色 (curses颜色对编号)
        self.color = random.randint(1, 6)
        
        # 状态: 'launching', 'exploding', 'done'
        self.state = 'launching'
        
        # 爆炸粒子
        self.particles = []
        self.explosion_time = 0
        self.explosion_duration = random.randint(20, 35)
        
        # 爆炸形状
        self.explosion_type = random.choice(['sphere', 'star', 'heart', 'random'])
    
    def update(self, dt):
        """更新烟花状态"""
        if self.state == 'launching':
            # 向上移动
            self.current_y -= self.speed
            self.speed -= self.gravity
            
            # 到达目标高度时爆炸
            if self.current_y <= self.target_y or self.speed <= 0:
                self.state = 'exploding'
                self.explosion_time = 0
                self.create_particles()
        
        elif self.state == 'exploding':
            self.explosion_time += dt
            
            # 更新所有粒子
            for particle in self.particles:
                particle['x'] += 2 * particle['vx'] * dt
                particle['y'] += particle['vy'] * dt
                particle['vy'] += 0.05 * dt # 重力
                particle['life'] -= dt
            
            # 移除死亡的粒子
            self.particles = [p for p in self.particles if p['life'] > 0]
            
            # 爆炸结束
            if self.explosion_time > self.explosion_duration or len(self.particles) == 0:
                self.state = 'done'
    
    def create_particles(self):
        """创建爆炸粒子"""
        if self.explosion_type == 'sphere':
            # 球状爆炸
            for angle in range(0, 360, 5):
                rad = math.radians(angle)
                speed = random.uniform(0.5, 1.5)
                self.particles.append({
                    'x': self.x,
                    'y': self.current_y,
                    'vx': math.cos(rad) * speed,
                    'vy': math.sin(rad) * speed,
                    'life': random.randint(15, 30),
                    'char': random.choice(['*', '.', '+', 'o'])
                })
        
        elif self.explosion_type == 'star':
            # 星状爆炸
            for i in range(60):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(0.3, 2.0)
                self.particles.append({
                    'x': self.x,
                    'y': self.current_y,
                    'vx': math.cos(angle) * speed,
                    'vy': math.sin(angle) * speed,
                    'life': random.randint(15, 30),
                    'char': random.choice(['*', '+'])
                })
        
        elif self.explosion_type == 'heart':
            # 心形爆炸
            for t in range(0, 100, 2):
                t = t / 10.0
                x = 32 * (math.sin(t) ** 3)
                y = 13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t)
                # 缩放和随机化
                scale = 0.2
                self.particles.append({
                    'x': self.x + x * scale,
                    'y': self.current_y - y * scale,
                    'vx': random.uniform(-0.2, 0.2),
                    'vy': random.uniform(-0.2, 0.2),
                    'life': random.randint(20, 35),
                    'char': random.choice(['<3', '*', '.'])  # 使用<3代替心形字符
                })
        
        else:  # random
            # 随机爆炸
            for i in range(random.randint(25, 80)):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(0.2, 1.8)
                self.particles.append({
                    'x': self.x,
                    'y': self.current_y,
                    'vx': math.cos(angle) * speed,
                    'vy': math.sin(angle) * speed,
                    'life': random.randint(15, 30),
                    'char': random.choice(['*', '.', '+', 'o', '@', '#'])
                })
    
    def draw(self, stdscr):
        """绘制烟花"""
        if self.state == 'launching':
            # 绘制上升轨迹
            trail_length = min(5, int(self.screen_height - self.current_y))
            for i in range(trail_length):
                if self.current_y + i < self.screen_height:
                    char = '|' if i == 0 else '.'
                    # 安全检查坐标
                    y_pos = int(self.current_y) + i
                    x_pos = int(self.x)
                    if 0 <= y_pos < self.screen_height and 0 <= x_pos < self.screen_width:
                        try:
                            stdscr.addstr(y_pos, x_pos, char, 
                                         curses.color_pair(self.color))
                        except curses.error:
                            pass
            
            # 绘制烟花头
            y_pos = int(self.current_y)
            x_pos = int(self.x)
            if 0 <= y_pos < self.screen_height and 0 <= x_pos < self.screen_width:
                try:
                    stdscr.addstr(y_pos, x_pos, '^', 
                                 curses.color_pair(self.color) | curses.A_BOLD)
                except curses.error:
                    pass
        
        elif self.state == 'exploding':
            # 绘制爆炸粒子
            for particle in self.particles:
                x = int(particle['x'])
                y = int(particle['y'])
                if 0 <= y < self.screen_height and 0 <= x < self.screen_width:
                    # 根据粒子生命值调整亮度
                    attr = curses.color_pair(self.color)
                    if particle['life'] < 5:
                        attr |= curses.A_DIM
                    
                    try:
                        stdscr.addstr(y, x, particle['char'], attr)
                    except curses.error:
                        pass