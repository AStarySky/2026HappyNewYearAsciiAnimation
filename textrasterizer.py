import curses
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def text_to_ascii(text, font_path="simsun.ttc", font_size=40):
    # 1. 初始渲染文字到一张大画布上
    # 这里的 255 是白色背景，0 是黑色文字
    font = ImageFont.truetype(font_path, font_size)
    temp_canvas = Image.new('L', (font_size * len(text) * 2, font_size * 2), 255)
    draw = ImageDraw.Draw(temp_canvas)
    draw.text((10, 10), text, font=font, fill=0)

    # 2. 自动寻找文字边缘并裁剪
    # invert 操作是因为 getbbox 会寻找非零区域（黑色在 L 模式下是 0，所以先反转）
    invert_img = Image.eval(temp_canvas, lambda x: 255 - x)
    bbox = invert_img.getbbox()
    
    if not bbox:
        return ""

    # 裁剪出紧贴文字边缘的矩形
    cropped_img = temp_canvas.crop(bbox)
    
    # 3. 横向拉伸 2 倍以适应终端长宽比
    w, h = cropped_img.size
    final_img = cropped_img.resize((w * 2, h), Image.NEAREST)
    
    # 4. 映射字符
    ascii_chars = "@@   " 
    new_w, new_h = final_img.size
    result = []
    
    for y in range(new_h):
        line = ""
        for x in range(new_w):
            gray = final_img.getpixel((x, y))
            char_idx = gray * (len(ascii_chars) - 1) // 255
            line += ascii_chars[char_idx]
        # 使用 rstrip() 确保行尾也没有多余空格，但保留行首用于对齐（如果有的话）
        result.append(line)
    return "\n".join(result)


class RasterizedText:
    """光栅化文字效果类"""
    
    def __init__(self, stdscr, texts, font_path=R"C:\Windows\Fonts\msyh.ttc", size=12):
        self.stdscr = stdscr
        self.height, self.width = stdscr.getmaxyx()
        
        # 字符密度序列（从稀疏到密集）
        self.density_chars = " .:-=+*%#@"
        
        # 定义"新年快乐"的字符画（16x32，每个字约8x8）
        self.text_arts = [text_to_ascii(text, font_path, size) for text in texts]
        self.text_art_index = 0
        
        # 文字位置
        self.text_x = 0
        self.text_y = 0
        self.text_width = 0
        self.text_height = 0

        self.speed = 0.5   # 进度增加速度
        self.is_animating = True
        self.state = "Emerging"
        
        # 每个字符的独立进度（用于更自然的过渡）
        self.char_progress = []
        
    def setup(self):
        """设置文字位置和初始化"""
        lines = self.text_arts[self.text_art_index].split('\n')
        
        self.text_height = len(lines)
        self.text_width = max(len(line) for line in lines) if lines else 0
        
        # 计算居中位置
        self.text_x = max(0, (self.width) // 2 - self.text_width//2)
        self.text_y = 1  # 顶部留一些空间
        
        # 初始化每个字符的进度
        self.char_progress = self.init_char_progress()
    
    def init_char_progress(self):
        char_progress = []
        for i, line in enumerate(self.text_arts[self.text_art_index].split("\n")):
            row_progress = []
            for j, char in enumerate(line):
                if char.strip():  # 如果是非空格字符
                    # 随机初始进度，让效果更自然
                    row_progress.append(20 *(-i - j))
                else:
                    row_progress.append(1 << 64)  # 标记为空格
            char_progress.append(row_progress)
        # print(np.array(char_progress))
        return np.array(char_progress)    
    
    def clear_char_progress(self):
        char_progress = []
        for i, line in enumerate(self.text_arts[self.text_art_index].split("\n")):
            row_progress = []
            for j, char in enumerate(line):
                if char.strip():  # 如果是非空格字符
                    # 随机初始进度，让效果更自然
                    row_progress.append(20 *(i + j))
                else:
                    row_progress.append(-1 << 64)  # 标记为空格
            char_progress.append(row_progress)
        return np.array(char_progress)   
    
    def update(self):
        """更新动画状态"""
        if not self.is_animating:
            return
        
        # 增加整体进度
        
        # 更新每个字符的进度
        if self.state == "Emerging":
            for i, row in enumerate(self.char_progress):
                for j, prog in enumerate(row):
                    if prog <= 1 << 63:  # 非空格
                        # 随机增加进度，但受整体进度限制
                        self.char_progress[i,j] += random.uniform(0.5, 2) * self.speed 
            if (self.char_progress >= 100).all():
                self.state = "Disappearing"
                self.char_progress = self.clear_char_progress()
        else:
            for i, row in enumerate(self.char_progress):
                for j, prog in enumerate(row):
                    if prog >= -1 << 63:  # 非空格
                        # 随机增加进度，但受整体进度限制
                        self.char_progress[i,j] -= random.uniform(0.5, 2) * self.speed 
                        # self.char_progress[i][j] = min(100, self.char_progress[i][j])
            if (self.char_progress <= 0).all():
                self.text_art_index = (self.text_art_index + 1) % len(self.text_arts)
                self.state = "Emerging"
                self.setup()
    
    def get_char_for_progress(self, progress):
        """根据进度获取对应密度的字符"""
        if progress < 0:
            return ' '  # 空格
        
        # 将进度映射到字符索引
        char_index = int(max(min(progress, 100) / 100, 0) * (len(self.density_chars) - 1))
        char_index = min(char_index, len(self.density_chars) - 1)
        
        # 确保至少有基本字符
        if char_index < 0:
            return '·'
        
        return self.density_chars[char_index]
    
    def draw(self):
        """绘制光栅化文字"""
        lines = self.text_arts[self.text_art_index].split('\n')
        self.stdscr.addstr(self.height-2, self.width - len(self.state)  - 13, "Text state: " + self.state)
        for i, line in enumerate(lines):
            y_pos = self.text_y + i
            
            # 绘制每一行
            for j, original_char in enumerate(line):
                x_pos = self.text_x + j
                
                if x_pos >= self.width or y_pos >= self.height:
                    continue
                
                if original_char.strip():  # 原始位置有字符
                    progress = self.char_progress[i][j] if i < len(self.char_progress) and j < len(self.char_progress[i]) else 0
                    char = self.get_char_for_progress(progress)
                    
                    color_idx = 3
                    if char != " ":
                        try:
                            self.stdscr.addstr(y_pos, x_pos, char, curses.color_pair(color_idx) | curses.A_BOLD)
                        except curses.error:
                            pass