import curses

def draw_ascii_art(stdscr, art_string, start_y, start_x, color_pair=1):
    """
    在curses窗口中绘制多行ASCII艺术
    
    参数:
        stdscr: curses窗口对象
        art_string: 完整的ASCII艺术字符串（包含换行）
        start_y: 绘制起始行（左上角Y坐标）
        start_x: 绘制起始列（左上角X坐标）
        color_pair: 颜色对编号（默认1=红色）
    """
    # 分割字符串为行列表
    lines = art_string.split('\n')
    
    # 获取屏幕尺寸用于边界检查
    screen_height, screen_width = stdscr.getmaxyx()
    
    # 逐行绘制
    for i, line in enumerate(lines):
        # 计算当前行的Y坐标
        current_y = start_y + i
        
        # 检查Y坐标是否在屏幕范围内
        if current_y < 0 or current_y >= screen_height:
            continue
        
        # 绘制当前行的每个字符
        for j, char in enumerate(line):
            # 计算当前字符的X坐标
            current_x = start_x + j
            
            # 检查X坐标是否在屏幕范围内
            if current_x < 0 or current_x >= screen_width:
                continue
            
            # 在屏幕上绘制字符
            try:
                stdscr.addstr(current_y, current_x, char, curses.color_pair(color_pair))
            except curses.error:
                # 忽略绘制错误（通常是因为坐标越界）
                pass

def draw_ascii_art_advanced(stdscr, art_string, position_type="bottom", horizontal="center", color_pair=1, offset_y=0, offset_x=0):
    """
    增强版ASCII艺术绘制函数，支持自动对齐
    
    参数:
        stdscr: curses窗口对象
        art_string: ASCII艺术字符串
        position_type: 垂直位置 ("top", "center", "bottom")
        horizontal: 水平位置 ("left", "center", "right")
        color_pair: 颜色对编号
        offset_y: 垂直偏移量（正数向下）
        offset_x: 水平偏移量（正数向右）
    """
    # 分割字符串为行列表
    lines = art_string.split('\n')
    
    # 获取屏幕尺寸
    screen_height, screen_width = stdscr.getmaxyx()
    
    # 计算ASCII艺术的高度和最大宽度
    art_height = len(lines)
    art_width = max(len(line) for line in lines) if lines else 0
    
    # 计算垂直起始位置
    if position_type == "top":
        start_y = 1 + offset_y  # 留出第0行可能用于显示信息
    elif position_type == "center":
        start_y = (screen_height - art_height) // 2 + offset_y
    elif position_type == "bottom":
        start_y = screen_height - art_height - 1 + offset_y  # -1为地面留出空间
    else:
        start_y = offset_y
    
    # 计算水平起始位置
    if horizontal == "left":
        start_x = 2 + offset_x  # 留出左边距
    elif horizontal == "center":
        start_x = (screen_width - art_width) // 2 + offset_x
    elif horizontal == "right":
        start_x = screen_width - art_width - 2 + offset_x  # 留出右边距
    else:
        start_x = offset_x
    
    # 确保位置在合理范围内
    start_y = max(0, min(start_y, screen_height - 1))
    start_x = max(0, min(start_x, screen_width - 1))
    
    # 调用基础绘制函数
    draw_ascii_art(stdscr, art_string, start_y, start_x, color_pair)
    
    # 返回实际绘制的位置，方便调试
    return start_y, start_x, art_height, art_width
def draw_ascii_buffered(stdscr, art_string, start_y, start_x, color_pair=1, ignore_chars=None):
    """
    使用缓冲区优化绘制效率
    
    特点：
    1. 识别连续非透明字符段
    2. 减少函数调用次数
    3. 智能边界处理
    """
    if ignore_chars is None:
        ignore_chars = {' ', '\t'}
    
    lines = art_string.rstrip('\n').split('\n')
    screen_height, screen_width = stdscr.getmaxyx()
    
    # 预编译颜色属性（减少重复计算）
    attr = curses.color_pair(color_pair)
    
    for rel_y, line in enumerate(lines):
        abs_y = start_y + rel_y
        
        # Y轴边界检查
        if abs_y < 0 or abs_y >= screen_height:
            continue
        
        line_len = len(line)
        if line_len == 0:
            continue
        
        # 遍历行，构建字符缓冲区
        buffer_start = -1
        buffer_chars = []
        
        for rel_x, char in enumerate(line):
            abs_x = start_x + rel_x
            
            # X轴边界检查
            if abs_x >= screen_width:
                break
            if abs_x < 0:
                continue
            
            if char in ignore_chars:
                # 遇到空格：如果缓冲区有内容，则绘制并清空
                if buffer_start != -1:
                    try:
                        stdscr.addstr(abs_y, buffer_start, ''.join(buffer_chars), attr)
                    except curses.error:
                        pass
                    buffer_start = -1
                    buffer_chars = []
            else:
                # 非空格字符：添加到缓冲区
                if buffer_start == -1:
                    buffer_start = abs_x
                buffer_chars.append(char)
        
        # 绘制行尾剩余的缓冲区内容
        if buffer_start != -1:
            try:
                stdscr.addstr(abs_y, buffer_start, ''.join(buffer_chars), attr)
            except curses.error:
                pass
def draw_ascii_transparent(stdscr, art_string, start_y, start_x, color_pair=1, 
                          ignore_chars=None, blend_mode="overwrite"):
    """
    绘制带透明效果的ASCII艺术（空格被视为透明）
    
    参数:
        stdscr: curses窗口对象
        art_string: ASCII艺术字符串（包含换行）
        start_y: 起始行（左上角Y坐标）
        start_x: 起始列（左上角X坐标）
        color_pair: 颜色对编号
        ignore_chars: 额外被视为透明的字符列表（默认为[' ']）
        blend_mode: 混合模式
            - "overwrite": 直接覆盖（默认）
            - "additive": 叠加模式（用于发光效果）
    """
    if ignore_chars is None:
        ignore_chars = [' ']
    
    # 分割字符串为行列表
    lines = art_string.rstrip('\n').split('\n')
    
    # 获取屏幕尺寸用于边界检查
    screen_height, screen_width = stdscr.getmaxyx()
    
    # 逐行绘制，只绘制非空格字符
    for y_offset, line in enumerate(lines):
        current_y = start_y + y_offset
        
        # 检查Y坐标是否在屏幕范围内
        if current_y < 0 or current_y >= screen_height:
            continue
        
        # 绘制当前行的每个非透明字符
        for x_offset, char in enumerate(line):
            # 如果字符在忽略列表中，跳过（保持透明）
            if char in ignore_chars:
                continue
                
            current_x = start_x + x_offset
            
            # 检查X坐标是否在屏幕范围内
            if current_x < 0 or current_x >= screen_width:
                continue
            
            try:
                if blend_mode == "additive":
                    # 叠加模式：获取当前位置原有字符
                    try:
                        # 尝试获取当前位置的字符和属性
                        # curses没有直接获取字符的方法，所以我们用另一种方式
                        # 对于叠加效果，我们简单地使用更亮的颜色
                        attr = curses.color_pair(color_pair) | curses.A_BOLD
                    except:
                        attr = curses.color_pair(color_pair)
                else:
                    # 覆盖模式
                    attr = curses.color_pair(color_pair)
                
                stdscr.addstr(current_y, current_x, char, attr)
            except curses.error:
                # 忽略绘制错误（通常是因为坐标越界）
                pass
            
def draw_ascii_art_transparent_advanced(stdscr, art_string, position_type="bottom", horizontal="center", color_pair=1, offset_y=0, offset_x=0, ignore_chars=None, blend_mode="overwrite"):
    """
    增强版ASCII艺术绘制函数，支持自动对齐
    
    参数:
        stdscr: curses窗口对象
        art_string: ASCII艺术字符串
        position_type: 垂直位置 ("top", "center", "bottom")
        horizontal: 水平位置 ("left", "center", "right")
        color_pair: 颜色对编号
        offset_y: 垂直偏移量（正数向下）
        offset_x: 水平偏移量（正数向右）
    """
    # 分割字符串为行列表
    lines = art_string.split('\n')
    
    # 获取屏幕尺寸
    screen_height, screen_width = stdscr.getmaxyx()
    
    # 计算ASCII艺术的高度和最大宽度
    art_height = len(lines)
    art_width = max(len(line) for line in lines) if lines else 0
    
    # 计算垂直起始位置
    if position_type == "top":
        start_y = 1 + offset_y  # 留出第0行可能用于显示信息
    elif position_type == "center":
        start_y = (screen_height - art_height) // 2 + offset_y
    elif position_type == "bottom":
        start_y = screen_height - art_height - 1 + offset_y  # -1为地面留出空间
    else:
        start_y = offset_y
    
    # 计算水平起始位置
    if horizontal == "left":
        start_x = 2 + offset_x  # 留出左边距
    elif horizontal == "center":
        start_x = (screen_width - art_width) // 2 + offset_x
    elif horizontal == "right":
        start_x = screen_width - art_width - 2 + offset_x  # 留出右边距
    else:
        start_x = offset_x
    
    # 确保位置在合理范围内
    start_y = max(0, min(start_y, screen_height - 1))
    start_x = max(0, min(start_x, screen_width - 1))
    
    # 调用基础绘制函数
    draw_ascii_buffered(stdscr, art_string, start_y, start_x, color_pair, ignore_chars)
    
    # 返回实际绘制的位置，方便调试
    return start_y, start_x, art_height, art_width