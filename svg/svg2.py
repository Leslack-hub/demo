from svgpathtools import svg2paths
from svgpathtools import wsvg, disvg


def extract_outer_contour(svg_filename):
    # 读取SVG文件中的所有路径
    paths, attributes = svg2paths(svg_filename)

    # 初始化最外层轮廓的变量
    outer_path = None
    max_area = 0

    # 遍历所有路径，找到最大的边界框
    for path in paths:
        # 计算路径的边界框
        min_x, max_x, min_y, max_y = path.bbox()
        area = (max_x - min_x) * (max_y - min_y)

        # 更新最外层轮廓
        if area > max_area:
            max_area = area
            outer_path = path

    # 返回最外层轮廓
    return outer_path


def save_new_svg_with_outer_contour(outer_path, output_filename):
    # 使用wsvg函数创建一个新的SVG文件，只包含最外层轮廓
    wsvg([outer_path], filename=output_filename)


# 使用示例
svg_filename = 'a_0010.svg'  # 输入的SVG文件名
output_filename = 'output.svg'  # 输出的SVG文件名

# 提取最外层轮廓
outer_path = extract_outer_contour(svg_filename)

# 如果找到了最外层轮廓，保存到新的SVG文件中
if outer_path:
    save_new_svg_with_outer_contour(outer_path, output_filename)
    print(f"Outer contour saved to {output_filename}")
else:
    print("No outer contour found.")
