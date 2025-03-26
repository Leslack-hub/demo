import xml.etree.ElementTree as ET
import svgpathtools
from pathlib import Path
from pathlib import Path as Path2
import numpy as np
from matplotlib.path import Path


def get_outer_contour(paths):
    # 计算每个路径的面积
    path_areas = []
    for path in paths:
        # 使用近似方法计算路径包围的面积
        bbox = path.bbox()
        area = abs((bbox[1] - bbox[0]) * (bbox[3] - bbox[2]))
        path_areas.append(area)

    # 返回面积最大的路径（最外层轮廓）
    max_area_index = np.argmax(path_areas)
    return [paths[max_area_index]]


def get_svg_outline(svg_file):
    paths, attributes = svg2paths(svg_file)
    outlines = []
    # outer_paths = []
    # for path, attr in zip(paths, attributes):
    # 检查路径的层级或位置
    # if attr.get('fill') and attr.get('fill') != 'none':
    #     outer_paths.append(path)
    for path in paths:
        outlines.append(path)

    # return path_to_svg_string(get_outer_contour(outer_paths))
    return path_to_svg_string(outlines)


def get_most_likely_outer_contour(svg_file):
    paths, attributes = svg2paths(svg_file)

    # 综合判断：长度、面积、层级
    def path_score(path):
        length_score = path.length()
        bbox = path.bbox()
        area_score = abs((bbox[1] - bbox[0]) * (bbox[3] - bbox[2]))
        return length_score * area_score

    # 选择得分最高的路径
    return path_to_svg_string([max(paths, key=path_score)])


def path_to_svg_string(path):
    return ''.join(segment.d() for segment in path)


def get_max_contour(svg_file):
    # 读取SVG文件
    paths, attributes = svg2paths(svg_file)

    max_contour = None
    max_area = 0

    for path in paths:
        # 获取路径的点
        points = np.array([complex(segment.start.real, segment.start.imag) for segment in path] +
                          [complex(segment.end.real, segment.end.imag) for segment in path])

        # 计算路径的面积
        area = 0.5 * np.abs(np.dot(points.real, np.roll(points.imag, 1)) - np.dot(points.imag, np.roll(points.real, 1)))

        # 更新最大轮廓线
        if area > max_area:
            max_area = area
            max_contour = path

    return max_contour


def add_path_to_svg2(input_file):
    try:
        tree = ET.parse(input_file)
    except ET.ParseError:
        return
    root = tree.getroot()
    path = ET.Element('path')
    path.set('d', get_svg_outline(input_file))
    path.set('fill', 'none')
    path.set('stroke', 'white')
    path.set('stroke-width', '1.5')
    path.set("transform", "translate(1.500000, 1.500000)")

    root.insert(0, path)
    root.set('xmlns', 'http://www.w3.org/2000/svg')
    view = root.get('viewBox').split(' ')
    view[2] = str(int(view[2]) + 3)
    view[3] = str(int(view[3]) + 3)
    root.set('viewBox', ' '.join(view))
    for i, elem in enumerate(root.iter()):
        elem.tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        if i > 0:
            elem.set("transform", "translate(1.500000, 1.500000)")
            elem.set("stroke-linejoin", "round")

    svg_string = ET.tostring(root, encoding='unicode')
    compressed_svg = ' '.join(svg_string.split())
    with open('new/' + input_file.name, 'w', encoding='utf-8') as f:
        f.write(compressed_svg)


from svgpathtools import svg2paths, wsvg
from shapely.geometry import Polygon, LineString


def extract_outermost_outline(svg_file_path):
    # 读取SVG文件并提取路径
    paths, attributes = svg2paths(svg_file_path)

    if not paths:
        return None, "No paths found in the SVG file."

    # 将所有路径转换为Shapely的LineString对象
    line_strings = [LineString([(p.start.real, p.start.imag) for p in path]) for path in paths]

    # 计算每个路径的面积，假设外轮廓的面积最大
    areas = [abs(Polygon(line_string).area) for line_string in line_strings]

    # 找到面积最大的路径索引
    max_area_index = areas.index(max(areas))

    # 获取最大面积的路径
    outline_path = paths[max_area_index]

    # 将路径转换为字符串形式的SVG path 'd'属性
    d = outline_path.d()

    # 创建一个新的SVG文件只包含这个路径
    new_svg_file = 'outermost_outline.svg'
    wsvg(paths=[outline_path], filename=new_svg_file)

    return d, new_svg_file


def extract_largest_contour(svg_filename):
    # 加载SVG文件并解析所有路径
    paths, attributes = svg2paths(svg_filename)

    largest_area = 0
    largest_path = None

    # 遍历路径并使用matplotlib的Path对象计算面积
    for path in paths:
        # 将svg路径转换为matplotlib路径
        codes = []
        verts = []
        for segment in path:
            if len(verts) == 0:  # 开始一个新的子路径
                verts.append((segment.start.real, segment.start.imag))
                codes.append(Path.MOVETO)
            verts.append((segment.end.real, segment.end.imag))
            codes.append(Path.LINETO)

        # 需要将verts从list转换为np.array
        verts_np = np.array(verts)

        # 创建路径对象
        mp_path = Path(verts_np, codes)

        # 计算路径的轮廓面积，修复此处的错误
        area = 0.5 * np.abs(np.dot(verts_np[:-1, 0], verts_np[1:, 1]) - np.dot(verts_np[:-1, 1], verts_np[1:, 0]))

        # 更新最大轮廓
        if area > largest_area:
            largest_area = area
            largest_path = mp_path

    # 返回最大轮廓的路径对象
    return largest_path


def path_to_svg_d_string(path):
    """将matplotlib的Path对象转换为SVG路径的d字符串"""
    d_string = ""
    # 遍历路径的codes和vertices
    for code, (x, y) in zip(path.codes, path.vertices):
        if code == Path.MOVETO:
            d_string += f"M {x} {y} "
        elif code == Path.LINETO:
            d_string += f"L {x} {y} "
        elif code == Path.CLOSEPOLY:
            d_string += "Z "
    return d_string.strip()


# 使用示例
# if __name__ == '__main__':
#     svg_file = 'input.svg'  # 替换为你的SVG文件路径
#     largest_contour_path = extract_largest_contour(svg_file)
#     if largest_contour_path:
#         svg_d_string = path_to_svg_d_string(largest_contour_path)
#         print("SVG Path d string:", svg_d_string)
#     else:
#         print("没有找到轮廓")

if __name__ == '__main__':
    folder_path = Path2('icon/')
    for file_path in folder_path.iterdir():
        if file_path.is_file():
            add_path_to_svg2(file_path)
