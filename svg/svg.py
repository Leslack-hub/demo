import xml.etree.ElementTree as ET
import svgpathtools
from pathlib import Path
from pathlib import Path as Path2
import numpy as np
from matplotlib.path import Path
from svgpathtools import svg2paths


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


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def path_to_svg_string(path):
    result = ''
    for segment in path:
        path_data = segment.d()
        formatted_path_data = ""
        for part in path_data.split():
            formatted_path_data += ' '
            p = []
            for j in part.split(","):
                if len(j) > 5:
                    p.append(j[:5])
                else:
                    p.append(j)
            formatted_path_data += ','.join(p)
        result += formatted_path_data.strip()
    return result
    # return ''.join(segment.d() for segment in path)


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
    path.set('stroke-width', '3')
    path.set("transform", "translate(3,3)")
    path.set("stroke-linejoin", "round")
    path.set("stroke-linecap", "round")

    root.insert(0, path)
    root.set('xmlns', 'http://www.w3.org/2000/svg')
    view = root.get('viewBox').split(' ')
    view[2] = str(int(view[2]) + 6)
    view[3] = str(int(view[3]) + 6)
    root.set('viewBox', ' '.join(view))
    for i, elem in enumerate(root.iter()):
        elem.tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        if elem.tag == 'g':
            continue
        if i > 0:
            elem.set("transform", "translate(3,3)")
            # elem.set("stroke-linejoin", "round")
            # elem.set("stroke-linecap", "round")

    svg_string = ET.tostring(root, encoding='unicode')
    compressed_svg = ' '.join(svg_string.split())
    with open('new/' + input_file.name, 'w', encoding='utf-8') as f:
        f.write(compressed_svg)


if __name__ == '__main__':
    folder_path = Path2('icon/')
    for file_path in folder_path.iterdir():
        if file_path.is_file():
            add_path_to_svg2(file_path)
