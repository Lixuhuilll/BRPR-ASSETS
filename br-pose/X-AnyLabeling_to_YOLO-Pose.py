import sys
from glob import glob
from json import load as json_load
from os import path as op, makedirs

from tqdm import tqdm

# 类别文件
classes_path = 'labels\\classes.txt'
# 输出 yolo txt 的文件夹
yolo_txt_dir = op.abspath('labels\\train')
# 关键点的数量（用于对齐测试集中的关键点）
key_point_count = 4
# 获取所有 json 的位置
json_paths = [op.abspath(p) for p in glob('images\\train\\*.json')]

classes_dict: dict[str, int]


def json_to_yolo(json_data):
    # 步骤：
    # 1. 找出所有的矩形，记录下矩形的坐标，以及对应group_id
    # 2. 遍历所有的head和tail，记下点的坐标，以及对应group_id，加入到对应的矩形中
    # 3. 转为yolo格式
    rectangles = {}
    # 遍历初始化
    for shape in json_data['shapes']:
        label = shape['label']  # pen, head, tail
        group_id = shape['group_id']  # 0, 1, 2, ...
        points = shape['points']  # x,y coordinates
        shape_type = shape['shape_type']

        # 只处理矩形,读矩形
        if shape_type == 'rectangle':
            if group_id not in rectangles:
                rectangles[group_id] = {
                    'label': label,
                    'rect': points[0] + points[2],  # Rectangle [x1, y1, x2, y2]
                    'key_point_list': []
                }

    # 遍历更新，将点加入对应group_id的矩形中，读关键点，根据group_id匹配
    for shape in json_data['shapes']:
        label = shape['label']
        group_id = shape['group_id']
        points = shape['points']
        shape_type = shape['shape_type']

        # 处理点
        if shape_type == 'point':
            points[0].append(int(label))
            rectangles[group_id]['key_point_list'].append(points[0])

    # 转为yolo格式
    yolo_list = []
    for _id, rectangle in rectangles.items():
        if rectangle['label'] not in classes_dict:
            continue
        label_id = classes_dict[rectangle['label']]
        # x1,y1,x2,y2
        x1, y1, x2, y2 = rectangle['rect']
        # center_x, center_y, width, height
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = abs(x1 - x2)
        height = abs(y1 - y2)
        # normalize
        h, w = json_data['imageHeight'], json_data['imageWidth']
        center_x /= w
        center_y /= h
        width /= w
        height /= h

        # 保留6位小数
        center_x = round(center_x, 6)
        center_y = round(center_y, 6)
        width = round(width, 6)
        height = round(height, 6)

        # 添加 label_id, center_x, center_y, width, height
        result_list = [label_id, center_x, center_y, width, height]

        # 添加 p1_x, p1_y, p1_v, p2_x, p2_y, p2_v
        for point in rectangle['key_point_list']:
            x, y, v = point
            x /= w
            y /= h
            # 保留6位小数
            x = round(x, 6)
            y = round(y, 6)
            result_list.extend([x, y, v])

        for _ in range(key_point_count - len(rectangle['key_point_list'])):
            result_list.extend([0, 0, 0])

        yolo_list.append(result_list)
    return yolo_list


def main():
    # 读取并创建物体类别字典
    with open('labels\\classes.txt') as f:
        global classes_dict
        classes_dict = {value: index for index, value in enumerate(f.read().splitlines())}

    # 创建保存 yolo txt 的文件夹
    makedirs(yolo_txt_dir, exist_ok=True)

    # 遍历 json 文件并输出进度条
    for json_path in tqdm(json_paths, file=sys.stdout):
        tqdm.write('In: ' + json_path)

        # 载入 json
        with open(json_path, 'r') as f:
            json_data = json_load(f)
        # 将 json 转化为 yolo txt
        yolo_list = json_to_yolo(json_data)

        # 确定 yolo txt 的输出位置
        yolo_txt_path = op.join(yolo_txt_dir, op.basename(
            op.splitext(json_data['imagePath'])[0]
        )) + '.txt'
        tqdm.write('Out: ' + yolo_txt_path)

        # 输出 yolo txt
        with open(yolo_txt_path, 'w') as f:
            for row in yolo_list:
                f.write(' '.join(str(element) for element in row))
                f.write('\n')


if __name__ == '__main__':
    main()
