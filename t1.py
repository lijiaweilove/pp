import glob
from pathlib import Path
import numpy as np
from pcdet.utils import object3d_kitti

root_split_path = '/home/lee/pp/data/stone/training'

def get_label(idx):
    label_file = root_split_path + '/lidar/' + ('%s.txt' % idx)
    # return object3d_kitti.get_objects_from_label(label_file)
    objects = []
    with open(label_file, 'r') as f:
        for i in f.readlines():
            object1 = i.strip().split(' ')
            # print(object1)
            objects.append(object1)
    return objects

def set_split():
    sample_file_list = glob.glob(root_split_path + '/lidar/*.txt')
    sample_file_list.sort()
    sample_id_list = []
    for i in range(len(sample_file_list)):
        sample_id_list.append(Path(sample_file_list[i]).stem)
    return sample_id_list

def get_infos(num_workers, count_inside_pts=True, sample_id_list=None):
    import concurrent.futures as futures

    def process_single_scene(sample_idx):
        info = {}
        obj_list = get_label(sample_idx)
        # for obj in obj_list:
        #     print(obj[0])
        annotations = {}
        # annotations['name'] = np.array([obj.cls_type for obj in obj_list])
        annotations['name'] = np.array([str(obj[0]) for obj in obj_list])
        num_gt = len(annotations['name'])
        annotations['name'].reshape(num_gt)
        annotations['x'] = np.array([float(obj[11]) for obj in obj_list]).reshape(num_gt, 1)
        annotations['y'] = np.array([float(obj[12]) for obj in obj_list]).reshape(num_gt, 1)
        annotations['z'] = np.array([float(obj[13]) for obj in obj_list]).reshape(num_gt, 1)
        annotations['l'] = np.array([float(obj[10]) for obj in obj_list]).reshape(num_gt, 1)
        annotations['w'] = np.array([float(obj[9]) for obj in obj_list]).reshape(num_gt, 1)
        annotations['h'] = np.array([float(obj[8]) for obj in obj_list]).reshape(num_gt, 1)
        annotations['rotation_y'] = np.array([float(obj[14]) for obj in obj_list]).reshape(num_gt, 1)
        x = annotations['x']
        y = annotations['y']
        z = annotations['z']
        l = annotations['l']
        w = annotations['w']
        h = annotations['h']
        # wanji数据集中3Dbbox角度是相对于y轴正方向 且正方向为顺时针角度 需要进行坐标角度转换
        # 在统一坐标系下 heading是相对于x轴的夹角 并且逆时针方向为正
        rots = annotations['rotation_y']
        gt_boxes_lidar = np.concatenate([x, y, z, l, w, h, rots], axis=1)
        annotations['gt_boxes_lidar'] = gt_boxes_lidar
        info['annos'] = annotations
        print(info)

    sample_id_list = set_split()
    with futures.ThreadPoolExecutor(num_workers) as executor:
        info=executor.map(process_single_scene, sample_id_list)

if __name__ == '__main__':
    get_infos(num_workers=4, count_inside_pts=True)