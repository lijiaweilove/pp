import glob
from pathlib import Path
import numpy as np
from pcdet.utils import object3d_kitti, calibration_kitti
import sys

root_split_path = '/home/lee/pp/data/stone/testing'


def get_label(idx):
    label_file = root_split_path + '/label/' + ('%s.txt' % idx)
    return object3d_kitti.get_objects_from_label(label_file)


def set_split():
    sample_file_list = glob.glob(root_split_path + '/lidar/*.bin')
    sample_file_list.sort()
    sample_id_list = []
    for i in range(len(sample_file_list)):
        sample_id_list.append(Path(sample_file_list[i]).stem)
    return sample_id_list


def get_calib(idx):
    calib_file = root_split_path + '/calib/' + ('%s.txt' % idx)
    return calibration_kitti.Calibration(calib_file)


def get_infos(num_workers, count_inside_pts=True, sample_id_list=None):
    import concurrent.futures as futures

    def process_single_scene(sample_idx):
        calib = get_calib(sample_idx)
        info = {}
        obj_list = get_label(sample_idx)
        annotations = {}
        annotations['sample_idx'] = sample_idx
        annotations['name'] = np.array([obj.cls_type for obj in obj_list])
        annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
        annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
        annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])

        num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
        num_gt = len(annotations['name'])
        index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
        annotations['index'] = np.array(index, dtype=np.int32)

        loc = annotations['location'][:num_objects]
        dims = annotations['dimensions'][:num_objects]
        rots = annotations['rotation_y'][:num_objects]
        loc_lidar = calib.rect_to_lidar(loc)
        l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
        loc_lidar[:, 2] += h[:, 0] / 2
        gt_boxes_lidar = np.concatenate([loc_lidar, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
        # gt_boxes_lidar = np.concatenate([loc, l, w, h, rots[..., np.newaxis]], axis=1)
        annotations['gt_boxes_lidar'] = gt_boxes_lidar
        print(annotations)
        # info['annos'] = annotations

    sample_id_list = set_split()
    with futures.ThreadPoolExecutor(num_workers) as executor:
        executor.map(process_single_scene, sample_id_list)


if __name__ == '__main__':
    get_infos(num_workers=4, count_inside_pts=True)


