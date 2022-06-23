from skimage import io
import numpy as np
import glob
from pathlib import Path

root_split_path = '/home/lee/pp/data/stones/testing/image/000001.png'
root_split_paths = '/home/lee/pp/data/stones/testing/lidar/000000.bin'


def set_split(self):
    sample_file_list = glob.glob(root_split_path + '/lidar/*.bin')
    sample_file_list.sort()
    sample_id_list = []
    for i in range(len(sample_file_list)):
        sample_id_list.append(Path(sample_file_list[i]).stem)
    return sample_id_list


def get_lidar():
    lidar_file = root_split_paths
    return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)


if __name__ == '__main__':
    a = 0
    # img_file = root_split_path
    # pts_img = np.array(io.imread(img_file).shape[:2], dtype=np.int32) #[375,1242]
    points = get_lidar()
    pts_rect = points[:, 0:3]
    print(len(pts_rect))
    val_flag_1 = np.logical_and(pts_rect[:, 0] >= 0, pts_rect[:, 0] < 100)
    val_flag_2 = np.logical_and(pts_rect[:, 1] >= 0, pts_rect[:, 1] < 100)
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, 3 >= 0)
    for i in val_flag_merge:
        if i:
            a += 1
    print(a)
