import pickle
import os
from copy import deepcopy

PEDESTRIAN_CLASS = 'pedestrian'

def filter_infos(in_path, out_path, drop_empty=True):
    with open(in_path, 'rb') as f:
        data = pickle.load(f)

    infos = data['infos']
    new_infos = []

    for info in infos:
        if 'gt_names' not in info:
            continue

        mask = [name == PEDESTRIAN_CLASS for name in info['gt_names']]

        if drop_empty and sum(mask) == 0:
            continue

        info_new = deepcopy(info)

        # filter GT fields
        info_new['gt_names'] = info['gt_names'][mask]
        info_new['gt_boxes'] = info['gt_boxes'][mask]

        if 'gt_velocity' in info:
            info_new['gt_velocity'] = info['gt_velocity'][mask]

        if 'num_lidar_pts' in info:
            info_new['num_lidar_pts'] = info['num_lidar_pts'][mask]

        if 'num_radar_pts' in info:
            info_new['num_radar_pts'] = info['num_radar_pts'][mask]

        new_infos.append(info_new)

    print(f'Original samples: {len(infos)}')
    print(f'Pedestrian samples: {len(new_infos)}')

    data['infos'] = new_infos

    with open(out_path, 'wb') as f:
        pickle.dump(data, f)

    print(f'Saved: {out_path}')


if __name__ == '__main__':
    root = 'data/nuscenes/'

    filter_infos(
        os.path.join(root, 'nuscenes_infos_train.pkl'),
        os.path.join(root, 'nuscenes_infos_train_ped.pkl'),
        drop_empty=True
    )

    filter_infos(
        os.path.join(root, 'nuscenes_infos_val.pkl'),
        os.path.join(root, 'nuscenes_infos_val_ped.pkl'),
        drop_empty=True
    )
