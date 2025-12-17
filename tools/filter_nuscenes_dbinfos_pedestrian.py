import pickle
import os

PEDESTRIAN_CLASS = 'pedestrian'

def filter_dbinfos(in_path, out_path):
    with open(in_path, 'rb') as f:
        dbinfos = pickle.load(f)

    print('Original classes:', list(dbinfos.keys()))

    if PEDESTRIAN_CLASS not in dbinfos:
        raise KeyError(f'{PEDESTRIAN_CLASS} not found in dbinfos')

    new_dbinfos = {
        PEDESTRIAN_CLASS: dbinfos[PEDESTRIAN_CLASS]
    }

    print(f'Pedestrian instances: {len(new_dbinfos[PEDESTRIAN_CLASS])}')

    with open(out_path, 'wb') as f:
        pickle.dump(new_dbinfos, f)

    print(f'Saved pedestrian-only dbinfos to: {out_path}')


if __name__ == '__main__':
    root = 'data/nuscenes/'

    filter_dbinfos(
        os.path.join(root, 'nuscenes_dbinfos_train.pkl'),
        os.path.join(root, 'nuscenes_dbinfos_train_ped.pkl')
    )
