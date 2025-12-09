#!/usr/bin/env python3
import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
META_DIR = BASE / 'v1.0-trainval'
SAMPLES_DIR = BASE / 'samples'
OUT_DIR = BASE / 'v1.0-trainval_trimmed'

def load_json(p):
    with open(p, 'r') as f:
        return json.load(f)

def save_json(obj, p):
    with open(p, 'w') as f:
        json.dump(obj, f, indent=2)

def main():
    print('Base dir:', BASE)
    OUT_DIR.mkdir(exist_ok=True)

    # Load large metadata files
    print('Loading sample_data.json ...')
    sample_data = load_json(META_DIR / 'sample_data.json')
    print('Loaded', len(sample_data), 'sample_data entries')

    # Determine which sample_data files actually exist
    kept_sample_data = []
    kept_sample_data_tokens = set()
    for sd in sample_data:
        # try common keys
        file_key = sd.get('file_name') or sd.get('filename') or sd.get('file_name_cam') or sd.get('file_path')
        if not file_key:
            # try data['filename'] style
            file_key = sd.get('filename', '')
        if not file_key:
            continue
        file_path = BASE / file_key
        if file_path.exists():
            kept_sample_data.append(sd)
            kept_sample_data_tokens.add(sd['token'])

    print(f'Kept {len(kept_sample_data)} sample_data entries (files exist)')

    print('Filtering sample.json ...')
    samples = load_json(META_DIR / 'sample.json')
    kept_samples = []
    kept_sample_tokens = set()
    # Many sample.json entries may have 'data' as None; instead use sample_token in sample_data
    sample_tokens_with_data = set(sd['sample_token'] for sd in kept_sample_data)
    for s in samples:
        if s['token'] in sample_tokens_with_data:
            kept_samples.append(s)
            kept_sample_tokens.add(s['token'])

    print(f'Kept {len(kept_samples)} samples (have at least one sample_data)')

    print('Filtering sample_annotation.json ...')
    annos = load_json(META_DIR / 'sample_annotation.json')
    kept_annos = [a for a in annos if a.get('sample_token') in kept_sample_tokens]
    print(f'Kept {len(kept_annos)} annotations')

    # Filter scene.json: keep scenes that have any kept samples, and update first/last sample tokens
    print('Filtering scene.json ...')
    scenes = load_json(META_DIR / 'scene.json')
    # build mapping sample_token -> sample object for ordering
    sample_by_token = {s['token']: s for s in kept_samples}
    # group samples by scene_token
    samples_by_scene = {}
    for s in kept_samples:
        samples_by_scene.setdefault(s['scene_token'], []).append(s)

    kept_scenes = []
    for sc in scenes:
        st = sc['token']
        sc_samples = samples_by_scene.get(st, [])
        if not sc_samples:
            continue
        # sort samples by timestamp (if present) or keep existing order
        sc_samples.sort(key=lambda x: x.get('timestamp', 0))
        sc['first_sample_token'] = sc_samples[0]['token']
        sc['last_sample_token'] = sc_samples[-1]['token']
        kept_scenes.append(sc)

    print(f'Kept {len(kept_scenes)} scenes')

    # Filter ego_pose and calibrated_sensor and sensor if referenced
    print('Filtering ego_pose/calibrated_sensor/sensor ...')
    ego_pose = load_json(META_DIR / 'ego_pose.json')
    calib = load_json(META_DIR / 'calibrated_sensor.json')
    sensor = load_json(META_DIR / 'sensor.json')

    # collect referenced tokens
    ego_tokens = set(sd.get('ego_pose_token') for sd in kept_sample_data if sd.get('ego_pose_token'))
    calib_tokens = set(sd.get('calibrated_sensor_token') for sd in kept_sample_data if sd.get('calibrated_sensor_token'))
    sensor_tokens = set()
    # calibrated_sensor entries reference sensor_token
    kept_calib = [c for c in calib if c['token'] in calib_tokens]
    sensor_tokens.update(c.get('sensor_token') for c in kept_calib if c.get('sensor_token'))
    kept_ego = [e for e in ego_pose if e['token'] in ego_tokens]
    kept_sensor = [s for s in sensor if s['token'] in sensor_tokens]

    print(f'Keeping {len(kept_ego)} ego_pose, {len(kept_calib)} calibrated_sensor, {len(kept_sensor)} sensor entries')

    # Save trimmed JSONs
    print('Saving trimmed metadata to', OUT_DIR)
    save_json(kept_sample_data, OUT_DIR / 'sample_data.json')
    save_json(kept_samples, OUT_DIR / 'sample.json')
    save_json(kept_annos, OUT_DIR / 'sample_annotation.json')
    # Copy others but try to filter where sensible
    save_json(kept_scenes, OUT_DIR / 'scene.json')
    save_json(kept_ego, OUT_DIR / 'ego_pose.json')
    save_json(kept_calib, OUT_DIR / 'calibrated_sensor.json')
    save_json(kept_sensor, OUT_DIR / 'sensor.json')

    # For any remaining metadata files that aren't filtered, copy them as-is
    other_files = ['attribute.json', 'category.json', 'instance.json', 'log.json', 'map.json', 'visibility.json']
    for of in other_files:
        src = META_DIR / of
        dst = OUT_DIR / of
        if src.exists():
            with open(src, 'r') as fsrc, open(dst, 'w') as fdst:
                fdst.write(fsrc.read())

    # write a trimmed meta text file
    trimmed_meta_txt = BASE / '.v1.0-trainval_meta_2blobs.txt'
    with open(trimmed_meta_txt, 'w') as f:
        f.write('Trimmed nuScenes meta for local use (2 blobs)\n')
        f.write('- v1.0-trainval01_blobs.tgz\n')
        f.write('- v1.0-trainval02_blobs.tgz\n')

    # create tarball
    import tarfile
    out_tgz = BASE / 'v1.0-trainval_meta_2blobs.tgz'
    print('Creating tarball', out_tgz)
    with tarfile.open(out_tgz, 'w:gz') as tar:
        # add the trimmed v1.0-trainval folder
        tar.add(OUT_DIR.as_posix(), arcname='v1.0-trainval')
        tar.add(trimmed_meta_txt.as_posix(), arcname='.v1.0-trainval_meta.txt')

    print('Done. Trimmed meta tarball created at', out_tgz)


if __name__ == '__main__':
    main()
