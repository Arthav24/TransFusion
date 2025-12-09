#!/usr/bin/env python3
"""
Trim nuScenes metadata to only the samples present in a selected set of blob archives.

Usage examples:
  # Use blobs by filename (they must exist in dataset root). Will extract them if --extract is set.
  python3 tools/trim_meta_by_blobs.py --root . --blobs v1.0-trainval03_blobs.tgz,v1.0-trainval04_blobs.tgz --extract

  # Use blobs by index (pairs like 1,2 or 3,4). The script will look for files named
  # `v1.0-trainval01_blobs.tgz`, `v1.0-trainval02_blobs.tgz`, etc. in the root.
  python3 tools/trim_meta_by_blobs.py --root . --blobs 3,4 --extract

Output:
  - Creates trimmed JSONs in `<root>/v1.0-trainval_trim_<tag>/`
  - Creates tarball `<root>/v1.0-trainval_meta_<tag>.tgz` containing the trimmed `v1.0-trainval/` and a small `.v1.0-trainval_meta.txt`

This is an automated, safer variant of the earlier one-off trimming process.
"""
import argparse
import json
import os
import tarfile
from pathlib import Path
from typing import List


def parse_blobs_arg(blobs_arg: str, root: Path) -> List[Path]:
    """Return a list of blob paths from a comma-separated arg.
    Accepts full filenames or numeric indices (1-based) which are expanded
    to `v1.0-trainvalXX_blobs.tgz`.
    """
    items = [b.strip() for b in blobs_arg.split(',') if b.strip()]
    out = []
    for it in items:
        if it.isdigit():
            idx = int(it)
            fname = f'v1.0-trainval{idx:02d}_blobs.tgz'
            out.append(root / fname)
        else:
            out.append(root / it)
    return out


def extract_blobs(blobs: List[Path], root: Path):
    for b in blobs:
        if not b.exists():
            raise FileNotFoundError(f'Blob archive not found: {b}')
        print(f'Extracting {b.name} ...')
        with tarfile.open(b, 'r:gz') as tar:
            tar.extractall(path=root)


def load_json(p: Path):
    with open(p, 'r') as f:
        return json.load(f)


def save_json(obj, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w') as f:
        json.dump(obj, f, indent=2)


def build_trimmed_meta(root: Path, out_dir: Path, tag: str):
    meta_dir = root / 'v1.0-trainval'
    if not meta_dir.exists():
        raise FileNotFoundError(f'{meta_dir} not found under root')

    print('Loading sample_data.json ...')
    sample_data = load_json(meta_dir / 'sample_data.json')

    print('Checking which sample_data files exist...')
    kept_sample_data = []
    kept_sample_tokens = set()
    for sd in sample_data:
        fname = sd.get('filename') or sd.get('file_name') or sd.get('fileformat') or sd.get('file_name_cam')
        if not fname:
            continue
        # Try relative to root
        fpath = root / fname
        if not fpath.exists():
            # try direct path as-is (in case filenames are absolute or already correct)
            if not Path(fname).exists():
                continue
            else:
                fpath = Path(fname)
        kept_sample_data.append(sd)
        kept_sample_tokens.add(sd['sample_token'])

    print(f'Kept {len(kept_sample_data)} sample_data entries')

    # Filter samples by sample_token present in sample_data
    samples = load_json(meta_dir / 'sample.json')
    kept_samples = [s for s in samples if s['token'] in kept_sample_tokens]
    kept_sample_tokens_set = set(s['token'] for s in kept_samples)
    print(f'Kept {len(kept_samples)} samples')

    # Filter annotations
    annos = load_json(meta_dir / 'sample_annotation.json')
    kept_annos = [a for a in annos if a.get('sample_token') in kept_sample_tokens_set]
    print(f'Kept {len(kept_annos)} annotations')

    # Filter scenes: keep only scenes that have kept samples
    scenes = load_json(meta_dir / 'scene.json')
    samples_by_scene = {}
    for s in kept_samples:
        samples_by_scene.setdefault(s['scene_token'], []).append(s)
    kept_scenes = []
    for sc in scenes:
        st = sc['token']
        sc_samples = samples_by_scene.get(st, [])
        if not sc_samples:
            continue
        sc_samples.sort(key=lambda x: x.get('timestamp', 0))
        sc['first_sample_token'] = sc_samples[0]['token']
        sc['last_sample_token'] = sc_samples[-1]['token']
        kept_scenes.append(sc)
    print(f'Kept {len(kept_scenes)} scenes')

    # Filter ego_pose, calibrated_sensor, sensor
    ego_pose = load_json(meta_dir / 'ego_pose.json')
    calib = load_json(meta_dir / 'calibrated_sensor.json')
    sensor = load_json(meta_dir / 'sensor.json')

    ego_tokens = set(sd.get('ego_pose_token') for sd in kept_sample_data if sd.get('ego_pose_token'))
    calib_tokens = set(sd.get('calibrated_sensor_token') for sd in kept_sample_data if sd.get('calibrated_sensor_token'))
    kept_calib = [c for c in calib if c['token'] in calib_tokens]
    sensor_tokens = set(c.get('sensor_token') for c in kept_calib if c.get('sensor_token'))
    kept_ego = [e for e in ego_pose if e['token'] in ego_tokens]
    kept_sensor = [s for s in sensor if s['token'] in sensor_tokens]

    # Map handling: keep entries only if the filename exists; if none exist, create a minimal entry
    mapf = meta_dir / 'map.json'
    map_entries = []
    if mapf.exists():
        map_entries_all = load_json(mapf)
        for m in map_entries_all:
            fname = m.get('filename')
            if not fname:
                continue
            if (root / fname).exists() or Path(fname).exists():
                map_entries.append(m)

    if not map_entries:
        # create a minimal map entry pointing to the first existing sample image
        if kept_sample_data:
            first_sd = kept_sample_data[0]
            candidate = first_sd.get('filename')
            if candidate and ((root / candidate).exists() or Path(candidate).exists()):
                map_entries = [{
                    'category': 'semantic_prior',
                    'token': 'dummy_map',
                    'filename': candidate,
                    'log_tokens': [l['token'] for l in load_json(meta_dir / 'log.json')]
                }]
            else:
                map_entries = []

    # Save trimmed files
    out_meta = out_dir / 'v1.0-trainval'
    out_meta.mkdir(parents=True, exist_ok=True)
    save_json(kept_sample_data, out_meta / 'sample_data.json')
    save_json(kept_samples, out_meta / 'sample.json')
    save_json(kept_annos, out_meta / 'sample_annotation.json')
    save_json(kept_scenes, out_meta / 'scene.json')
    save_json(kept_ego, out_meta / 'ego_pose.json')
    save_json(kept_calib, out_meta / 'calibrated_sensor.json')
    save_json(kept_sensor, out_meta / 'sensor.json')
    save_json(map_entries, out_meta / 'map.json')

    # copy other metadata files unchanged (attribute, category, instance, log, visibility)
    for of in ['attribute.json', 'category.json', 'instance.json', 'log.json', 'visibility.json']:
        src = meta_dir / of
        if src.exists():
            with open(src, 'r') as fsrc, open(out_meta / of, 'w') as fdst:
                fdst.write(fsrc.read())

    # small meta text
    meta_txt = out_dir / f'.v1.0-trainval_meta_{tag}.txt'
    with open(meta_txt, 'w') as f:
        f.write(f'Trimmed metadata for local use ({tag})\n')
        f.write('Blobs used:\n')
        f.write('\n')

    # create tarball
    out_tgz = root / f'v1.0-trainval_meta_{tag}.tgz'
    print('Creating', out_tgz)
    with tarfile.open(out_tgz, 'w:gz') as tar:
        tar.add(out_meta.as_posix(), arcname='v1.0-trainval')
        tar.add(meta_txt.as_posix(), arcname=f'.v1.0-trainval_meta.txt')

    print('Trimmed meta created:', out_tgz)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='.', help='dataset root folder')
    parser.add_argument('--blobs', type=str, required=True,
                        help='Comma-separated blob filenames or indices (e.g. "1,2" or "v1.0-trainval01_blobs.tgz,v1.0-trainval02_blobs.tgz")')
    parser.add_argument('--extract', action='store_true', help='Extract the selected blob archives into the root')
    parser.add_argument('--tag', type=str, default=None, help='tag to embed in output names (default: blobs joined)')
    args = parser.parse_args()

    root = Path(args.root).resolve()
    blobs = parse_blobs_arg(args.blobs, root)
    tag = args.tag or '_'.join([b.name.replace('.tgz', '') for b in blobs])

    if args.extract:
        extract_blobs(blobs, root)

    out_dir = root / f'v1.0-trainval_trim_{tag}'
    if out_dir.exists():
        print('Removing existing out_dir', out_dir)
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    build_trimmed_meta(root, out_dir, tag)


if __name__ == '__main__':
    main()
