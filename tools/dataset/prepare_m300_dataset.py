#!/usr/bin/env python3
"""Validate Sly export and prepare OpenPCDet custom points/labels.

Default is dry-run. --write creates a new output directory, never overwriting.
Requires NumPy only. Does not modify the OpenPCDet loader or start training.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def read_pcd(path):
    """Read CVAT's uncompressed little-endian float32 XYZI binary PCD."""
    with path.open('rb') as f:
        header = {}
        for _ in range(100):
            line = f.readline().decode('ascii').strip()
            if not line or line.startswith('#'):
                continue
            key, *values = line.split()
            header[key] = values
            if key == 'DATA':
                break
        expected = {'FIELDS': ['x','y','z','intensity'], 'SIZE': ['4']*4,
                    'TYPE': ['F']*4, 'COUNT': ['1']*4, 'DATA': ['binary']}
        for key, value in expected.items():
            if header.get(key) != value:
                raise ValueError(f'{path}: unsupported PCD {key}: {header.get(key)}')
        n = int(header['POINTS'][0])
        payload = f.read()
        if len(payload) != n * 16:
            raise ValueError(f'{path}: incorrect PCD payload size')
        return np.frombuffer(payload, dtype='<f4').reshape(n, 4)


def read_boxes(path):
    ann = json.loads(path.read_text())
    objects = {o['key']: o['classTitle'] for o in ann['objects']}
    boxes = []
    for fig in ann['figures']:
        if fig['geometryType'] != 'cuboid_3d' or objects[fig['objectKey']] != 'person':
            raise ValueError(f'{path}: expected person cuboid_3d only')
        g = fig['geometry']
        pos, dim, rot = [np.array([g[k][a] for a in 'xyz'], dtype=float)
                         for k in ('position','dimensions','rotation')]
        if not np.isfinite(np.r_[pos,dim,rot]).all() or (dim <= 0).any():
            raise ValueError(f'{path}: invalid box values')
        if np.abs(rot[:2]).max() > 1e-5:
            raise ValueError(f'{path}: roll/pitch not supported by yaw-only boxes')
        yaw = (rot[2] + math.pi) % (2*math.pi) - math.pi
        boxes.append([*pos, *dim, yaw])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 7)


def split_records(records, ratios, gap_ns):
    n = len(records)
    a, b = int(n*ratios[0]), int(n*(ratios[0]+ratios[1]))
    groups = {'train': records[:a], 'val': records[a:b], 'test': records[b:]}
    previous_end = None
    for name, group in groups.items():
        kept = []
        for r in group:
            if previous_end is not None and r['start_ns'] <= previous_end + gap_ns:
                r['split'] = 'excluded_gap'
            else:
                r['split'] = name
                kept.append(r)
        if not kept:
            raise ValueError(f'{name} has no samples after temporal gap; revise split/gap')
        previous_end = max(r['end_ns'] for r in kept)
    return records


def preserve_reference_splits(records, reference_path, gap_ns):
    """Keep existing held-out assignments for a corrected subset, never guess."""
    with reference_path.open() as f:
        reference_rows = list(csv.DictReader(f))
    reference = {}
    allowed = {'train', 'val', 'test', 'excluded_gap'}
    for row in reference_rows:
        identity = (str(Path(row['source_path']).resolve()), row['source_sample_id'])
        if identity in reference or row['split'] not in allowed:
            raise ValueError(f'{reference_path}: duplicate identity or invalid split')
        reference[identity] = row
    for record in records:
        identity = (record['source_path'], record['source_sample_id'])
        previous = reference.get(identity)
        if previous is None:
            raise ValueError(f'{record["sample_id"]}: not in reference splits; '
                             'this option requires a corrected subset of the original dataset')
        if any(int(previous[key]) != record[key] for key in ('start_ns', 'end_ns')):
            raise ValueError(f'{record["sample_id"]}: reference timestamps differ')
        record['split'] = previous['split']
    for source_path in {r['source_path'] for r in records}:
        previous_end = None
        for name in ('train', 'val', 'test'):
            group = [r for r in records if r['source_path'] == source_path and r['split'] == name]
            if not group:
                continue
            if previous_end is not None and min(r['start_ns'] for r in group) <= previous_end + gap_ns:
                raise ValueError(f'{source_path}: reference splits violate temporal gap')
            previous_end = max(r['end_ns'] for r in group)
    for name in ('train', 'val', 'test'):
        if not any(r['split'] == name for r in records):
            raise ValueError(f'{name}: reference split has no retained samples')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', required=True, type=Path)
    parser.add_argument('--source', required=True, type=Path, action='append',
                        help='Extractor directory; repeat --source for multiple recordings')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--ratios', nargs=3, type=float, default=[.7,.15,.15])
    parser.add_argument('--gap-seconds', type=float, default=1.0,
                        help='Extra gap between source windows across splits; default 1s')
    parser.add_argument('--reference-splits', type=Path,
                        help='Original samples.csv: preserve splits when correcting/deleting frames; '
                             'all retained frames must match the reference source and timestamps')
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    if not all(math.isfinite(v) and v > 0 for v in args.ratios) or not math.isclose(sum(args.ratios),1):
        parser.error('ratios must be positive and sum to 1')
    if not math.isfinite(args.gap_seconds) or args.gap_seconds < 0:
        parser.error('gap-seconds must be finite and nonnegative')
    if args.output.exists():
        parser.error('output must be a new directory')
    sources = []
    seen_paths, seen_bags = set(), set()
    for index, source in enumerate(args.source):
        source = source.resolve()
        manifest = json.loads((source/'manifest.json').read_text())
        if source in seen_paths:
            parser.error(f'duplicate source directory: {source}')
        seen_paths.add(source)
        if manifest.get('status') != 'complete':
            parser.error(f'{source}: source export is incomplete')
        if manifest.get('features') != ['x','y','z','intensity'] or manifest.get('dtype') != '<f4':
            parser.error(f'{source}: source must be float32 XYZI')
        if not manifest.get('standard'):
            parser.error(f'{source}: missing standard metadata')
        if sources and manifest['standard'] != sources[0]['manifest']['standard']:
            parser.error(f'{source}: incompatible preprocessing standard')
        bag = Path(manifest.get('bag', '')).name
        if not bag or bag in seen_bags:
            parser.error(f'{source}: missing or repeated bag identity; avoid duplicate recordings')
        seen_bags.add(bag)
        with (source/'frames.csv').open() as f:
            rows = list(csv.DictReader(f))
        mapping = {r['sample_id']: r for r in rows}
        if len(mapping) != len(rows):
            parser.error(f'{source}: duplicate source IDs')
        sources.append(dict(key=f'src{index:02d}', path=source,
                            manifest=manifest, mapping=mapping))
    manifest = sources[0]['manifest']
    files = sorted(args.export.glob('*/ann/*.pcd.json'))
    if not files:
        parser.error('no Sly annotations found')
    records, ids, prepared = [], set(), []
    for i, ann in enumerate(files):
        sid = ann.name.removesuffix('.pcd.json')
        pcd = ann.parent.parent/'pointcloud'/f'{sid}.pcd'
        cloud = read_pcd(pcd)
        # CVAT may append numeric suffixes to resolve exported name collisions.
        # Names only select candidates: require a unique XYZI match across all
        # candidates, including the original name, before choosing a source.
        candidate_ids = [sid]
        base = sid
        while '_' in base:
            stem, suffix = base.rsplit('_', 1)
            if not stem or not suffix.isascii() or not suffix.isdigit():
                break
            candidate_ids.append(stem)
            base = stem
        matches = []
        for source in sources:
            for candidate_id in candidate_ids:
                row = source['mapping'].get(candidate_id)
                if row is None:
                    continue
                binary = source['path']/'points'/f'{candidate_id}.bin'
                points = np.fromfile(binary, dtype='<f4')
                if points.size % 4:
                    raise ValueError(f'{binary}: malformed XYZI')
                points = points.reshape(-1, 4)
                if not np.isfinite(points).all() or len(points) != int(row['saved_points']):
                    raise ValueError(f'{binary}: invalid source points/count')
                if cloud.shape == points.shape and np.allclose(cloud, points, rtol=1e-6, atol=1e-6):
                    matches.append((source, row, binary, candidate_id))
        if len(matches) != 1:
            raise ValueError(f'{ann}: expected exactly one source XYZI match, got {len(matches)}; '
                             f'candidate IDs: {candidate_ids}; '
                             'missing or ambiguous clouds must be resolved, not guessed')
        source, row, binary, source_sid = matches[0]
        identity = (source['key'], source_sid)
        if identity in ids:
            raise ValueError(f'{ann}: same source frame exported more than once')
        ids.add(identity)
        output_id = f"{source['key']}_{source_sid}" if len(sources) > 1 else source_sid
        end = int(row['header_timestamp_ns'])
        start = int(row['window_start_ns'])
        if not 0 < start <= end:
            raise ValueError(f'{sid}: invalid time window')
        boxes = read_boxes(ann)
        records.append(dict(sample_id=output_id,source_key=source['key'],
                            source_sample_id=source_sid,export_sample_id=sid,
                            source_path=str(source['path']),
                            annotation_path=str(ann.resolve()),start_ns=start,end_ns=end,
                            points=len(cloud),boxes=len(boxes)))
        prepared.append((output_id,binary,ann))
        if (i+1)%100==0:
            print(f'Validated {i+1}/{len(files)}',flush=True)
    records.sort(key=lambda r:(r['end_ns'],r['sample_id']))
    for source in sources:
        group = [r for r in records if r['source_key'] == source['key']]
        if not group:
            raise ValueError(f"{source['path']}: no matching exported frames")
        if any(b['end_ns']<=a['end_ns'] for a,b in zip(group,group[1:])):
            raise ValueError(f"{source['path']}: duplicate/nonincreasing sample timestamps")
        if args.reference_splits is None:
            split_records(group,args.ratios,round(args.gap_seconds*1e9))
    if args.reference_splits is not None:
        preserve_reference_splits(records, args.reference_splits, round(args.gap_seconds*1e9))
    report = dict(status='validated',export=str(args.export.resolve()),
                  source=str(sources[0]['path']) if len(sources)==1 else None,
                  source_manifest=manifest,
                  sources=[dict(key=s['key'],path=str(s['path']),manifest=s['manifest']) for s in sources],
                  split_policy='chronological within each source, with temporal gap',
                  source_manifest_note='First source for legacy infos compatibility; all standards verified equal',
                  ratios=args.ratios,gap_seconds=args.gap_seconds,
                  label_format='center_x center_y center_z dx dy dz yaw_radians person',
                  label_mapping='Sly position=center, dimensions=local XYZ, rotation.z=yaw; verify overlays',
                  warnings=['Single recording split does not prove generalization to new scenes',
                            'Current loader needs empty-box shape (0,7) and negative resampling changes',
                            'Check gt_sampling policy before training',
                            'No PKL, model config, loader modifications or training performed'])
    if args.reference_splits is not None:
        report.update(reference_splits=str(args.reference_splits.resolve()),
                      split_policy='preserved source/timestamp-verified assignments from reference samples.csv',
                      ratios=None,
                      ratios_note='Not recomputed: deletions may change actual split proportions')
    report['splits']={}
    for name in ('train','val','test','excluded_gap'):
        group=[r for r in records if r['split']==name]
        report['splits'][name]=dict(samples=len(group),boxes=sum(r['boxes'] for r in group),
                                    negatives=sum(r['boxes']==0 for r in group))
    print(json.dumps(report['splits'],indent=2))
    if not args.write:
        print('DRY RUN passed. Add --write to create dataset; no output files written.')
        return
    args.output.mkdir(parents=True,exist_ok=False)
    for directory in ('points','labels','ImageSets'):
        (args.output/directory).mkdir()
    report['status']='incomplete'
    (args.output/'preparation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    # Keep gap-excluded files for auditing, but omit them from all split lists.
    for sid,binary,ann in prepared:
        points=np.fromfile(binary,dtype='<f4').reshape(-1,4)
        np.save(args.output/'points'/f'{sid}.npy',points,allow_pickle=False)
        boxes=read_boxes(ann)
        text=''.join(' '.join(format(float(v),'.9g') for v in box)+' person\n' for box in boxes)
        (args.output/'labels'/f'{sid}.txt').write_text(text)
    for name in ('train','val','test'):
        (args.output/'ImageSets'/f'{name}.txt').write_text(''.join(r['sample_id']+'\n' for r in records if r['split']==name))
    with (args.output/'samples.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    report['status']='complete'
    (args.output/'preparation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Prepared {len(records)} samples at {args.output}. Review before training.')


if __name__=='__main__':
    main()
