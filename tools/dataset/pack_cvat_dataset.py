#!/usr/bin/env python3
"""Package extracted XYZI .bin samples for CVAT without modifying inputs.

Accepts an extractor output directory or its points/ directory. Repeat --input
to combine sources without concatenating their temporal windows. Multi-source
filenames append _00, _01, ... so prepare_m300_dataset.py can remove numeric
suffixes and verify the original source by XYZI content. A companion
<output>.manifest.json records every archive name and original source file.
--sample-every N selects every Nth existing sample within EACH source; it
does not recompute points, change accumulation, or truncate to the first clip.
Uses only the Python standard library. No ROS environment is required.
"""
import argparse
import json
import math
from pathlib import Path
import struct
from zipfile import ZipFile, ZIP_DEFLATED


def validate(path):
    size = path.stat().st_size
    if size == 0 or size % 16:
        raise ValueError(f'{path.name}: expected nonempty float32 Nx4 (16 bytes/point)')
    with path.open('rb') as stream:
        while chunk := stream.read(16 * 65536):
            if len(chunk) % 16:
                raise ValueError(f'{path.name}: incomplete point')
            for point in struct.iter_unpack('<ffff', chunk):
                if not all(map(math.isfinite, point)):
                    raise ValueError(f'{path.name}: contains NaN/Inf')
    return size // 16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, action='append',
                        help='Dataset directory or points directory; repeat to combine sources')
    parser.add_argument('--output', required=True, type=Path,
                        help='New .zip file; never overwrites an existing file')
    parser.add_argument('--max-files', type=int, default=0,
                        help='First N files in input/filename order across sources; 0 means all')
    parser.add_argument('--sample-every', type=int, default=1,
                        help='Take every Nth existing sample per source; stride5 input with N=6 gives stride30')
    args = parser.parse_args()
    if args.max_files < 0:
        parser.error('max-files must be >= 0')
    if args.sample_every < 1:
        parser.error('sample-every must be >= 1')
    mapping_output = args.output.with_suffix('.manifest.json')
    if args.output.suffix.lower() != '.zip' or args.output.exists() or mapping_output.exists():
        parser.error('Output must be a new .zip file')
    sources, files, seen_paths = [], [], set()
    reference_standard = None
    multiple = len(args.input) > 1
    for index, input_path in enumerate(args.input):
        points_dir = (input_path / 'points' if (input_path / 'points').is_dir() else input_path).resolve()
        if not points_dir.is_dir():
            parser.error(f'Input directory does not exist: {points_dir}')
        if points_dir in seen_paths:
            parser.error(f'Duplicate input directory: {points_dir}')
        seen_paths.add(points_dir)
        manifest = points_dir.parent / 'manifest.json'
        metadata = json.loads(manifest.read_text()) if manifest.is_file() else None
        if metadata is not None:
            if metadata.get('status') != 'complete':
                parser.error(f'Dataset manifest is not complete: {manifest}')
            if metadata.get('features') != ['x', 'y', 'z', 'intensity'] or metadata.get('dtype') != '<f4':
                parser.error(f'Dataset must have little-endian float32 XYZI features: {manifest}')
        if multiple:
            if not metadata or not metadata.get('standard'):
                parser.error('Combining sources requires complete extractor manifests with standards')
            if reference_standard is not None and metadata['standard'] != reference_standard:
                parser.error('Cannot combine incompatible preprocessing standards')
            reference_standard = metadata['standard']
        source_files = sorted(points_dir.glob('*.bin'))
        if not source_files:
            parser.error(f'No .bin files found: {points_dir}')
        original_stride = metadata.get('stride') if metadata else None
        regular = (isinstance(original_stride, int) and not isinstance(original_stride, bool)
                   and original_stride > 0 and all(f.stem.isascii() and f.stem.isdigit() for f in source_files)
                   and all(int(b.stem) - int(a.stem) == original_stride
                           for a, b in zip(source_files, source_files[1:])))
        sources.append(dict(index=index, points_dir=str(points_dir), manifest=metadata,
                            available_samples=len(source_files), sample_every=args.sample_every,
                            effective_source_stride=original_stride * args.sample_every if regular else None))
        for file in source_files[::args.sample_every]:
            name = f'{file.stem}_{index:02d}.bin' if multiple else file.name
            files.append((index, file, f'velodyne_points/data/{name}'))
    if args.max_files:
        files = files[:args.max_files]
    names = [archive_name for _, _, archive_name in files]
    if len(names) != len(set(names)):
        parser.error('Duplicate archive filenames; refusing ambiguous package')
    total, mapping = 0, []
    for i, (source_index, file, archive_name) in enumerate(files, 1):
        points = validate(file)
        total += points
        mapping.append(dict(archive_name=archive_name, source_index=source_index,
                            source_sample_id=file.stem, source_file=str(file),
                            points=points, bytes=file.stat().st_size))
        if i % 100 == 0:
            print(f'Validated {i}/{len(files)} files', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects an existing archive even if it appeared during validation.
    zip_created = mapping_created = False
    try:
        with args.output.open('xb') as output:
            zip_created = True
            with ZipFile(output, 'w', compression=ZIP_DEFLATED, compresslevel=6,
                         allowZip64=True) as archive:
                for i, (_, file, archive_name) in enumerate(files, 1):
                    archive.write(file, arcname=archive_name)
                    if i % 100 == 0:
                        print(f'Packed {i}/{len(files)} files', flush=True)
        with mapping_output.open('x') as stream:
            mapping_created = True
            json.dump(dict(status='complete', output=str(args.output.resolve()),
                           samples=len(files), points=total, sources=sources, files=mapping,
                           sample_every=args.sample_every,
                           naming='numeric source suffix for multiple inputs; originals unchanged',
                           accumulation_note='Windows stay within each original bag; no cross-bag accumulation'),
                      stream, indent=2)
            stream.write('\n')
    except BaseException:
        # Remove only outputs created by THIS run, never pre-existing artifacts.
        if zip_created:
            args.output.unlink()
        if mapping_created:
            mapping_output.unlink()
        raise
    print(f'Created {args.output}: {len(files)} samples, {total} points')
    print(f'Source/filename mapping: {mapping_output}')
    print('Upload this ZIP when creating a new CVAT task.')


if __name__ == '__main__':
    main()
