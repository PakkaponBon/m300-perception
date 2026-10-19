#!/usr/bin/env python3
"""Create infos for prepared M300 data without training or GT sampling."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    root = parser.parse_args().data
    report = json.loads((root/'preparation_report.json').read_text())
    if report['status'] != 'complete' or report['source_manifest']['accumulation_seconds'] != 0.3:
        raise ValueError('Expected complete decay 0.3 dataset')
    all_ids = set()
    outputs = {}
    for split in ('train','val','test'):
        output = root/f'custom_infos_{split}.pkl'
        if output.exists():
            raise FileExistsError(f'Refusing overwrite: {output}')
        ids = (root/'ImageSets'/f'{split}.txt').read_text().split()
        if not ids or len(set(ids)) != len(ids) or all_ids.intersection(ids):
            raise ValueError(f'Empty/duplicate/overlapping split {split}')
        all_ids.update(ids)
        infos=[]
        for sid in ids:
            points=np.load(root/'points'/f'{sid}.npy',allow_pickle=False,mmap_mode='r')
            if points.ndim!=2 or points.shape[1]!=4 or not np.isfinite(points).all():
                raise ValueError(f'{sid}: invalid XYZI points')
            boxes=[]; names=[]
            for line in (root/'labels'/f'{sid}.txt').read_text().splitlines():
                values=line.split()
                if not values:
                    continue
                if len(values)!=8 or values[-1]!='person':
                    raise ValueError(f'{sid}: invalid label')
                boxes.append([float(x) for x in values[:7]]); names.append(values[-1])
            boxes=np.asarray(boxes,dtype=np.float32).reshape(-1,7)
            if not np.isfinite(boxes).all() or (boxes[:,3:6]<=0).any():
                raise ValueError(f'{sid}: invalid boxes')
            infos.append({'point_cloud':{'num_features':4,'lidar_idx':sid},
                          'annos':{'name':np.array(names,dtype=object),'gt_boxes_lidar':boxes}})
        outputs[output]=infos
    for output,infos in outputs.items():
        with output.open('xb') as stream:
            pickle.dump(infos,stream)
        negatives=sum(len(x['annos']['name'])==0 for x in infos)
        print(f'{output.name}: {len(infos)} samples, {negatives} negatives')


if __name__=='__main__':
    main()
