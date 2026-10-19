#!/usr/bin/env python3
"""Smoke-test data and random-weight model. No optimizer step/checkpoint/training."""
import argparse
import logging
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--openpcdet', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    root = args.openpcdet.expanduser().resolve()
    data = args.data.expanduser().resolve()
    sys.path.insert(0, str(root))
    os.chdir(root / 'tools')
    import numpy as np
    import torch
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.datasets.custom.torch_eval_overlap import rotate_overlap

    np.random.seed(666)
    torch.manual_seed(666)
    cfg_from_yaml_file('cfgs/custom_models/pointpillar_m300_decay03.yaml', cfg)
    cfg.DATA_CONFIG.DATA_PATH = str(data)
    logger = logging.getLogger('m300_check')
    ds, _, _ = build_dataloader(cfg.DATA_CONFIG, cfg.CLASS_NAMES, 2, False,
                                workers=0, logger=logger, training=True)
    negatives = [i for i,x in enumerate(ds.custom_infos) if not len(x['annos']['name'])]
    if len(negatives) < 2:
        raise ValueError('This optional mixed/empty batch check requires at least two negatives')
    positive = next(i for i,x in enumerate(ds.custom_infos) if len(x['annos']['name']))
    for i in range(len(ds)):
        sample = ds[i]
        assert sample['frame_id'] == ds.sample_id_list[i]
        assert np.isfinite(sample['points']).all()
        assert len(sample['voxels']) > 0
        if i in negatives:
            assert sample['gt_boxes'].shape == (0, 8)
    print(f'PASS all {len(ds)} training samples, including {len(negatives)} negatives', flush=True)
    model = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), ds).cuda().train()
    for name, indices in [('mixed', [negatives[0], positive]), ('empty', negatives[:2])]:
        batch = ds.collate_batch([ds[i] for i in indices])
        load_data_to_gpu(batch)
        model.zero_grad(set_to_none=True)
        ret, _, _ = model(batch)
        loss = ret['loss'].mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        print(f'PASS {name} forward/backward: loss={loss.item():.6f}', flush=True)
    val, _, _ = build_dataloader(cfg.DATA_CONFIG, cfg.CLASS_NAMES, 1, False,
                                 workers=0, logger=logger, training=False)
    model.eval()
    batch = val.collate_batch([val[0]])
    load_data_to_gpu(batch)
    with torch.no_grad():
        predictions, _ = model(batch)
    assert len(predictions) == 1
    print('PASS validation inference', flush=True)
    a = np.array([[0,0,2,1,0]],dtype=np.float32)
    b = np.array([[0,0,2,1,0],[0,0,2,1,np.pi/2],[10,10,2,1,0]],dtype=np.float32)
    np.testing.assert_allclose(rotate_overlap(a,b),[[1,1/3,0]],atol=1e-5)
    synthetic = [dict(name=x['annos']['name'].copy(),
                      boxes_lidar=x['annos']['gt_boxes_lidar'].copy(),
                      score=np.ones(len(x['annos']['name']))) for x in val.custom_infos]
    _, metrics = val.evaluation(synthetic, cfg.CLASS_NAMES, eval_metric='kitti')
    overlap_scores = [v for k,v in metrics.items() if '3d' in k or 'bev' in k]
    assert overlap_scores and all(v > 99.9 for v in overlap_scores)
    print('PASS evaluator synthetic perfect-box check; this is NOT model accuracy')


if __name__ == '__main__':
    main()
