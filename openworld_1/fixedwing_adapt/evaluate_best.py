"""Evaluate the validation-selected checkpoint on the original mixed test set."""
import argparse
import json
from pathlib import Path

from mmengine import Config
from mmengine.runner import Runner
from mmengine.evaluator import DumpResults
from mmyolo.utils import register_all_modules


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    weights = list(run.glob('best_coco_fixedwing_ap50_iter_*.pth'))
    if len(weights) != 1:
        raise RuntimeError(f'Expected exactly one validation-selected weight, found {weights}')
    config = run / 'finetune.py'
    if not config.exists():
        raise FileNotFoundError(config)
    register_all_modules()
    cfg = Config.fromfile(str(config))
    test_dir = run.parents[1] / 'test_results' / run.name
    cfg.work_dir = str(test_dir)
    cfg.load_from = str(weights[0])
    cfg.resume = False
    runner = Runner.from_cfg(cfg)
    runner.test_evaluator.metrics.append(DumpResults(out_file_path=str(test_dir/'predictions.pkl')))
    metrics = runner.test()
    ap50 = float(metrics['coco/fixedwing_ap50'])
    report = dict(checkpoint=str(weights[0]), test_dir=str(test_dir),
                  selection='maximum fixedwing AP50 on original val; never selected on test',
                  novel_training_images=10, novel_training_instances=10,
                  extra_novel_validation_images=55, test_images=938,
                  historical_baseline_ap50=.395, target_ap50=.55,
                  fixedwing_ap50=ap50, target_met=ap50 >= .55,
                  metrics=metrics)
    (test_dir/'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
