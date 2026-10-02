"""Test only the validation-selected five-shot checkpoint."""
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
    assert len(weights)==1, weights
    register_all_modules()
    cfg = Config.fromfile(str(run/'finetune.py'))
    test_dir = Path(__file__).resolve().parent/'test_results'/run.name
    cfg.work_dir, cfg.load_from, cfg.resume = str(test_dir), str(weights[0]), False
    runner = Runner.from_cfg(cfg)
    runner.test_evaluator.metrics.append(DumpResults(out_file_path=str(test_dir/'predictions.pkl')))
    metrics = runner.test()
    result = dict(checkpoint=str(weights[0]), novel_training_images=5, novel_training_instances=5,
                  extra_novel_validation_images=55, test_images=938,
                  selection='maximum original-val fixedwing AP50; no test-based selection',
                  fixedwing_ap50=float(metrics['coco/fixedwing_ap50']),
                  target_ap50=.55, target_met=float(metrics['coco/fixedwing_ap50'])>=.55,
                  metrics=metrics)
    (test_dir/'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__=='__main__':
    main()
