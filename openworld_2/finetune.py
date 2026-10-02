"""Five-shot counterpart to the fixed 10-shot experiment."""
_base_ = '../openworld_1/fixedwing_adapt/finetune.py'
exp_root = '/home/songziqi/project/YOLO-World/openworld_2'
train_dataloader = dict(dataset=dict(dataset=dict(ann_file=f'{exp_root}/data/train_5shot.json')))
val_dataloader = dict(dataset=dict(dataset=dict(ann_file=f'{exp_root}/data/val.json')))
val_evaluator = dict(ann_file=f'{exp_root}/data/val.json')
work_dir = f'{exp_root}/runs/neck_5shot'
