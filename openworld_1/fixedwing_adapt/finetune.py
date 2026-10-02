_base_ = './baseline.py'
model = dict(type='FewShotYOLOWorldDetector', num_train_classes=3, num_test_classes=3,
             backbone=dict(image_model=dict(frozen_stages=4, norm_eval=True)),
             bbox_head=dict(head_module=dict(num_classes=3)),
             train_cfg=dict(assigner=dict(num_classes=3)))
load_from = '/home/songziqi/project/YOLO-World/openworld_1/runs/v2_m_ovd_standard_base2_novel_fixedwing_1280_e150/best_coco_bbox_mAP_epoch_100.pth'
exp_root = '/home/songziqi/project/YOLO-World/openworld_1/fixedwing_adapt'
classes = ('multirotor drone', 'helicopter', 'fixed-wing aircraft')
train_pipeline = [
    dict(type='LoadImageFromFile'), dict(type='LoadAnnotations', with_bbox=True),
    dict(type='YOLOv5KeepRatioResize', scale=(1280, 1280)),
    dict(type='LetterResize', scale=(1280, 1280), allow_scale_up=True, pad_val=dict(img=114.)),
    dict(type='YOLOv5RandomAffine', max_rotate_degree=5., max_shear_degree=0.,
         max_translate_ratio=.05, scaling_ratio_range=(.8, 1.2), max_aspect_ratio=100., border_val=(114,114,114)),
    dict(type='YOLOv5HSVRandomAug'), dict(type='mmdet.RandomFlip', prob=.5),
    dict(type='RandomLoadText', num_neg_samples=(2,2), max_num_samples=3, padding_to_max=True, padding_value=''),
    dict(type='mmdet.PackDetInputs', meta_keys=('img_id','img_path','ori_shape','img_shape','flip','flip_direction','texts'))]
train_dataloader = dict(_delete_=True, batch_size=4, num_workers=4, persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True), dataset=dict(type='MultiModalDataset',
        dataset=dict(type='YOLOv5CocoDataset', data_root='', ann_file=f'{exp_root}/data/train_10shot.json',
                     data_prefix=dict(img=''), metainfo=dict(classes=classes), filter_cfg=dict(filter_empty_gt=False, min_size=32)),
        class_text_path='/home/songziqi/project/Data/openworld_base/texts/openworld_class_texts.json', pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(
    class_text_path='/home/songziqi/project/Data/openworld_base/texts/openworld_class_texts.json',
    dataset=dict(data_root='', ann_file=f'{exp_root}/data/val.json', data_prefix=dict(img=''),
                 metainfo=dict(classes=classes), test_mode=True)))
val_evaluator = dict(_delete_=True, type='FixedWingCocoMetric', classwise=True,
    proposal_nums=(100,1,10), metric='bbox', ann_file=f'{exp_root}/data/val.json')
train_cfg = dict(_delete_=True, type='IterBasedTrainLoop', max_iters=600, val_begin=100, val_interval=100)
param_scheduler = None
default_hooks = dict(param_scheduler=None,
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=100,
                    filename_tmpl='iter_{}.pth', max_keep_ckpts=2,
                    save_best='coco/fixedwing_ap50', rule='greater'),
    logger=dict(type='LoggerHook', interval=20))
custom_hooks = [dict(type='TrainableParameterAuditHook', allowed_prefixes=('neck.', 'bbox_head.')),
                dict(type='PositiveDetectionLossHook', check_iters=25)]
optim_wrapper = dict(_delete_=True, type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=3e-5, weight_decay=.01),
    clip_grad=dict(max_norm=10., norm_type=2),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
log_processor = dict(by_epoch=False, window_size=20)
randomness = dict(seed=20260912)
