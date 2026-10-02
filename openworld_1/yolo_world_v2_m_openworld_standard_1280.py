_base_ = ('../third_party/mmyolo/configs/yolov8/'
          'yolov8_m_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(imports=['yolo_world', 'openworld_1.standard_ovd_components'],
                      allow_failed_imports=False)

data_root = '/home/songziqi/project/Data/openworld_base'
ann_root = f'{data_root}/annotations'
text_root = f'{data_root}/texts'
classes = ('multirotor drone', 'helicopter')
metainfo = dict(classes=classes)
num_training_classes = 2
num_test_classes = 2
max_epochs = 150
close_mosaic_epochs = 10
save_epoch_intervals = 5
text_channels = 512
img_scale = (1280, 1280)
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-4
weight_decay = 0.05
train_batch_size_per_gpu = 4
load_from = ('/home/songziqi/project/YOLO-World/'
             'yolo_world_v2_m_obj365v1_goldg_pretrain_1280ft-77d0346d.pth')
text_model_name = ('/home/songziqi/.cache/huggingface/hub/'
                   'models--openai--clip-vit-base-patch32/snapshots/'
                   '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268')

model = dict(
    type='YOLOWorldDetector', mm_neck=True,
    num_train_classes=num_training_classes, num_test_classes=num_test_classes,
    data_preprocessor=dict(type='mmdet.DetDataPreprocessor', mean=[0, 0, 0],
                           std=[255, 255, 255], bgr_to_rgb=True,
                           pad_size_divisor=32),
    backbone=dict(_delete_=True, type='MultiModalYOLOBackbone',
                  image_model={{_base_.model.backbone}},
                  text_model=dict(type='HuggingCLIPLanguageBackbone',
                                  model_name=text_model_name,
                                  frozen_modules=['all'])),
    neck=dict(type='YOLOWorldPAFPN', guide_channels=text_channels,
              embed_channels=neck_embed_channels, num_heads=neck_num_heads,
              block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv')),
    bbox_head=dict(type='StandardYOLOWorldHead',
                   head_module=dict(type='YOLOWorldHeadModule', use_bn_head=True,
                                    embed_dims=text_channels,
                                    num_classes=num_training_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_training_classes)))

text_transform = [
    dict(type='RandomLoadText', num_neg_samples=(2, 2),
         max_num_samples=num_training_classes, padding_to_max=True, padding_value=''),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts'))]
pre_transform = _base_.pre_transform
mosaic_affine_transform = [
    dict(type='MultiModalMosaic', img_scale=img_scale, pad_val=114.0,
         pre_transform=pre_transform),
    dict(type='YOLOv5RandomAffine', max_rotate_degree=0.0, max_shear_degree=0.0,
         max_aspect_ratio=100.,
         scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),
         border=(-img_scale[0] // 2, -img_scale[1] // 2), border_val=(114, 114, 114))]
train_pipeline = [
    *pre_transform, *mosaic_affine_transform,
    dict(type='YOLOv5MultiModalMixUp', prob=0.15,
         pre_transform=[*pre_transform, *mosaic_affine_transform]),
    *_base_.last_transform[:-1], *text_transform]
train_pipeline_stage2 = [
    *pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(type='LetterResize', scale=img_scale, allow_scale_up=True,
         pad_val=dict(img=114.0)),
    dict(type='YOLOv5RandomAffine', max_rotate_degree=0.0,
         max_shear_degree=0.0,
         scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),
         max_aspect_ratio=100., border_val=(114, 114, 114)),
    *_base_.last_transform[:-1], *text_transform]

train_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(type='YOLOv5CocoDataset', data_root=data_root,
                 ann_file=f'{ann_root}/instances_train_base.json',
                 data_prefix=dict(img='train_base/images/'), metainfo=metainfo,
                 filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path=f'{text_root}/base_class_texts.json', pipeline=train_pipeline)
train_dataloader = dict(_delete_=True, batch_size=train_batch_size_per_gpu,
                        num_workers=8, persistent_workers=True,
                        sampler=dict(type='DefaultSampler', shuffle=True),
                        dataset=train_dataset)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(type='LetterResize', scale=img_scale, allow_scale_up=False,
         pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(type='LoadText'),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param', 'texts'))]
val_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(type='YOLOv5CocoDataset', data_root=data_root,
                 ann_file=f'{ann_root}/instances_val_base.json',
                 data_prefix=dict(img='val_base/images/'), metainfo=metainfo,
                 filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path=f'{text_root}/base_class_texts.json', pipeline=test_pipeline)
val_dataloader = dict(_delete_=True, batch_size=1, num_workers=4,
                      persistent_workers=True,
                      sampler=dict(type='DefaultSampler', shuffle=False),
                      dataset=val_dataset)
test_dataloader = val_dataloader

default_hooks = dict(
    param_scheduler=dict(scheduler_type='linear', lr_factor=0.01, max_epochs=max_epochs),
    checkpoint=dict(max_keep_ckpts=-1, save_best='coco/bbox_mAP', rule='greater',
                    interval=save_epoch_intervals))
custom_hooks = [
    dict(type='PositiveDetectionLossHook', check_iters=25),
    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0001,
         update_buffers=True, strict_load=False, priority=49),
    dict(type='mmdet.PipelineSwitchHook', switch_epoch=max_epochs - close_mosaic_epochs,
         switch_pipeline=train_pipeline_stage2)]
train_cfg = dict(max_epochs=max_epochs, val_interval=5,
                 dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                                     _base_.val_interval_stage2)])
optim_wrapper = dict(
    optimizer=dict(_delete_=True, type='AdamW', lr=base_lr,
                   weight_decay=weight_decay, batch_size_per_gpu=train_batch_size_per_gpu),
    paramwise_cfg=dict(custom_keys={'backbone.text_model': dict(lr_mult=0.01),
                                    'logit_scale': dict(weight_decay=0.0)}),
    constructor='YOLOWv5OptimizerConstructor')
val_evaluator = dict(_delete_=True, type='mmdet.CocoMetric', classwise=True,
                     proposal_nums=(100, 1, 10),
                     ann_file=f'{ann_root}/instances_val_base.json', metric='bbox')
