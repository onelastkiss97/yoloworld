"""Reparameterized inference config for Sparse-Zoom YOLO-World V2-M."""

_base_ = './sparse_zoom_yoloworld_v2_m_drone_1280.py'

num_classes = 1
text_channels = 512
widen_factor = 0.75
last_stage_out_channels = 768
strides = [4, 8, 16, 32]

# Override this on the command line when testing another converted checkpoint.
load_from = (
    '/home/songziqi/project/YOLO-World/newwork/weights/'
    'sparse_zoom_yoloworld_v2_m_drone_1280_rep.pth')

model = dict(
    reparameterized=True,
    embedding_path='',
    neck=dict(
        reparameterized=True,
        num_prompts=num_classes,
        guide_channels=num_classes,
        block_cfg=dict(
            _delete_=True,
            type='RepConvMaxSigmoidCSPLayerWithTwoConv',
            guide_channels=num_classes)),
    bbox_head=dict(
        type='YOLOv8Head',
        head_module=dict(
            _delete_=True,
            type='RepYOLOWorldHeadModule',
            num_classes=num_classes,
            num_guide=num_classes,
            embed_dims=text_channels,
            in_channels=[128, 256, 512, last_stage_out_channels],
            widen_factor=widen_factor,
            reg_max=16,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True),
            featmap_strides=strides)))

