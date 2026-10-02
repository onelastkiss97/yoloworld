_base_ = '../yolo_world_v2_m_openworld_standard_1280.py'
custom_imports = dict(imports=['yolo_world', 'openworld_1.standard_ovd_components',
                              'openworld_1.fixedwing_adapt.components'], allow_failed_imports=False)
classes = ('multirotor drone', 'helicopter', 'fixed-wing aircraft')
model = dict(num_test_classes=3)
test_dataloader = dict(dataset=dict(
    class_text_path='/home/songziqi/project/Data/openworld_base/texts/openworld_class_texts.json',
    dataset=dict(ann_file='/home/songziqi/project/Data/openworld_base/annotations/instances_test_openworld.json',
                 data_prefix=dict(img='test/images/'), metainfo=dict(classes=classes), test_mode=True)))
test_evaluator = dict(_delete_=True, type='FixedWingCocoMetric', classwise=True,
    proposal_nums=(100, 1, 10), metric='bbox',
    ann_file='/home/songziqi/project/Data/openworld_base/annotations/instances_test_openworld.json')
