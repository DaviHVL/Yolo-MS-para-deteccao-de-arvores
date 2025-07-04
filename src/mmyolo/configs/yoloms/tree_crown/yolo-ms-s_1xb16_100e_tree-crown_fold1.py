_base_ = [
    '../yoloms-xs_syncbn_fast_8xb32-300e_coco.py',
]

data_root = '../../data/'

class_name = ('tree',)
num_classes = 1
metainfo = {
    'classes': class_name,
    'palette': [(107, 142, 35)]  # verde oliva
}

model = dict(
    bbox_head=dict(
        head_module=dict(
            num_classes=num_classes
        )
    ),
    train_cfg=dict(
        assigner=dict(num_classes=num_classes)
    )
)

# Agora a pasta das imagens mudou para "images/train", "images/val", "images/test"
train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/fold_1/instances_train.json',
        data_prefix=dict(img='rgb/'),
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/fold_1/instances_val.json',
        data_prefix=dict(img='rgb/'),
        test_mode=True
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/fold_1/instances_test.json',
        data_prefix=dict(img='rgb/'),
        test_mode=True
    )
)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=100,
    val_interval=10
)

val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + 'annotations/fold_1/instances_val.json',
    metric='bbox',
    format_only=False,
    classwise=True,
)

test_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + 'annotations/fold_1/instances_test.json',
    metric='bbox',
    format_only=False,
    classwise=True,
)

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=5e-4
    )
)

# --- Hook de Early Stopping ---
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP_50',
        rule='greater',         # a métrica deve crescer (mAP maior é melhor)
        patience=10,            # número de épocas sem melhora antes de parar
        min_delta=0.001,        # mudança mínima para considerar "melhoria"
    )
]

load_from = 'configs/yoloms/tree_crown/yoloms-xs_pre_trained.pth'



