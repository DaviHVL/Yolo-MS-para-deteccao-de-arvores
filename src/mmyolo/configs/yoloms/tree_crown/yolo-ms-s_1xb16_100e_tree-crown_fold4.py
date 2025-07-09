# Base de configuração herdada do modelo YOLO-MS XS treinado no COCO
_base_ = [
    '../yoloms-xs_syncbn_fast_8xb32-300e_coco.py',
]

# Caminho raiz dos dados
data_root = '../../data/'

# Definição da única classe do dataset
class_name = ('tree',)
num_classes = 1

# Metainformações usadas pelo dataloader e visualizador
metainfo = {
    'classes': class_name,
    'palette': [(107, 142, 35)]  # verde oliva
}

# Modificações do modelo para refletir o número correto de classes 
model = dict(
    bbox_head=dict(
        head_module=dict(
            num_classes=num_classes # número de classes na cabeça do detector
        )
    ),
    train_cfg=dict(
        assigner=dict(num_classes=num_classes) # configuração do assigner
    )
)

# Configuração do dataloader de treino
train_dataloader = dict(
    batch_size=4,       # tamanho do batch de treinamento
    num_workers=2,      # número de threads para leitura paralela
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/fold_4/instances_train.json',  # anotações do fold 4
        data_prefix=dict(img='rgb/'),                        # pasta onde estão as imagens
    )
)

# Configuração do dataloader de validação 
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/fold_4/instances_val.json',     # anotações de validação
        data_prefix=dict(img='rgb/'),
        test_mode=True     # habilita modo de inferência
    )
)

# Configuração do dataloader de teste
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/fold_4/instances_test.json',    # anotações de teste
        data_prefix=dict(img='rgb/'),
        test_mode=True
    )
)

# Configuração do loop de treinamento
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=100,
    val_interval=10
)

# Avaliador para validação usando métricas COCO
val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + 'annotations/fold_4/instances_val.json',
    metric='bbox',      # avaliação baseada em bounding boxes
    format_only=False,
    classwise=True,     # exibe métricas por classe
)

# Avaliador para teste usando métricas COCO
test_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + 'annotations/fold_4/instances_test.json',
    metric='bbox',
    format_only=False,
    classwise=True,
)

# Configuração do otimizador
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',   # otimizador AdamW
        lr=5e-4         # taxa de aprendizado inicial
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

# Caminho para o checkpoint de pesos pré-treinados
load_from = 'configs/yoloms/tree_crown/yoloms-xs_pre_trained.pth'


