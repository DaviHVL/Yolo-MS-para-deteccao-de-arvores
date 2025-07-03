# configs/yolo_ms/yolo_ms_xs_tree_crown.py

from mmyolo.models.detectors import YOLODetector
from mmyolo.models.dense_heads import YOLOMSHead # Esta é a que está faltando
from mmyolo.models.necks import YOLOMSNeck
from mmyolo.models.backbones import YOLOMSBackbone
from mmyolo.models.layers import MSBlockBottleNeckLayer # Se você usar diretamente no config

# Herda de uma configuração base do YOLO-MS (verifique se o caminho está correto)
_base_ = './yoloms-xs_syncbn_fast_8xb32-300e_coco.py'

# --- DEFINIÇÕES DO DATASET ---
dataset_type = 'CocoDataset'
data_root = 'data/' # Garanta que a pasta 'data' está na raiz do projeto MMDetection

# --- CONFIGURAÇÃO DOS DATALOADERS (COM CAMINHOS CORRIGIDOS) ---
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # CORREÇÃO: Aponta para o JSON dentro do fold
        ann_file='annotations/fold_0/instances_train.json',
        # CORREÇÃO: Caminho completo para as imagens
        data_prefix=dict(img='individual_urban_tree_crown_detection/rgb/'),
        metainfo=dict(classes=('tree',))
    )
)

val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # CORREÇÃO: Aponta para o JSON dentro do fold
        ann_file='annotations/fold_0/instances_val.json',
        # CORREÇÃO: Caminho completo para as imagens
        data_prefix=dict(img='individual_urban_tree_crown_detection/rgb/'),
        metainfo=dict(classes=('tree',))
    )
)

test_dataloader = val_dataloader

# --- CONFIGURAÇÃO DO AVALIADOR (COM CAMINHO CORRIGIDO) ---
val_evaluator = dict(
    type='CocoMetric',
    # CORREÇÃO: Aponta para o JSON dentro do fold
    ann_file=data_root + 'annotations/fold_0/instances_val.json',
    metric='bbox'
)
test_evaluator = val_evaluator

# --- AJUSTE DO MODELO (COM ESTRUTURA CORRIGIDA) ---
model = dict(
    bbox_head=dict(
        type='YOLOMSHead', # É bom especificar o tipo para clareza
        num_classes=1      # CORREÇÃO: num_classes está diretamente aqui
    )
)

# --- AJUSTES DE TREINAMENTO (OPCIONAL) ---
# Altera o treinamento para 100 épocas e salva a cada 10.
train_cfg = dict(max_epochs=100, val_interval=10)

# Altera a frequência de salvamento dos checkpoints
default_hooks = dict(
    checkpoint=dict(interval=10)
)

