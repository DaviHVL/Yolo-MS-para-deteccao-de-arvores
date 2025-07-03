# Herda a configuração base do YOLO-MS XS para COCO e o runtime padrão.
_base_ = [
    '../yoloms-xs_syncbn_fast_8xb32-300e_coco.py',
]

# --- Configurações do Dataset ---

# 1. Caminho raiz do seu dataset
#    Importante: Mantenha este caminho relativo ao seu ambiente de execução,
#    ou forneça o caminho absoluto. Ele deve apontar para a pasta que contém
#    os diretórios 'rgb/' (imagens) e 'annotations/' (JSONs COCO).
#    No seu script de treinamento, 'data_root' é definido em 'setCFG',
#    garantindo que o caminho correto seja passado.
data_root = '../../data/' # Este é um valor padrão, será sobrescrito pelo setCFG.

# 2. Definição das classes e metadados
#    Estamos trabalhando com uma única classe: 'tree'.
class_name = ('tree',)
num_classes = len(class_name)
metainfo = {
    'classes': class_name,
    # Uma paleta de cores para visualização das caixas delimitadoras.
    # Usamos verde oliva para 'tree'.
    'palette': [(107, 142, 35)]
}

# 3. Configuração dos dataloaders para treino, validação e teste
#    Eles apontam para os arquivos JSON de anotação e as pastas de imagem.

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root=data_root, # Caminho raiz do dataset.
        metainfo=metainfo,   # Informações sobre as classes.
        ann_file='annotations/fold_0/instances_train.json', # Arquivo de anotações para treino.
        data_prefix=dict(img='rgb/'), # Subpasta onde as imagens estão localizadas.
        # Filtra caixas delimitadoras vazias e define um tamanho mínimo para as detecções.
        # Ajuste 'min_size' se suas árvores forem muito pequenas (ex: 10x10 pixels).
        filter_cfg=dict(filter_empty_gt=True, min_size=32)
    )
)

val_dataloader = dict(
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/fold_0/instances_val.json', # Arquivo de anotações para validação.
        data_prefix=dict(img='rgb/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32)
    )
)

# O dataloader de teste geralmente usa a mesma configuração do dataloader de validação.
test_dataloader = val_dataloader


# --- Modificações do Modelo ---

# 4. Ajustar o número de classes na cabeça da rede (bbox_head)
#    O modelo base foi pré-treinado no dataset COCO (80 classes).
#    Precisamos reconfigurá-lo para detectar nossa única classe ('tree').
model = dict(
    bbox_head=dict(
        head_module=dict(
            num_classes=num_classes # Define o número de classes conforme nosso dataset (1).
        )
    )
)

# --- Modificações do Treinamento e Avaliação ---

# 5. Ajustar a agenda de treinamento
#    O _base_ prevê 300 épocas, o que é excessivo para um dataset com apenas uma classe.
#    Reduzimos para 100 épocas e validamos a cada 5 épocas para monitorar o progresso.
train_cfg = dict(max_epochs=100, val_interval=5)

# 6. Modificar os avaliadores de validação e teste
#    Eles precisam apontar para o nosso arquivo de anotação de validação.
val_evaluator = dict(ann_file=data_root + 'annotations/fold_0/instances_val.json')
test_evaluator = val_evaluator

# 7. Otimizador (ajuste opcional de Learning Rate)
#    A configuração base usa AdamW com learning rate 0.001. Para datasets menores,
#    muitas vezes é benéfico reduzir o learning rate. Aqui, ajustamos para 0.0005.
optim_wrapper = dict(
    optimizer=dict(
        lr=5e-4 # Learning rate de 0.0005.
    )
)

# 8. Hooks de Runtime (comportamento durante o treinamento)
#    Configuramos um 'CheckpointHook' para salvar o modelo em intervalos regulares
#    e, mais importante, o "melhor" modelo baseado na métrica mAP@0.50 (mAP_50),
#    que é uma métrica comum para detecção de objetos (quanto maior, melhor).
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,  # Salva um checkpoint a cada 10 épocas.
        save_best='coco/bbox_mAP_50', # Critério para salvar o melhor modelo.
        rule='greater' # Um mAP maior é considerado melhor.
    )
)

# Visualização (opcional, útil para depuração)
# Descomente o bloco abaixo para habilitar a visualização e salvar imagens
# de predição durante a validação. Isso ajuda a entender o desempenho do modelo.
# visualizer = dict(
#     vis_backends = [dict(type='LocalVisBackend')],
#     name='visualizer'
# )
# default_hooks.update(
#     visualization=dict(type='DetVisualizationHook', draw=True, interval=1)
# )