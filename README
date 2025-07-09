# Projeto 2 IIA - Yolo-MS para detecção de Árvores

**Autores**: 
* Caio Medeiros Balaniuk - 231025190
* Davi Henrique Vieira Lima - 231013529
* Lucca Schoen de Almeida - 231018900

**Curso**: Introdução à Inteligência Artificial

**Instituição**: Universidade de Brasília (UnB)

*Brasília, Julho de 2025*

---
<br>

# 🚀 Resumo do Projeto
Este trabalho tem como objetivo aplicar e avaliar a arquitetura **YOLO-MS** para a **detecção de copas de árvores individuais em imagens RGB de alta resolução de ambientes urbanos**. A análise é baseada na comparação dos resultados com os 21 métodos testados no artigo de Zamboni et al. (2021). Utilizamos a base de dados disponibilizada pelo artigo original e medimos o desempenho com as mesmas métricas propostas, demonstrando a eficácia da abordagem recente YOLO-MS.

---
<br>

## 🎯 Objetivo do projeto
Este projeto tem como principais objetivos:

- **Aplicar a arquitetura YOLO-MS** na tarefa de detecção de copas de árvores individuais em imagens RGB de alta resolução, obtidas em ambientes urbanos.

- **Avaliar o desempenho do modelo YOLO-MS** utilizando as mesmas métricas propostas no artigo de Zamboni et al. (2021), garantindo uma comparação justa e reprodutível.

- **Comparar os resultados obtidos com YOLO-MS** com os 21 métodos previamente analisados no benchmark de Zamboni et al., visando identificar vantagens e limitações da abordagem.

- **Utilizar o mesmo conjunto de dados** disponibilizado pelo trabalho original para garantir consistência na avaliação dos modelos.

- **Demonstrar a eficácia da abordagem YOLO-MS**, especialmente no que diz respeito à detecção em múltiplas escalas, mantendo alto desempenho e eficiência computacional.

---
<br>

## 📁 Estrutura Modular do Projeto

```text
Yolo-MS-para-deteccao-de-arvores/
├── data/   
│   └── annotations                 
│       ├── fold_0
│       ├── fold_1
│       ├── fold_2
│       ├── fold_3
│       └── fold_4
│   ...
│   └── convert_yolo_to_coco.py
│   ...
├── midia/
│   ├── Teste(Fold 0).png
│   ├── Teste(Fold 1).png
│   ├── Teste(Fold 2).png
│   ├── Teste(Fold 3).png
│   ├── Teste(Fold 4).png
│   └── Validação Cruzada.png     
├── src/
│   ...
│   └── mmyolo/
│       ...
│       └── configs/
│           ... 
│           └── yoloms/
│               ... 
│               └── tree_crown/
│                   ├── yolo-ms-s_1xb16_100e_tree-crown_fold0.py
│                   ├── yolo-ms-s_1xb16_100e_tree-crown_fold0.py
│                   ├── yolo-ms-s_1xb16_100e_tree-crown_fold0.py
│                   ├── yolo-ms-s_1xb16_100e_tree-crown_fold0.py
│                   ├── yolo-ms-s_1xb16_100e_tree-crown_fold0.py
│                   └── yoloms-xs_pre_trained.pth  
│   ...
└── README.md              
```

---
<br>

## 🛠️ Desenvolvimento do Projeto

O desenvolvimento deste projeto seguiu as seguintes etapas:

1. **Clonagem dos Repositórios**

Inicialmente, clonamos dois repositórios essenciais para o projeto:

- O repositório `individual_urban_tree_crown_detection`, que fornece o dataset com as imagens e anotações no formato YOLO. Este repositório foi adicionado à pasta `data/`.

- O repositório `YOLO-MS`, que fornece o código e modelos para treinamento com MMYOLO. Este foi colocado na pasta `src/`.

2. **Conversão para o Formato COCO**

Foi criado o script `convert_yolo_to_coco.py` para converter as anotações do dataset original no formato YOLO para o formato COCO.

Como resultado, foi gerada a pasta `annotations/`, contendo 5 subpastas — `fold_0`, `fold_1`, ..., `fold_4` — que representam os folds utilizados para validação cruzada.

Cada pasta contém três arquivos:

- `instances_train.json` — imagens de treino
- `instances_val.json` — imagens de validação
- `instances_test.json` — imagens de teste

3. **Configurações para Treinamento com MMYOLO**

No caminho `src/mmyolo/configs/yoloms/`, foi criada a subpasta `tree_crown/`, que contém 5 arquivos `.py`, cada um responsável pela configuração de um fold para:

- Treinamento
- Validação
- Teste

Esses arquivos adaptam o modelo YOLO-MS-XS para o dataset de detecção de copas de árvores urbanas.

4. **Pesos Pré-Treinados**

Também foi adicionado o arquivo `yoloms-xs_pre_trained.pth`, que contém os pesos pré-treinados do modelo YOLO-MS-XS, disponibilizado no Model Zoo do repositório YOLO-MS.

Esse arquivo é carregado antes do treinamento para acelerar o aprendizado e melhorar o desempenho inicial do modelo.

---
<br>

## 🧪 Como executar o projeto
Para executar este projeto, siga os passos abaixo:

1. **Instalação dos Requisitos**

Antes de mais nada, é necessário configurar o ambiente com todas as dependências corretas.

> ✅ **Siga as instruções descritas no arquivo [`install_mmyolo.md`](src/docs/install_mmyolo.md)**, localizado na pasta `src/docs`.

---

2. **Treinamento do Modelo**

Após configurar o ambiente, você pode treinar o modelo com o seguinte comando:

```bash
python tools/train.py {ARQUIVO_DE_CONFIGURAÇÃO}
```

🔹 Exemplo:

```bash
python tools/train.py configs/yoloms/tree_crown/yolo-ms-s_1xb16_100e_tree-crown_fold2.py
```

3. **Avaliação do Modelo**

Após o treinamento, para avaliar o modelo em um conjunto de teste:
```bash
python tools/test.py {ARQUIVO_DE_CONFIGURAÇÃO} {ARQUIVO_DE_CHECKPOINT}
```
🔹 Exemplo:
```bash
python tools/test.py configs/yoloms/tree_crown/yolo-ms-s_1xb16_100e_tree-crown_fold2.py work_dirs/yolo-ms-s_1xb16_100e_tree-crown_fold2/best_coco/tree_precision_epoch_100.pth
```
Esse comando testa o modelo utilizando o melhor checkpoint salvo durante o treinamento.

4. **Problemas com Bibliotecas**

Se ocorrer algum erro relacionado a bibliotecas ausentes ou incompatíveis, instale-as manualmente utilizando o pip:
```bash
pip install nome-da-biblioteca
```
🔹 Exemplo:
```bash
pip install pycocotools
```

Com esses passos, você poderá treinar, avaliar e ajustar o modelo YOLO-MS para o seu conjunto de dados.

---
<br>

## 🤝 Contribuições
Contribuições são bem-vindas! Sinta-se à vontade para:
- Reportar issues
- Sugerir melhorias
- Enviar pull requests

