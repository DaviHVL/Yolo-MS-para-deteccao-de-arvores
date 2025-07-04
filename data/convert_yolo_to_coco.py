import os
import json
from PIL import Image


# --------- CONFIGURAÇÕES ---------
IMG_DIR     = "rgb"           # pasta com os patches .png
LBL_DIR     = "bbox_txt"      # pasta com os .txt de cada imagem: x_min y_min x_max y_max
LIST_DIR    = "img_list"      # pasta com subpastas 0,1,2,3,4 contendo train.txt, val.txt, test.txt
OUT_DIR     = "annotations"   # saída COCO JSON
CLASSES     = ["tree"]        # única classe
FOLDS       = sorted(os.listdir(LIST_DIR))  # ['0','1','2','3','4']


# --------- FUNÇÕES AUXILIARES ---------
def load_split_list(fold, split):
    """Lê img_list/{fold}/{split}.txt e retorna lista de filenames (ex: '0.png')."""
    path = os.path.join(LIST_DIR, fold, f"{split}.txt")
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def parse_boxes(label_path):
    """Lê arquivo .txt com linhas 'x_min y_min x_max y_max' e retorna lista de bboxes."""
    bboxes = []
    for line in open(label_path):
        x1, y1, x2, y2 = map(float, line.split())
        w, h = x2 - x1, y2 - y1
        bboxes.append((x1, y1, w, h))
    return bboxes


# --------- PROCESSAMENTO POR FOLD ---------
for fold in FOLDS:
    print(f"\n=== Fold {fold} ===")
    out_fold = os.path.join(OUT_DIR, f"fold_{fold}")
    os.makedirs(out_fold, exist_ok=True)


    # Para cada subset: train, val, test
    for split in ("train", "val", "test"):
        img_list = load_split_list(fold, split)
        images, annotations = [], []
        ann_id = 1


        for img_id, fname in enumerate(img_list, start=1):
            img_path = os.path.join(IMG_DIR, fname)
            if not os.path.exists(img_path):
                print(f"  ❌ imagem não encontrada: {img_path}")
                continue


            # abre imagem para pegar width e height
            w, h = Image.open(img_path).size
            images.append({
                "id": img_id,
                "file_name": fname,
                "width": w,
                "height": h
            })


            # monta caminho do .txt correspondente
            nome_base = os.path.splitext(fname)[0]
            lbl_path = os.path.join(LBL_DIR, f"{nome_base}.txt")


            if not os.path.exists(lbl_path):
                continue


            for bbox in parse_boxes(lbl_path):
                x, y, bw, bh = bbox
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 0,  # única classe: 'tree'
                    "bbox": [x, y, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0
                })
                ann_id += 1


        coco = {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": i, "name": n} for i, n in enumerate(CLASSES)]
        }


        out_path = os.path.join(out_fold, f"instances_{split}.json")
        with open(out_path, "w") as f:
            json.dump(coco, f, indent=2)
        print(f"  ✔ {split}: {len(images)} imgs, {len(annotations)} annots → {out_path}")

