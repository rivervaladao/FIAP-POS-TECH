import os
import json
import shutil
from pycocotools.coco import COCO

# Configurações centralizadas
CONFIG = {
    'data_dir': '/home/river/Downloads/coco-dataset',
    'train_img_dir': 'train2017',
    'ann_file': 'annotations_trainval2017/annotations/instances_train2017.json',
    'output_dir': 'filtered_dataset',
    'classes': ['knife', 'scissors'],  # Classes desejadas
    'max_negatives': 500  # Limite de imagens negativas
}

# Inicializar diretórios
def setup_directories(config):
    output_img_dir = os.path.join(config['data_dir'], config['output_dir'], 'images')
    output_label_dir = os.path.join(config['data_dir'], config['output_dir'], 'labels')
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    return output_img_dir, output_label_dir

# Converter anotações COCO para YOLO
def convert_to_yolo(bbox, img_w, img_h, class_id):
    x_min, y_min, w, h = bbox
    x_center = (x_min + w / 2) / img_w
    y_center = (y_min + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return f"{class_id} {x_center} {y_center} {w_norm} {h_norm}"

# Processar imagem (positiva ou negativa)
def process_image(img_info, coco, output_img_dir, output_label_dir, train_img_dir, cat_ids=None, coco_to_yolo=None):
    img_path = os.path.join(train_img_dir, img_info['file_name'])
    output_img_path = os.path.join(output_img_dir, img_info['file_name'])
    
    # Copiar imagem
    shutil.copy(img_path, output_img_path)
    
    # Criar arquivo de anotações
    label_path = os.path.join(output_label_dir, img_info['file_name'].replace('.jpg', '.txt'))
    if cat_ids:  # Para imagens positivas
        ann_ids = coco.getAnnIds(imgIds=img_info['id'], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        with open(label_path, 'w') as f:
            for ann in anns:
                if ann['category_id'] in coco_to_yolo:
                    yolo_class = coco_to_yolo[ann['category_id']]
                    bbox = convert_to_yolo(ann['bbox'], img_info['width'], img_info['height'], yolo_class)
                    f.write(bbox + '\n')
    else:  # Para imagens negativas
        open(label_path, 'w').close()  # Arquivo vazio para negativas

# Função principal
def main():
    # Expandir caminhos
    config = CONFIG.copy()
    config['data_dir'] = os.path.expanduser(config['data_dir'])
    config['train_img_dir'] = os.path.join(config['data_dir'], config['train_img_dir'])
    config['ann_file'] = os.path.join(config['data_dir'], config['ann_file'])
    
    # Configurar diretórios
    output_img_dir, output_label_dir = setup_directories(config)
    
    # Inicializar COCO
    coco = COCO(config['ann_file'])
    
    # Obter IDs das classes desejadas
    cat_ids = coco.getCatIds(catNms=config['classes'])
    coco_to_yolo = {43: 0, 74: 1}  # Mapear COCO IDs para YOLO (0: knife, 1: scissors)
    
    # Obter imagens positivas
    positive_img_ids = set()
    for cat_id in cat_ids:
        positive_img_ids.update(coco.getImgIds(catIds=[cat_id]))
    
    # Obter imagens negativas
    all_img_ids = coco.getImgIds()
    negative_img_ids = [img_id for img_id in all_img_ids if img_id not in positive_img_ids]
    
    # Processar imagens positivas
    print("Processando imagens positivas...")
    for img_id in positive_img_ids:
        img_info = coco.loadImgs(img_id)[0]
        process_image(img_info, coco, output_img_dir, output_label_dir, config['train_img_dir'], 
                     cat_ids=cat_ids, coco_to_yolo=coco_to_yolo)
    
    # Processar imagens negativas (limitado a max_negatives)
    print("Processando imagens negativas...")
    for img_id in negative_img_ids[:config['max_negatives']]:
        img_info = coco.loadImgs(img_id)[0]
        process_image(img_info, coco, output_img_dir, output_label_dir, config['train_img_dir'])
    
    print(f"Total de imagens processadas: {len(os.listdir(output_img_dir))}")

if __name__ == "__main__":
    main()