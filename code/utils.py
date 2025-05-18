import zipfile
import os
import yaml
import matplotlib.pyplot as plt
import random
import shutil
from pathlib import Path
import csv
import pandas as pd

def extract_zip_files(path_zip, file_dir):
    if not os.path.exists(path_zip):
        os.makedirs(path_zip)
        print(f"Pasta '{path_zip}' criada.")

    with zipfile.ZipFile(path_zip, 'r') as zip_ref:
        zip_ref.extractall(file_dir)
        print(f"Arquivos extraídos para '{file_dir}' com sucesso!")


def initial_imbalance_analysis(train_labels_path, yaml_path):
    with open(yaml_path, 'r') as yaml_file:
        data = yaml.safe_load(yaml_file)
        class_labels = data['names']

    class_counts = {i: 0 for i in range(len(class_labels))}
    for label_file in os.listdir(train_labels_path):
        if label_file.endswith('.txt'):
            file_path = os.path.join(train_labels_path, label_file)
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1

    return class_labels, class_counts

def vision_balance(class_idx, class_labels, output_path):
    counts = [class_idx[i] for i in class_idx.keys()]

    plt.figure(figsize=(12, 6))
    plt.bar(class_labels, counts, color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Quantidade de objetos detectados')
    plt.title('Distribuição de objetos por classe')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()


def prepare_cv_folders(base_output_dir, k=5, class_names=None):
    names = class_names
    for fold in range(1, k + 1):
        fold_dir = base_output_dir / f"fold_{fold}"

        for split in ['train', 'val']:
            for sub in ['images', 'labels']:
                path = fold_dir / split / sub
                path.mkdir(parents=True, exist_ok=True)

        yaml_path = fold_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            f.write(f"path: {fold_dir}\n")
            f.write("train: train/images\n")
            f.write("val: val/images\n")
            f.write(f"names: {names}\n")

    print(f"\nEstrutura criada para {k} folds em: {base_output_dir}/fold_*/")


def copy_files_to_folds(original_data_dir, base_output_dir, k=5, seed=42):
    images_dir = os.path.join(original_data_dir, 'train', 'images')
    labels_dir = os.path.join(original_data_dir, 'train', 'labels')
    all_images = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.jpg')]

    random.seed(seed)
    random.shuffle(all_images)
    fold_size = len(all_images) // k

    for fold in range(1, k + 1):
        val_start = (fold - 1) * fold_size
        val_end = fold * fold_size if fold < k else len(all_images)
        val_images = all_images[val_start:val_end]
        train_images = [img for img in all_images if img not in val_images]

        fold_dir = base_output_dir / f"fold_{fold}"

        for split, split_images in [('train', train_images), ('val', val_images)]:
            for img_path in split_images:
                img_path = Path(img_path)
                dir_labels = Path(labels_dir)
                label_path = dir_labels / img_path.with_suffix('.txt').name

                dest_img = fold_dir / split / 'images' / img_path.name
                dest_lbl = fold_dir / split / 'labels' / label_path.name

                shutil.copy(img_path, dest_img)

                if label_path.exists():
                    shutil.copy(label_path, dest_lbl)
                else:
                    print(f" Label não encontrada para: {img_path.name}, ignorando.")

    print(f"\nArquivos copiados para {k} folds em: {base_output_dir}/fold_*/")

def remove_empty_labels(root_dir):
    removed = 0
    for fold_name in os.listdir(root_dir):
        fold_path = os.path.join(root_dir, fold_name)
        labels_path = os.path.join(fold_path, 'val', 'labels')

        for fname in os.listdir(labels_path):
            fpath = os.path.join(labels_path, fname)
            if os.path.isfile(fpath) and os.path.getsize(fpath) == 0:
                print(f'\nRemovendo label vazia: {fpath}')
                os.remove(fpath)
                base_name = os.path.splitext(fname)[0]
                for ext in ['.jpg', '.png']:
                    image_path = os.path.join(fold_path, 'val', 'images', base_name + ext)
                    if os.path.exists(image_path):
                        print(f'   ↳ Removendo imagem: {image_path}')
                        os.remove(image_path)
                        break
                removed += 1

    print(f'\n Total de arquivos removidos: {removed}')


def results_fold(fold: int, summary_file: str):
    if not os.path.exists(summary_file):
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Fold', 'Precision', 'Recall', 'Combined_Score'])

    results_csv_path = f"runs/train/fold_{fold}_results/results.csv"

    if not os.path.exists(results_csv_path):
        print(f"Results file not found: {results_csv_path}")
        return

    try:
        df = pd.read_csv(results_csv_path)

        if df.empty:
            print(f"Empty results file for fold {fold}")
            return

        precision_col = next((col for col in df.columns if 'precision' in col.lower()), None)
        recall_col = next((col for col in df.columns if 'recall' in col.lower()), None)

        if not precision_col or not recall_col:
            print(f"Metric columns not found in {results_csv_path}")
            print(f"Available columns: {', '.join(df.columns)}")
            return

        last_epoch = df.iloc[-1]
        precision = last_epoch[precision_col]
        recall = last_epoch[recall_col]
        combined_score = precision + recall

        with open(summary_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                f'Fold_{fold}',
                round(precision, 4),
                round(recall, 4),
                round(combined_score, 4)
            ])

        print(f"Fold {fold} - Precision: {precision:.3f}, Recall: {recall:.3f}, Combined: {combined_score:.3f}")

    except Exception as e:
        print(f"Error processing fold {fold}: {str(e)}")

def calculate_summary_stats(summary_file: str):
    df = pd.read_csv(summary_file)
    df = df[df['Fold'].str.startswith('Fold_', na=False)]
    print(f'\n Média Precision: {df['Precision'].mean()}, Recall: {df['Recall'].mean()}, Combined: {df['Combined_Score']}')






