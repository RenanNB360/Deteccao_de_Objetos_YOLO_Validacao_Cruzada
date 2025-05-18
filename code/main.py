from utils import (
    extract_zip_files,
    initial_imbalance_analysis,
    vision_balance,
    prepare_cv_folders,
    copy_files_to_folds,
    remove_empty_labels,
    results_fold,
    calculate_summary_stats
)
from pathlib import Path
import os
from ultralytics import YOLO
import ultralytics.data.build as build
from yolo_weighted_dataset import YOLOWeightedDataset

raw_data = Path('vehicles.v2-release.yolov11.zip')
kfold_dir = Path('kfold_data')
os.makedirs(kfold_dir, exist_ok=True)
origin_raw_data = Path('data')
os.makedirs(origin_raw_data, exist_ok=True)
result = Path('results')
os.makedirs(result, exist_ok=True)
raw_train_labels = os.path.join(origin_raw_data, 'train', 'labels')
origin_yaml = os.path.join(origin_raw_data, 'data.yaml')
summary_file = os.path.join(result, 'folds_summary.csv')

# 1. Extrai os dados .zip para a pasta "data"
extract_zip_files(raw_data, origin_raw_data)

# 2. Analise do balanceamento dos dados originais
class_labels, class_counts = initial_imbalance_analysis(raw_train_labels, origin_yaml)
print('\nVisão do Balanceamento das Classes:')
print(class_counts)
first_distribution_analysis = os.path.join(result, 'raw_distribution.png')
vision_balance(class_counts, class_labels, first_distribution_analysis)

# 3. Substitui a classe usada internamente pelo YOLO
build.YOLODataset = YOLOWeightedDataset

# 4. Cria as pastas para a validação cruzada
prepare_cv_folders(kfold_dir, class_names=class_labels)

# 5. Copia e os arquivos originais para as pastas de validação cruzada
copy_files_to_folds(origin_raw_data, kfold_dir)

# 6. Remove os labels vazios e suas imagens
remove_empty_labels(kfold_dir)

# 7. Treinamento com validação cruzada
for fold in range(1, 6):
    print(f"\n=== Treinando Fold {fold} ===")

    fold_path = kfold_dir / f"fold_{fold}"
    fold_yaml = fold_path / "data.yaml"

    model = YOLO('yolov8s.pt')

    results = model.train(
        data=str(fold_yaml),
        epochs=20,
        imgsz=640,
        batch=16,
        name=f"fold_{fold}_results",
        project="runs/train",
        verbose=True,
        workers=0,
        device=0,
        save_period=1,
        patience=3,
    )

    results_fold(fold, summary_file)

    print(f" Treinamento do Fold {fold} concluído.")

# 8. Verificando as médias de precision, recall e combinadas
calculate_summary_stats(summary_file)