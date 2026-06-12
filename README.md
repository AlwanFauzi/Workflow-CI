# Workflow-CI

Repository ini berisi MLflow Project & GitHub Actions workflow CI untuk submission
**Membangun Sistem Machine Learning** (Kriteria 3) — model regresi harga rumah (House Prices).

## Struktur
- `MLProject/` — MLflow Project (`MLProject`, `conda.yaml`, `modelling.py`).
- `namadataset_preprocessing/` — dataset siap latih (`train_preprocessed.csv`, `test_preprocessed.csv`).
- `.github/workflows/ci.yml` — workflow CI yang:
  1. Menjalankan `mlflow run MLProject` untuk retraining otomatis.
  2. Mengunggah & mem-commit hasil tracking (`mlruns/`) ke repository.
  3. Build Docker image dari model hasil training menggunakan `mlflow models build-docker`.
  4. Push image ke Docker Hub.

## Menjalankan secara lokal
```bash
pip install mlflow==2.19.0 pandas numpy scikit-learn matplotlib
mlflow run ./MLProject --env-manager=local -P data_dir=$(pwd)/namadataset_preprocessing
```

## Konfigurasi Secrets GitHub (wajib untuk push Docker image)
Tambahkan secrets berikut pada **Settings > Secrets and variables > Actions**:
- `DOCKERHUB_USERNAME` — username Docker Hub.
- `DOCKERHUB_TOKEN` — access token Docker Hub (Account Settings > Security > New Access Token).

## Docker Image
Image hasil build akan tersedia di:
```
docker pull <DOCKERHUB_USERNAME>/house-price-model:latest
```
