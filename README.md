# Music Genre Clustering from Keyword TF-IDF & PCA Embeddings

A reproducible pipeline that vectorizes song keywords (position-wise TF–IDF), reduces dimensionality with PCA, combines multiple keyword-position embeddings into a single representation, performs K-Means clustering, and evaluates cluster quality using both intrinsic and extrinsic metrics. This repository contains a script and a notebook that produce `song_clusters.csv` and evaluation plots.

---

## Project summary

This project implements a full clustering workflow for songs described by three keyword slots (`keyword_1`, `keyword_2`, `keyword_3`). The pipeline performs:

1. **TF–IDF (by keyword position)** — compute TF–IDF separately for each keyword column.  
2. **PCA (per position)** — reduce each TF–IDF matrix to 2 principal components.  
3. **Combine embeddings** — merge the three 2D embeddings using configurable weights into a final 2D embedding per song.  
4. **Clustering** — run K-Means on the combined embedding; use Elbow (WCSS) + Silhouette to help pick `k`.  
5. **Cluster→Genre mapping** — assign each cluster a predicted genre by majority vote of ground truth.  
6. **Evaluation** — compute accuracy, Purity, NMI, ARI, FMI and save cluster results.

---

## Repository contents

- `TASK2_dataset.csv` — input dataset (required). Expected columns: `keyword_1`, `keyword_2`, `keyword_3`, `genre`.  
- `cluster_songs.py` — main pipeline script (copy-paste friendly).  
- `Task 1.ipynb` — optional notebook with experiments and visualizations.  
- `song_clusters.csv` — example output produced by the pipeline (clusters + predicted genres).  
- `output/` — default folder where plots and CSVs are saved (created by the script).

---

## Requirements

Install required Python packages:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy
