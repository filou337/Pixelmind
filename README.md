# PIXELMIND

**Multimodal Geospatial Intelligence Platform** : détection de changements de
bâtiments entre deux images satellitaires (T1/T2), avec évaluation reproductible,
puis une couche RAG/agents pour expliquer les résultats.

Architecture de référence : [`docs/PIXELMIND_Dossier_Architecture_Technique.pdf`](docs/PIXELMIND_Dossier_Architecture_Technique.pdf).
Décisions : [`docs/adr/`](docs/adr/).

## Statut

Sprint 3 (Deep Learning) : U-Net et Siamese U-Net, suivi MLflow, entraînement GPU sur Kaggle.

| Sprint | Contenu | Statut |
|---|---|---|
| S0 | Repo, conventions, CI, Docker | ✅ |
| S1 | LEVIR-CD, manifeste, EDA, qualité | ✅ |
| S2 | Baselines (différence d'images, gradient boosting) | ✅ |
| S3 | U-Net, Siamese U-Net, MLflow | ⏳ code prêt, entraînement GPU à lancer |
| S4 | API d'analyse, stockage | |
| S5 | RAG (Qdrant) | |
| S6 | Agents (LangGraph) + Streamlit | |
| S7 | Gouvernance / MLOps | |

## Démarrage

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # sans GPU NVIDIA
pip install -e ".[dev]"
cp .env.example .env

pytest                 # tests
ruff check . && ruff format --check .   # lint
uvicorn pixelmind.api.main:app --reload # API → http://localhost:8000/api/v1/health
```

## Données (Sprint 1)

```bash
python -m pixelmind.data.download     # LEVIR-CD (~2,5 Go, MD5 vérifié) -> data/raw/levircd/{train,val,test}
python -m pixelmind.data.manifest     # manifeste + statut qualité -> data/manifests/levircd.jsonl
python -m pixelmind.data.eda          # rapport -> docs/data/levircd_quality_report.md
```

Le manifeste (versionné dans Git, sans images) contient pour chaque paire : chemins,
SHA-256, dimensions, modes, ratio de pixels changés, statut `PASS/WARN/REVIEW/FAIL`
et anomalies. Le split officiel train/val/test est conservé tel quel ; une même
image présente dans deux splits est marquée `FAIL` (fuite de données).

Rapport : [`docs/data/levircd_quality_report.md`](docs/data/levircd_quality_report.md).

## Baselines (Sprint 2)

```bash
python -m pixelmind.experiments.baselines            # tous les modèles
python -m pixelmind.experiments.baselines --models normdiff gbm
python -m pixelmind.experiments.report               # regénère le rapport comparatif
```

Protocole commun à tous les modèles (`pixelmind.evaluation.protocol`) : seuil choisi
sur **val** uniquement, métriques pixel sur **test** (précision, rappel, F1/Dice, IoU,
PR-AUC), par scène, et analyse d'erreurs par quantité de changement. Les métriques sont
cumulées par histogramme de scores et ne demandent donc que très peu de RAM.

Rapport comparatif (baselines + DL) : [`docs/experiments/levircd_benchmark.md`](docs/experiments/levircd_benchmark.md).

## Deep Learning (Sprint 3)

U-Net (T1 et T2 concaténées, 6 canaux) et Siamese U-Net (encodeur partagé, fusion
`|f1 − f2|` ou concaténation à chaque échelle), loss BCE + Dice, AdamW + cosine,
early stopping sur le F1 de validation, suivi MLflow (SQLite).

```bash
# local, CPU : vérifie le pipeline (pas la qualité)
python -m pixelmind.models.dl.train --config configs/smoke_cpu.yaml
# GPU (Kaggle/Colab) : notebooks/03_train_dl_kaggle.ipynb, ou directement
python -m pixelmind.models.dl.train --config configs/siamese_unet.yaml --set epochs=50
python -m pixelmind.experiments.evaluate_checkpoint artifacts/models/siamese_unet_diff_best.pt
mlflow ui --backend-store-uri sqlite:///artifacts/models/mlflow.db
```

Pourquoi entraîner sur Kaggle : voir [ADR 0003](docs/adr/0003-entrainement-dl-sur-gpu-distant.md).

Avec Docker :

```bash
docker compose up --build
```

## Structure

```
src/pixelmind/   code applicatif (api/, puis data/, models/, evaluation/, rag/, agents/…)
tests/           tests unitaires et d'intégration
docs/            dossier d'architecture, ADR, model cards
data/, artifacts/  locaux, ignorés par Git
```

## Données et licence

Le dataset MVP est [LEVIR-CD](https://justchenhao.github.io/LEVIR/) (Chen & Shi, 2020),
**réservé à un usage académique, non commercial**. Les images et masques ne sont
jamais versionnés dans ce dépôt. Ce projet est un projet d'apprentissage/portfolio,
pas un produit commercial.
