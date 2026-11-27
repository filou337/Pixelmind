# PIXELMIND

**Multimodal Geospatial Intelligence Platform** : détection de changements de
bâtiments entre deux images satellitaires (T1/T2), avec évaluation reproductible,
puis une couche RAG/agents pour expliquer les résultats.

Architecture de référence : [`docs/PIXELMIND_Dossier_Architecture_Technique.pdf`](docs/PIXELMIND_Dossier_Architecture_Technique.pdf).
Décisions : [`docs/adr/`](docs/adr/).

## Statut

Sprint 1 (Data) : téléchargement LEVIR-CD, manifeste, contrôles qualité, EDA.

| Sprint | Contenu | Statut |
|---|---|---|
| S0 | Repo, conventions, CI, Docker | ✅ |
| S1 | LEVIR-CD, manifeste, EDA, qualité | ✅ |
| S2 | Baselines (différence d'images, RF) | ⏳ |
| S3 | U-Net, Siamese U-Net, MLflow | |
| S4 | API d'analyse, stockage | |
| S5 | RAG (Qdrant) | |
| S6 | Agents (LangGraph) + Streamlit | |
| S7 | Gouvernance / MLOps | |

## Démarrage

```bash
python3 -m venv .venv && source .venv/bin/activate
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
