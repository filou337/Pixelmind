# Runbook d'exploitation – PIXELMIND

Procédures pour la version de démonstration (Docker Compose, une machine).

## Démarrer, arrêter, vérifier

```bash
cp .env.example .env            # puis renseigner PIXELMIND_API_KEYS et ANTHROPIC_API_KEY
docker compose up -d --build    # API :8000, interface :8501
curl localhost:8000/api/v1/health
docker compose logs -f api      # journal d'accès JSON, une ligne par requête
docker compose down
```

Clés d'API : `PIXELMIND_API_KEYS=alice:<clé>,bob:<clé>` (au moins 16 caractères,
générer avec `python -c "import secrets; print(secrets.token_urlsafe(24))"`).
L'interface utilise `PIXELMIND_API_KEY`. Rotation : changer la valeur dans `.env`, puis
`docker compose up -d` (redémarre l'API).

## Données persistantes et sauvegarde

Tout l'état est dans `artifacts/` (volume monté) :

| Chemin | Contenu |
|---|---|
| `artifacts/analyses/analyses.db` + `artifacts/analyses/{id}/` | analyses (audit), images produites, rapports |
| `artifacts/workflows/workflows.db` | workflows, revues, points de reprise LangGraph |
| `artifacts/rag/qa_log.db` | questions, réponses, citations, tokens |
| `artifacts/rag/qdrant/`, `index_manifest.json` | index documentaire (reconstructible) |
| `artifacts/experiments/levircd/`, `artifacts/models/` | résultats de benchmark et poids servis |

Sauvegarde à chaud des bases SQLite : `sqlite3 <db> ".backup <fichier>"`, pour
`analyses.db`, `workflows.db` et `qa_log.db`, plus une copie de `artifacts/analyses/`
et `artifacts/models/`. L'index RAG se reconstruit, inutile de le sauvegarder.

## Modèles : publier, revenir en arrière

L'API sert les modèles qui ont un résultat dans `artifacts/experiments/levircd/{nom}.json`
(seuil choisi sur val, métriques test). Le modèle par défaut est le meilleur F1 test.

- **Publier** un checkpoint : `python -m pixelmind.experiments.evaluate_checkpoint
  artifacts/models/<nom>_best.pt`, mettre à jour la model card, puis redémarrer l'API.
- **Revenir en arrière** : déplacer `{nom}.json` (et le checkpoint) hors du dossier, puis
  redémarrer. Le modèle disparaît de `/api/v1/models` et le précédent redevient le
  modèle par défaut. Les anciennes analyses gardent la version (SHA-256) qui les a
  produites.

## Index documentaire

Après une modification de la documentation, ou d'un document de `configs/rag_corpus.yaml` :
arrêter l'API (Qdrant embarqué n'accepte qu'un processus), lancer
`python -m pixelmind.rag.index` puis `python -m pixelmind.rag.evaluate`, vérifier que
Recall@5 ne baisse pas, et redémarrer.

## Incidents

| Symptôme | Diagnostic | Action |
|---|---|---|
| Questions en 503 « LLM non configuré » | clé `ANTHROPIC_API_KEY` absente ou invalide | corriger la clé ; en attendant, `PIXELMIND_LLM_PROVIDER=extractive` |
| Questions en 429 | limite de débit de l'API Claude | patienter ; réduire l'usage ; envisager `claude-haiku-5-5` |
| Questions en 503 « index RAG absent » | index jamais construit | `python -m pixelmind.rag.index` (API arrêtée) |
| Alerte de dérive dans Gouvernance | `GET /api/v1/monitoring` → `drift` | revue humaine des analyses récentes, vérifier le capteur, la résolution et la région. **Pas de réentraînement automatique** : il faut de nouvelles annotations, un split indépendant et une évaluation avant toute publication (dossier §11.4) |
| Taux de rejet qualité en hausse | `data_quality.rejected_reasons` | identifier la source des fichiers (format, tailles différentes) |
| Latence p95 en hausse | `service.analysis_latency_s` | taille des images (`PIXELMIND_MAX_PIXELS`), modèle utilisé, charge CPU |
| Workflow bloqué « en attente de revue » | `GET /api/v1/workflows/{id}` | soumettre la revue ; l'état survit aux redémarrages |

## Retrouver l'historique d'un résultat

`GET /api/v1/analyses/{id}/audit` renvoie :
- l'analyse : auteur, empreintes des entrées, version du modèle, seuil et paramètres ;
- ses workflows et ses revues (qui a décidé, quand, avec quel commentaire) ;
- les questions posées à son sujet, avec leurs sources.

Le rapport `report.md` en donne une version lisible.
