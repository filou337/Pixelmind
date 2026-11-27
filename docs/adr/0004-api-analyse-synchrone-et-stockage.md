# ADR 0004 – API d'analyse synchrone, stockage SQLite + fichiers

- Statut : accepté
- Date : 2026-11-27

## Contexte

Le Sprint 4 expose l'analyse T1/T2 par API (dossier §12.2). Une paire 1024×1024 prend
environ 1 s (différence d'images) à 6 s (gradient boosting) sur CPU. Le dossier propose
PostgreSQL et un mode asynchrone (`job_id`) « si le besoin est confirmé ».

## Décision

- **Analyse synchrone** : `POST /api/v1/analyses` exécute le pipeline dans un thread et
  répond directement. La taille d'image est plafonnée (`PIXELMIND_MAX_PIXELS`, 4 Mpx par
  défaut) pour borner le temps et la mémoire. Ce plafond est vérifié sur l'en-tête, avant
  le décodage.
- **Stockage** : une ligne SQLite par analyse (`artifacts/analyses/analyses.db`, audit),
  y compris les analyses rejetées par le contrôle qualité. Les images (entrées
  normalisées, probabilité, masque, overlay) sont rangées dans
  `artifacts/analyses/{analysis_id}/`.
- **Modèles servis** : uniquement ceux qui ont un résultat de benchmark
  (`artifacts/experiments/levircd/{nom}.json`). Le seuil de décision vient de ce fichier
  (choisi sur val) ; la version est le SHA-256 des poids. Le modèle par défaut est le
  meilleur F1 test.
- **Post-traitement** : les régions de moins de `PIXELMIND_MIN_REGION_PX` pixels
  (16 px, soit 4 m² à 0,5 m) sont supprimées. Le benchmark, lui, évalue le masque brut.
- **Surfaces** : en m² seulement si `gsd_m` est fourni, et présentées comme une
  estimation (ADR 0001).

## Conséquences

- Pas de file de tâches ni de base externe : `docker compose up` suffit.
- Passage à PostgreSQL : seule la classe `AnalysisStore` change.
- Passage en asynchrone si des lots ou de grandes images arrivent : `AnalysisService.analyze`
  est déjà indépendant de HTTP et peut tourner dans un worker.
