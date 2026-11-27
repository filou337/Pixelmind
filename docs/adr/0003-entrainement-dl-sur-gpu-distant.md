# ADR 0003 – Entraînement deep learning sur GPU distant (Kaggle/Colab)

- Statut : accepté
- Date : 2026-11-27

## Contexte

La machine de développement n'a pas de GPU CUDA (Ryzen 5 5500U, iGPU AMD) et dispose de
5,7 Go de RAM. Un U-Net sur LEVIR-CD (445 paires 1024×1024) demanderait plusieurs heures
par epoch sur ce CPU.

## Décision

- Le code est développé et testé **en local sur CPU** : tests unitaires et smoke test
  `configs/smoke_cpu.yaml` (8 images, réseau minuscule, 2 epochs).
- L'entraînement complet tourne sur un **GPU gratuit (Kaggle T4/P100, ou Colab)** via
  `notebooks/03_train_dl_kaggle.ipynb`. Le notebook ne contient aucune logique : il clone
  le dépôt et appelle les mêmes commandes (`pixelmind.models.dl.train`,
  `pixelmind.experiments.evaluate_checkpoint`).
- Le notebook vérifie que les données téléchargées sont identiques au manifeste versionné
  (SHA-256 de chaque image) avant d'entraîner.
- Les checkpoints, la base MLflow (SQLite) et les résultats sont rapatriés dans un zip.
  Le rapport comparatif est regénéré en local.

## Conséquences

- Le dépôt reste l'unique source de vérité : un run Kaggle se reproduit à partir d'un
  commit, d'une config YAML et d'une seed.
- Dans la base MLflow rapatriée, les métriques et les paramètres sont lisibles en local,
  mais les chemins d'artefacts pointent vers `/kaggle/working`. Les checkpoints utiles
  sont aussi copiés dans `artifacts/models/`.
- Il faut une session GPU avec Internet (Kaggle impose une vérification par téléphone).
