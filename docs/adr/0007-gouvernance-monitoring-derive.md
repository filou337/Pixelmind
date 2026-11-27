# ADR 0007 – Gouvernance : clés d'API, audit, monitoring et dérive des entrées

- Statut : accepté
- Date : 2026-11-27

## Contexte

Le Sprint 7 doit rendre la démonstration exploitable et auditable (dossier §10–11) :
savoir qui a lancé quoi, suivre la qualité du service et repérer des entrées qui ne
ressemblent plus aux données d'entraînement.

## Décision

- **Clés d'API nommées** (`PIXELMIND_API_KEYS=nom:clé,…`, clés d'au moins 16
  caractères), comparées en temps constant. Le nom de la clé est enregistré comme auteur
  dans les analyses, les questions, les workflows et les revues (`submitted_by`, en plus
  du nom saisi par le relecteur). `/health` reste public. Sans clé configurée, l'API est
  ouverte et l'affiche au démarrage : c'est réservé à la démo locale. Pas de gestion de
  rôles pour l'instant (un seul profil).
- **Journal d'accès** : une ligne JSON par requête (identifiant, méthode, chemin, statut,
  durée, auteur). L'identifiant est renvoyé dans `X-Request-ID` et repris s'il est fourni
  par l'appelant.
- **Audit** : `GET /api/v1/analyses/{id}/audit` rassemble l'analyse, ses workflows, ses
  revues et les questions posées à son sujet (dossier §10.5).
- **Monitoring** : `GET /api/v1/monitoring` calcule les indicateurs du §11.3 à partir de
  ce qui est déjà stocké : qualité des données, usage et sorties des modèles, latences
  p50/p95, revues, usage du RAG (abstentions, tokens, coût estimé avec les tarifs publics
  de 2026-11). Aucune instrumentation séparée à maintenir.
- **Dérive des entrées** : chaque analyse enregistre six statistiques simples de ses
  images (luminosité, contraste, part du rouge et du bleu, écart de luminosité et
  différence moyenne entre T1 et T2). La référence, ce sont les déciles de LEVIR-CD
  train, versionnés dans `docs/data/levircd_reference_stats.json` (statistiques
  agrégées uniquement). Pour chaque statistique, on compare les analyses récentes à la
  référence :
  - le **PSI** mesure l'ampleur de l'écart ;
  - un **test χ² à deux échantillons** vérifie que l'écart n'est pas dû au hasard
    (les déciles sont eux-mêmes estimés sur 445 images).

  Le niveau est « significatif » si le PSI vaut au moins 0,25 et p < 0,01, « modéré » si
  le PSI vaut au moins 0,10 et p < 0,05. Il faut au moins 30 analyses. Une dérive
  significative déclenche une alerte et une revue, jamais un réentraînement automatique.

## Pourquoi PSI et test statistique ensemble

Le PSI seul donne de fausses alertes sur les petites fenêtres. Sans aucune dérive, il
vaut environ (déciles − 1) / n, soit 0,3 pour 30 analyses, au-dessus du seuil usuel de
0,25. Mesuré par simulation (référence de 445 images, fenêtres de 30 à 200 analyses) :
3 % de fausses alertes, et 100 % de détection pour un décalage d'un écart type dès 30
analyses.

## Conséquences

- La CI ajoute `pip-audit` (avis de sécurité PyPI), la construction de l'image, un scan
  Trivy qui échoue sur toute vulnérabilité HIGH ou CRITICAL corrigeable, et un test de
  démarrage du conteneur. Elle tourne aussi chaque semaine pour attraper les nouvelles
  vulnérabilités. Dependabot propose les mises à jour.
- Rotation d'une clé : modifier `PIXELMIND_API_KEYS` et redémarrer l'API (runbook).
