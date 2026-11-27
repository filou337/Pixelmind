# Model card – baselines LEVIR-CD (différence d'images et gradient boosting)

## Objectif

Servir de points de comparaison pour les modèles de deep learning (dossier §6.1). Ces
baselines ne doivent pas être utilisées seules pour une décision.

## Modèles

| Nom | Principe | Seuil (choisi sur val) | F1 test | IoU test | PR-AUC test |
|---|---|---|---|---|---|
| `absdiff` | Différence absolue RGB moyenne entre T1 et T2 | 104 / 255 | 13,0 % | 7,0 % | 7,3 % |
| `normdiff` | Différence après standardisation de chaque image | 115 / 255 | 12,8 % | 6,9 % | 7,5 % |
| `normdiff_smooth` | `normdiff` + moyenne locale 7×7 | 82 / 255 | 14,8 % | 8,0 % | 9,2 % |
| `gbm` | 18 features par pixel + HistGradientBoosting | 200 / 255 | 43,1 % | 27,4 % | 40,0 % |

## Données et protocole

Les modèles sont entraînés sur LEVIR-CD train (`gbm` : 835 317 pixels échantillonnés,
équilibrés). Le seuil est choisi sur val en maximisant le F1, puis les métriques sont
calculées au niveau pixel sur test (128 paires). Rapport :
`docs/experiments/levircd_benchmark.md`.

## Limites et biais connus

- Les différences d'images réagissent à tout changement radiométrique : saison,
  éclairage, végétation, ombres. Elles produisent beaucoup de faux positifs.
- `gbm` s'appuie surtout sur la couleur des toits en T2 (features `t2_b` et `t2_r` les
  plus importantes). Sur une scène résidentielle, il signale aussi des maisons qui
  existaient déjà en T1. Il reconnaît des bâtiments plutôt que des changements.
- Les changements très petits (moins de 1 % de la scène) sont mal détectés : F1 de
  12,5 % pour `gbm`.
- Domaine de validité : images RGB proches de LEVIR-CD (Google Earth, environ 0,5 m,
  zones résidentielles du Texas). Pas de garantie en dehors.

## Conditions d'emploi

Ces modèles produisent des signaux d'aide à l'analyse, à vérifier par un humain. L'API
ajoute une alerte à chaque analyse faite avec une baseline.
