# Fiche dataset – LEVIR-CD

## Description

LEVIR-CD (Chen & Shi, 2020) est un jeu de données de détection de changements de
bâtiments. Il contient 637 paires d'images RGB de 1024×1024 pixels, extraites de
Google Earth, d'une résolution d'environ 0,5 m par pixel, avec un masque binaire des
changements de bâtiments (constructions et démolitions) entre les deux dates.

Source officielle : https://justchenhao.github.io/LEVIR/
Article : Chen, H., & Shi, Z. (2020). A Spatial-Temporal Attention-Based Method and a
New Dataset for Remote Sensing Image Change Detection. Remote Sensing, 12(10), 1662.

## Licence et conditions d'utilisation

Les images et les annotations sont réservées à un **usage académique**. Tout usage
commercial est interdit. PIXELMIND, projet d'apprentissage et de portfolio, respecte
cette restriction :
- aucune image ni aucun masque n'est versionné dans le dépôt Git ;
- les auteurs sont cités ;
- le projet n'est pas présenté comme un produit commercial.

## Split utilisé

Le split officiel est conservé tel quel : 445 paires en train, 64 en validation et 128
en test. Le split est fait au niveau de l'image d'origine, avant tout découpage en
patches, pour éviter les fuites de données. Le contrôle qualité du Sprint 1 n'a trouvé
aucune image partagée entre deux splits.

## Caractéristiques mesurées (Sprint 1)

- Toutes les images font 1024×1024 pixels.
- Environ 4,6 % des pixels sont « changés » en moyenne (forte classe minoritaire).
- 11 % des paires de train ne contiennent aucun changement.
- 32 masques contiennent quelques pixels parasites (valeurs 156 ou 254, au plus
  0,013 % des pixels). Ils sont binarisés avec le seuil ≥ 128 (ADR 0002).
- Les images ne sont pas géoréférencées : pas de CRS ni de transformation affine
  (ADR 0001). Une surface en m² n'est donc qu'une estimation (pixels × 0,25 m²).

## Limites connues

Les paires montrent de fortes variations de saison, d'éclairage et de végétation entre
T1 et T2. Un modèle entraîné sur LEVIR-CD peut mal se généraliser à d'autres régions,
d'autres capteurs ou d'autres résolutions.
