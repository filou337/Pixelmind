# ADR 0001 – MVP sans couche géospatiale

- Statut : accepté
- Date : 2026-11-27

## Contexte

Le dataset MVP, LEVIR-CD, est composé de paires PNG RGB 1024×1024 (~0,5 m/pixel)
**sans géoréférencement** (pas de CRS ni de transformation affine). Le dossier
d'architecture prévoit Rasterio/GDAL, CRS, reprojection, STAC, Geo Tool et PostGIS.

## Décision

Le MVP ne contient pas de couche géospatiale. Les sorties sont exprimées en pixels
(`changed_pixel_ratio`). Une surface approximative (pixels × 0,25 m²) peut être
affichée, explicitement marquée comme estimation, jamais comme surface géographique.

## Conséquences

- Dépendances plus légères (pas de GDAL), installation et CI plus simples.
- La couche géo (module `geo/`, STAC, PostGIS) sera ajoutée quand un dataset
  géoréférencé (GeoTIFF/COG, SpaceNet, Sentinel-2) entrera dans le périmètre.
