# ADR 0002 – Binarisation des masques LEVIR-CD

- Statut : accepté
- Date : 2026-11-27

## Contexte

Le contrôle qualité du Sprint 1 a trouvé 32 masques sur 637 (24 train, 3 val,
5 test) qui ne sont pas strictement dans {0, 255}. Ils contiennent quelques pixels à
156 ou 254, soit au plus 0,013 % des pixels d'un masque. Ce sont des artefacts
d'annotation, pas des masques corrompus. Les classer en FAIL retirerait 5 % du dataset.

## Décision

- Règle unique de binarisation, `pixelmind.data.quality.binarize_mask` : masques
  {0,1} → `> 0`, masques 0–255 → `>= 128`. Toute la chaîne (stats, entraînement,
  évaluation) l'utilise.
- Valeurs hors {0, 255} sur ≤ 0,1 % des pixels → `WARN mask_near_binary`.
  Au-delà → `FAIL mask_not_binary`.

## Conséquences

Les 637 paires restent utilisables, et le split officiel 445/64/128 est intact.
Les 32 paires concernées restent identifiables dans le manifeste (`quality_issues`).
