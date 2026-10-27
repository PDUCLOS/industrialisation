# PredictIRA — industrialisation (API FastAPI + Docker)

Première industrialisation du projet [PredictIRA](https://github.com/PDUCLOS/PredictIRA) : un modèle LightGBM
exposé par `POST /predict/` (FastAPI), une image Docker et un client HTML.

> **État au 27 octobre 2026 : dépôt historique.** La version maintenue (API validée par Pydantic, interface,
> tests, Kubernetes, mesures) est dans [PredictIRA, branche `feat/vitrine-predictira`](https://github.com/PDUCLOS/PredictIRA/tree/feat/vitrine-predictira).

## Ce que la remise en route a mesuré

| Constat | Détail |
|---|---|
| Client HTML incompatible avec le modèle | `index.html` envoyait `week`, `year`, `indice`, `SYM5`, `SYM24` (inconnus du modèle) et omettait `SYM1`, `SYM28`, `SYM31`, `SYM35`, `average_O_3`. LightGBM ne contrôle pas les noms : il prédisait « par position » (1 115 passages, valeur sans signification). **Corrigé** : 28 bons champs, et l'API refuse désormais tout champ manquant ou inconnu (HTTP 422). |
| Modèle non exploitable en l'état | Rejoué sur le jeu de test 2024 de PredictIRA (429 semaines, 11 départements) : MAE 510, biais moyen +488 passages, R² −3,36. Le modèle n'a pas de variable `dept` et prédit entre 530 et 1 016 passages partout, alors que l'observé va de 14 à 1 919. Le modèle à 24 variables de PredictIRA obtient MAE 41,2 et R² 0,926 sur les mêmes semaines. |
| Pickle | Le modèle est désormais chargé depuis `LightGBM_best_model.txt` (format texte LightGBM, exporté du `.pkl` via une liste blanche de classes) : aucune exécution de code au chargement. |

Mesures et scripts : `scripts/evaluate.py` et `docs/PORTFOLIO.md` du dépôt PredictIRA.

## Lancer

```bash
pip install -r requirements.txt
uvicorn api:app --port 18000
```

Le client `index.html` appelle `http://localhost:8000/predict/` : lancer sur le port 8000, ou adapter l'URL.

L'image Docker (`dockerfile`, base `python:3.9`) n'a pas été reconstruite pendant la session de 2026 (moteur Docker arrêté).

## Équipe

Sarra Abbassi, Mohammed Abusarar, Patrice Duclos, Yannick Repillez (projet de formation M2i).
