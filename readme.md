# Simulation de Produits Financiers

Ce projet est un mini simulateur de produits financiers permettant de comparer le cours de deux actifs (actions ou indices) sur un graphique interactif.

## Fonctionnalités

- **Comparaison Multi-Axes** : Affiche deux actifs sur un même graphique avec des échelles (ordonnées) indépendantes.
- **Données en temps réel** : Utilise l'API `yfinance` pour récupérer les cours les plus récents.
- **Flexibilité** : L'utilisateur peut modifier les tickers et choisir la période d'analyse (1 mois à Max).
- **Indicateurs de performance** : Affiche le dernier prix et la performance sur la période sélectionnée pour chaque actif.

## Installation

1. Créer un environnement virtuel :
   ```bash
   python3 -m venv venv
   ```

2. Activer l'environnement virtuel :
   - Sur macOS/Linux :
     ```bash
     source venv/bin/activate
     ```
   - Sur Windows :
     ```bash
     .\venv\Scripts\activate
     ```

3. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

4. Lancer l'application :
   ```bash
   streamlit run app.py
   ```
