# Simulation de Produits Financiers

Ce projet est un mini simulateur de produits financiers permettant de comparer le cours de deux actifs (actions ou indices) sur un graphique interactif.

## Fonctionnalités

- **Comparaison Multi-Axes** : Affiche deux actifs sur des graphiques séparés mais alignés temporellement.
- **Simulateur d'Investissement** : Calcule l'évolution d'un portefeuille en fonction d'une somme investie, d'une date d'achat et d'une répartition personnalisée.
- **Bilan Financier** : Affiche la valeur actuelle, la plus-value latente et la performance globale du portefeuille.
- **Données en temps réel** : Utilise l'API `yfinance` pour récupérer les cours les plus récents.

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
