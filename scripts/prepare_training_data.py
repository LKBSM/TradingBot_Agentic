# =============================================================================
# PRÉPARATION DES DONNÉES D'ENTRAÎNEMENT
# =============================================================================
# Ce script:
# 1. Charge les données Gold OHLCV
# 2. Crée l'Economic Calendar avec vraies données
# 3. Fusionne les deux par timestamp
# 4. Crée les splits Train/Val/Test
# 5. Sauvegarde les fichiers prêts pour l'entraînement
#
# Usage (sur Colab ou local):
#   python scripts/prepare_training_data.py
# =============================================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_GOLD_FILE = "data/XAU_15MIN_2019_2025.csv"
OUTPUT_DIR = "data/prepared"

# Splits
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# =============================================================================
# VRAIES DONNÉES ÉCONOMIQUES HISTORIQUES
# =============================================================================

# NFP (Non-Farm Payrolls) - en milliers de jobs
NFP_DATA = {
    2019: {1: 304, 2: 56, 3: 189, 4: 263, 5: 75, 6: 224, 7: 164, 8: 130, 9: 136, 10: 128, 11: 266, 12: 147},
    2020: {1: 225, 2: 273, 3: -701, 4: -20687, 5: 2509, 6: 4800, 7: 1763, 8: 1371, 9: 661, 10: 638, 11: 245, 12: -140},
    2021: {1: 233, 2: 468, 3: 916, 4: 278, 5: 614, 6: 962, 7: 1091, 8: 366, 9: 312, 10: 546, 11: 249, 12: 510},
    2022: {1: 504, 2: 714, 3: 431, 4: 428, 5: 390, 6: 372, 7: 528, 8: 315, 9: 263, 10: 261, 11: 263, 12: 223},
    2023: {1: 517, 2: 311, 3: 236, 4: 294, 5: 339, 6: 306, 7: 187, 8: 236, 9: 297, 10: 150, 11: 199, 12: 216},
    2024: {1: 353, 2: 275, 3: 315, 4: 165, 5: 272, 6: 206, 7: 114, 8: 142, 9: 254, 10: 227, 11: 256, 12: 212},
    2025: {1: 143, 2: 151, 3: 228, 4: 177, 5: 139, 6: 147, 7: 72, 8: -4, 9: 119, 10: -173, 11: 41, 12: 50},
}

# CPI Year-over-Year (%)
CPI_DATA = {
    2019: {1: 1.6, 2: 1.5, 3: 1.9, 4: 2.0, 5: 1.8, 6: 1.6, 7: 1.8, 8: 1.7, 9: 1.7, 10: 1.8, 11: 2.1, 12: 2.3},
    2020: {1: 2.5, 2: 2.3, 3: 1.5, 4: 0.3, 5: 0.1, 6: 0.6, 7: 1.0, 8: 1.3, 9: 1.4, 10: 1.2, 11: 1.2, 12: 1.4},
    2021: {1: 1.4, 2: 1.7, 3: 2.6, 4: 4.2, 5: 5.0, 6: 5.4, 7: 5.4, 8: 5.3, 9: 5.4, 10: 6.2, 11: 6.8, 12: 7.0},
    2022: {1: 7.5, 2: 7.9, 3: 8.5, 4: 8.3, 5: 8.6, 6: 9.1, 7: 8.5, 8: 8.3, 9: 8.2, 10: 7.7, 11: 7.1, 12: 6.5},
    2023: {1: 6.4, 2: 6.0, 3: 5.0, 4: 4.9, 5: 4.0, 6: 3.0, 7: 3.2, 8: 3.7, 9: 3.7, 10: 3.2, 11: 3.1, 12: 3.4},
    2024: {1: 3.1, 2: 3.2, 3: 3.5, 4: 3.4, 5: 3.3, 6: 3.0, 7: 2.9, 8: 2.5, 9: 2.4, 10: 2.6, 11: 2.7, 12: 2.9},
    2025: {1: 3.0, 2: 2.8, 3: 2.4, 4: 2.3, 5: 2.4, 6: 2.7, 7: 2.7, 8: 2.9, 9: 3.0, 10: 2.8, 11: 2.7, 12: 2.7},
}

# Federal Funds Rate (%)
FED_RATE_DATA = {
    2019: {1: 2.50, 2: 2.50, 3: 2.50, 4: 2.50, 5: 2.50, 6: 2.50, 7: 2.25, 8: 2.25, 9: 2.00, 10: 1.75, 11: 1.75, 12: 1.75},
    2020: {1: 1.75, 2: 1.75, 3: 0.25, 4: 0.25, 5: 0.25, 6: 0.25, 7: 0.25, 8: 0.25, 9: 0.25, 10: 0.25, 11: 0.25, 12: 0.25},
    2021: {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25, 5: 0.25, 6: 0.25, 7: 0.25, 8: 0.25, 9: 0.25, 10: 0.25, 11: 0.25, 12: 0.25},
    2022: {1: 0.25, 2: 0.25, 3: 0.50, 4: 0.50, 5: 1.00, 6: 1.75, 7: 2.50, 8: 2.50, 9: 3.25, 10: 3.25, 11: 4.00, 12: 4.50},
    2023: {1: 4.50, 2: 4.75, 3: 5.00, 4: 5.00, 5: 5.25, 6: 5.25, 7: 5.50, 8: 5.50, 9: 5.50, 10: 5.50, 11: 5.50, 12: 5.50},
    2024: {1: 5.50, 2: 5.50, 3: 5.50, 4: 5.50, 5: 5.50, 6: 5.50, 7: 5.50, 8: 5.50, 9: 5.00, 10: 5.00, 11: 4.75, 12: 4.50},
    2025: {1: 4.50, 2: 4.50, 3: 4.50, 4: 4.50, 5: 4.50, 6: 4.50, 7: 4.50, 8: 4.50, 9: 4.25, 10: 4.00, 11: 4.00, 12: 3.75},
}

# FOMC Meeting Dates (MM-DD format)
FOMC_DATES = {
    2019: ['01-30', '03-20', '05-01', '06-19', '07-31', '09-18', '10-30', '12-11'],
    2020: ['01-29', '03-03', '03-15', '04-29', '06-10', '07-29', '09-16', '11-05', '12-16'],
    2021: ['01-27', '03-17', '04-28', '06-16', '07-28', '09-22', '11-03', '12-15'],
    2022: ['01-26', '03-16', '05-04', '06-15', '07-27', '09-21', '11-02', '12-14'],
    2023: ['02-01', '03-22', '05-03', '06-14', '07-26', '09-20', '11-01', '12-13'],
    2024: ['01-31', '03-20', '05-01', '06-12', '07-31', '09-18', '11-07', '12-18'],
    2025: ['01-29', '03-19', '05-07', '06-18', '07-30', '09-17', '10-29', '12-10'],
}


def get_first_friday(year, month):
    """Retourne le premier vendredi du mois (jour du NFP)."""
    first_day = datetime(year, month, 1)
    days_until_friday = (4 - first_day.weekday()) % 7
    return first_day + timedelta(days=days_until_friday)


def get_cpi_release_date(year, month):
    """Retourne la date approximative du CPI (2ème semaine, mardi/mercredi)."""
    # CPI sort généralement le 2ème mardi ou mercredi du mois
    first_day = datetime(year, month, 1)
    days_until_tuesday = (1 - first_day.weekday()) % 7
    first_tuesday = first_day + timedelta(days=days_until_tuesday)
    return first_tuesday + timedelta(days=7)  # 2ème semaine


def create_economic_calendar():
    """Crée le calendar économique avec les vraies données historiques."""
    events = []

    for year in range(2019, 2026):
        for month in range(1, 13):
            # Skip si pas de données
            if year == 2024 and month > 12:
                continue

            # =================================================================
            # NFP - Non-Farm Payrolls (Premier vendredi du mois)
            # =================================================================
            nfp_date = get_first_friday(year, month)
            nfp_value = NFP_DATA.get(year, {}).get(month, 200)
            nfp_prev = NFP_DATA.get(year, {}).get(month - 1) if month > 1 else NFP_DATA.get(year - 1, {}).get(12, 200)

            # Calculer la surprise
            if nfp_prev and nfp_prev != 0:
                nfp_surprise = (nfp_value - nfp_prev) / abs(nfp_prev) if nfp_prev else 0
            else:
                nfp_surprise = 0

            events.append({
                'datetime': nfp_date.replace(hour=12, minute=30),
                'event': 'NFP',
                'event_full': 'Non-Farm Payrolls',
                'impact': 'HIGH',
                'actual': nfp_value,
                'previous': nfp_prev,
                'surprise': nfp_surprise,
                'surprise_direction': 1 if nfp_value > (nfp_prev or 0) else -1
            })

            # =================================================================
            # CPI - Consumer Price Index
            # =================================================================
            cpi_date = get_cpi_release_date(year, month)
            cpi_value = CPI_DATA.get(year, {}).get(month, 2.0)
            cpi_prev = CPI_DATA.get(year, {}).get(month - 1) if month > 1 else CPI_DATA.get(year - 1, {}).get(12, 2.0)

            cpi_surprise = (cpi_value - cpi_prev) if cpi_prev else 0

            events.append({
                'datetime': cpi_date.replace(hour=12, minute=30),
                'event': 'CPI',
                'event_full': 'Consumer Price Index YoY',
                'impact': 'HIGH',
                'actual': cpi_value,
                'previous': cpi_prev,
                'surprise': cpi_surprise,
                'surprise_direction': 1 if cpi_value > (cpi_prev or 0) else -1
            })

        # =================================================================
        # FOMC Meetings
        # =================================================================
        for date_str in FOMC_DATES.get(year, []):
            try:
                month = int(date_str.split('-')[0])
                day = int(date_str.split('-')[1])
                fomc_date = datetime(year, month, day, 18, 0)

                rate = FED_RATE_DATA.get(year, {}).get(month, 2.0)
                rate_prev = FED_RATE_DATA.get(year, {}).get(month - 1) if month > 1 else FED_RATE_DATA.get(year - 1, {}).get(12, 2.0)

                rate_change = rate - (rate_prev or rate)

                events.append({
                    'datetime': fomc_date,
                    'event': 'FOMC',
                    'event_full': 'FOMC Rate Decision',
                    'impact': 'HIGH',
                    'actual': rate,
                    'previous': rate_prev,
                    'surprise': rate_change,
                    'surprise_direction': 1 if rate_change > 0 else (-1 if rate_change < 0 else 0)
                })
            except:
                continue

    df = pd.DataFrame(events)
    df = df.sort_values('datetime').reset_index(drop=True)
    return df


def merge_gold_with_calendar(gold_df, calendar_df, window_before=60, window_after=120):
    """
    Fusionne les données Gold avec le calendar économique.

    Args:
        gold_df: DataFrame avec prix Gold (index = datetime)
        calendar_df: DataFrame avec événements économiques
        window_before: Minutes avant l'événement à marquer
        window_after: Minutes après l'événement à marquer

    Returns:
        DataFrame fusionné avec colonnes news ajoutées
    """
    df = gold_df.copy()

    # Initialiser les colonnes news
    df['news_event'] = 0
    df['news_impact'] = 0.0
    df['news_type'] = ''
    df['news_surprise'] = 0.0
    df['news_direction'] = 0
    df['minutes_to_news'] = 999  # Grande valeur par défaut

    for _, event in calendar_df.iterrows():
        event_time = event['datetime']

        # Fenêtre temporelle autour de l'événement
        window_start = event_time - timedelta(minutes=window_before)
        window_end = event_time + timedelta(minutes=window_after)

        # Trouver les barres dans la fenêtre
        mask = (df.index >= window_start) & (df.index <= window_end)

        if mask.any():
            df.loc[mask, 'news_event'] = 1
            df.loc[mask, 'news_impact'] = 1.0 if event['impact'] == 'HIGH' else 0.5
            df.loc[mask, 'news_type'] = event['event']
            df.loc[mask, 'news_surprise'] = event['surprise']
            df.loc[mask, 'news_direction'] = event['surprise_direction']

            # Calculer minutes jusqu'à l'événement
            for idx in df.loc[mask].index:
                minutes_diff = (event_time - idx).total_seconds() / 60
                if abs(minutes_diff) < abs(df.loc[idx, 'minutes_to_news']):
                    df.loc[idx, 'minutes_to_news'] = minutes_diff

    # Remplacer les 999 par 0 pour les barres sans news proche
    df.loc[df['minutes_to_news'] == 999, 'minutes_to_news'] = 0

    return df


def create_training_splits(df, train_ratio=0.70, val_ratio=0.15):
    """
    Crée les splits Train/Val/Test de manière chronologique.

    IMPORTANT: Split chronologique, pas random!
    """
    total_len = len(df)
    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))

    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    return df_train, df_val, df_test


def main():
    """Fonction principale."""
    print("=" * 70)
    print("PRÉPARATION DES DONNÉES D'ENTRAÎNEMENT")
    print("=" * 70)

    # Créer le dossier output
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =================================================================
    # ÉTAPE 1: Charger les données Gold
    # =================================================================
    print(f"\n📂 Chargement des données Gold...")

    if not os.path.exists(INPUT_GOLD_FILE):
        print(f"❌ Fichier non trouvé: {INPUT_GOLD_FILE}")
        print("   Assure-toi que le fichier existe dans le dossier data/")
        return

    gold_df = pd.read_csv(INPUT_GOLD_FILE, parse_dates=['Date'])
    gold_df = gold_df.set_index('Date')
    gold_df = gold_df.sort_index()

    print(f"   ✅ {len(gold_df):,} barres chargées")
    print(f"   📅 Période: {gold_df.index.min()} → {gold_df.index.max()}")

    # =================================================================
    # ÉTAPE 2: Créer l'Economic Calendar
    # =================================================================
    print(f"\n📅 Création de l'Economic Calendar...")

    calendar_df = create_economic_calendar()

    print(f"   ✅ {len(calendar_df)} événements créés")
    print(f"   📊 Par type:")
    print(f"      NFP:  {len(calendar_df[calendar_df['event'] == 'NFP'])} événements")
    print(f"      CPI:  {len(calendar_df[calendar_df['event'] == 'CPI'])} événements")
    print(f"      FOMC: {len(calendar_df[calendar_df['event'] == 'FOMC'])} événements")

    # Sauvegarder le calendar séparément
    calendar_df.to_csv(f"{OUTPUT_DIR}/economic_calendar.csv", index=False)

    # =================================================================
    # ÉTAPE 3: Fusionner Gold + Calendar
    # =================================================================
    print(f"\n🔗 Fusion Gold + Economic Calendar...")

    merged_df = merge_gold_with_calendar(gold_df, calendar_df)

    news_bars = merged_df['news_event'].sum()
    print(f"   ✅ Fusion terminée")
    print(f"   📰 Barres avec news: {news_bars:,} ({news_bars/len(merged_df)*100:.1f}%)")

    # =================================================================
    # ÉTAPE 4: Créer les splits
    # =================================================================
    print(f"\n✂️ Création des splits Train/Val/Test...")

    df_train, df_val, df_test = create_training_splits(merged_df, TRAIN_RATIO, VAL_RATIO)

    print(f"   Train:      {len(df_train):>8,} barres ({df_train.index.min().date()} → {df_train.index.max().date()})")
    print(f"   Validation: {len(df_val):>8,} barres ({df_val.index.min().date()} → {df_val.index.max().date()})")
    print(f"   Test:       {len(df_test):>8,} barres ({df_test.index.min().date()} → {df_test.index.max().date()})")

    # =================================================================
    # ÉTAPE 5: Sauvegarder les fichiers
    # =================================================================
    print(f"\n💾 Sauvegarde des fichiers...")

    # Fichier complet fusionné
    merged_df.to_csv(f"{OUTPUT_DIR}/gold_with_news_COMPLETE.csv")
    print(f"   ✅ {OUTPUT_DIR}/gold_with_news_COMPLETE.csv")

    # Splits
    df_train.to_csv(f"{OUTPUT_DIR}/train.csv")
    df_val.to_csv(f"{OUTPUT_DIR}/validation.csv")
    df_test.to_csv(f"{OUTPUT_DIR}/test.csv")
    print(f"   ✅ {OUTPUT_DIR}/train.csv")
    print(f"   ✅ {OUTPUT_DIR}/validation.csv")
    print(f"   ✅ {OUTPUT_DIR}/test.csv")

    # =================================================================
    # RÉSUMÉ
    # =================================================================
    print(f"\n{'=' * 70}")
    print("✅ PRÉPARATION TERMINÉE!")
    print(f"{'=' * 70}")
    print(f"""
Fichiers créés dans {OUTPUT_DIR}/:
├── economic_calendar.csv      # Calendar seul
├── gold_with_news_COMPLETE.csv # Données fusionnées complètes
├── train.csv                  # 70% pour entraînement
├── validation.csv             # 15% pour validation
└── test.csv                   # 15% pour test final

Colonnes ajoutées:
├── news_event      # 0 ou 1 (événement proche?)
├── news_impact     # 0.0 à 1.0 (importance)
├── news_type       # NFP, CPI, FOMC
├── news_surprise   # Valeur de la surprise (actual - expected)
├── news_direction  # 1 (positif), -1 (négatif), 0 (neutre)
└── minutes_to_news # Minutes jusqu'au prochain événement

Tu peux maintenant uploader ces fichiers sur Google Colab!
    """)

    return merged_df, df_train, df_val, df_test


if __name__ == "__main__":
    main()
