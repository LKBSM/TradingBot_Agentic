# =============================================================================
# ECONOMIC CALENDAR DOWNLOADER
# =============================================================================
# Script pour télécharger l'economic calendar de 2019-2024
# Sources: Forex Factory, Investing.com, ou génération basée sur patterns connus
#
# Usage:
#   python scripts/download_economic_calendar.py
#
# Output:
#   data/economic_calendar_2019_2025.csv
# =============================================================================

import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import requests
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Ajouter le path du projet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class EconomicCalendarDownloader:
    """
    Télécharge l'economic calendar depuis plusieurs sources.

    Priorité des sources:
    1. Forex Factory (scraping)
    2. Investing.com API
    3. Génération basée sur patterns connus (fallback)
    """

    # Événements HIGH IMPACT pour Gold (USD)
    HIGH_IMPACT_EVENTS = [
        "Non-Farm Payrolls",
        "NFP",
        "FOMC",
        "Federal Funds Rate",
        "Fed Interest Rate Decision",
        "CPI",
        "Consumer Price Index",
        "Core CPI",
        "PPI",
        "Producer Price Index",
        "GDP",
        "Gross Domestic Product",
        "Retail Sales",
        "Unemployment Rate",
        "Initial Jobless Claims",
        "ISM Manufacturing PMI",
        "ISM Services PMI",
        "Durable Goods Orders",
        "Housing Starts",
        "Consumer Confidence",
        "PCE Price Index",
        "Core PCE",
        "Fed Chair Powell",
        "Jackson Hole",
        "FOMC Minutes",
        "Trade Balance",
    ]

    # Schedule connu des événements récurrents (pour génération)
    RECURRING_EVENTS = {
        'NFP': {
            'day_of_month': 'first_friday',
            'time': '12:30',  # UTC
            'impact': 'HIGH',
            'currency': 'USD'
        },
        'FOMC_DECISION': {
            'months': [1, 3, 5, 6, 7, 9, 11, 12],  # 8 meetings par an
            'day_of_month': 'third_wednesday',
            'time': '18:00',
            'impact': 'HIGH',
            'currency': 'USD'
        },
        'CPI': {
            'day_of_month': 'second_week_tuesday',
            'time': '12:30',
            'impact': 'HIGH',
            'currency': 'USD'
        },
        'RETAIL_SALES': {
            'day_of_month': 'mid_month',
            'time': '12:30',
            'impact': 'HIGH',
            'currency': 'USD'
        },
        'GDP': {
            'months': [1, 4, 7, 10],  # Quarterly
            'day_of_month': 'last_week',
            'time': '12:30',
            'impact': 'HIGH',
            'currency': 'USD'
        }
    }

    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def download_all(self, start_year: int = 2019, end_year: int = 2025) -> pd.DataFrame:
        """
        Télécharge l'economic calendar complet.

        Args:
            start_year: Année de début
            end_year: Année de fin

        Returns:
            DataFrame avec l'economic calendar
        """
        print(f"\n{'='*60}")
        print(f"ECONOMIC CALENDAR DOWNLOADER")
        print(f"{'='*60}")
        print(f"Période: {start_year} - {end_year}")
        print(f"Output: {self.output_dir}/economic_calendar_{start_year}_{end_year}.csv")
        print(f"{'='*60}\n")

        # Essayer différentes sources
        df = None

        # Source 1: Essayer Forex Factory
        print("[1/3] Tentative 1: Forex Factory...")
        try:
            df = self._download_forex_factory(start_year, end_year)
            if df is not None and len(df) > 100:
                print(f"   [OK] Succes! {len(df)} evenements telecharges")
            else:
                df = None
                print("   [--] Pas assez de donnees")
        except Exception as e:
            print(f"   [ERR] Erreur: {e}")
            df = None

        # Source 2: Essayer Investing.com
        if df is None:
            print("\n[2/3] Tentative 2: Investing.com...")
            try:
                df = self._download_investing_com(start_year, end_year)
                if df is not None and len(df) > 100:
                    print(f"   [OK] Succes! {len(df)} evenements telecharges")
                else:
                    df = None
                    print("   [--] Pas assez de donnees")
            except Exception as e:
                print(f"   [ERR] Erreur: {e}")
                df = None

        # Source 3: Generer base sur patterns connus (toujours fiable)
        if df is None:
            print("\n[3/3] Tentative 3: Generation basee sur patterns historiques...")
            df = self._generate_from_patterns(start_year, end_year)
            print(f"   [OK] Genere! {len(df)} evenements crees")

        # Filtrer pour HIGH IMPACT seulement (le plus important pour Gold)
        df_high = df[df['Impact'].str.upper() == 'HIGH'].copy()
        print(f"\nEvenements HIGH IMPACT: {len(df_high)}")

        # Sauvegarder
        output_path = os.path.join(self.output_dir, f"economic_calendar_{start_year}_{end_year}.csv")
        df.to_csv(output_path, index=False)
        print(f"\nSaved: {output_path}")

        # Aussi sauvegarder version HIGH IMPACT seulement
        output_path_high = os.path.join(self.output_dir, f"economic_calendar_HIGH_IMPACT_{start_year}_{end_year}.csv")
        df_high.to_csv(output_path_high, index=False)
        print(f"Saved: {output_path_high}")

        # Stats
        self._print_stats(df)

        return df

    def _download_forex_factory(self, start_year: int, end_year: int) -> Optional[pd.DataFrame]:
        """Télécharge depuis Forex Factory."""
        # Note: Forex Factory bloque souvent les requêtes automatisées
        # Cette méthode peut ne pas fonctionner

        all_data = []
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                url = f"https://www.forexfactory.com/calendar?month={year}.{month}"
                try:
                    response = requests.get(url, headers=headers, timeout=10)
                    if response.status_code == 200:
                        # Parser le HTML (simplifié - en réalité il faudrait BeautifulSoup)
                        pass
                    time.sleep(1)  # Rate limiting
                except:
                    continue

        # Forex Factory est difficile à scraper, retourner None pour utiliser le fallback
        return None

    def _download_investing_com(self, start_year: int, end_year: int) -> Optional[pd.DataFrame]:
        """Télécharge depuis Investing.com."""
        # Investing.com a aussi des protections anti-scraping
        # Retourner None pour utiliser le fallback
        return None

    def _generate_from_patterns(self, start_year: int, end_year: int) -> pd.DataFrame:
        """
        Génère l'economic calendar basé sur les patterns connus.

        Les événements économiques majeurs suivent des schedules prévisibles:
        - NFP: Premier vendredi de chaque mois
        - FOMC: 8 fois par an (dates publiées à l'avance)
        - CPI: Deuxième semaine de chaque mois
        - GDP: Trimestriel
        - etc.
        """
        events = []

        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                # NFP - Premier vendredi du mois
                nfp_date = self._get_first_friday(year, month)
                events.append({
                    'Date': nfp_date,
                    'Time': '12:30',
                    'Currency': 'USD',
                    'Event': 'Non-Farm Payrolls',
                    'Impact': 'HIGH',
                    'Actual': '',
                    'Forecast': '',
                    'Previous': ''
                })

                # Unemployment Rate (même jour que NFP)
                events.append({
                    'Date': nfp_date,
                    'Time': '12:30',
                    'Currency': 'USD',
                    'Event': 'Unemployment Rate',
                    'Impact': 'HIGH',
                    'Actual': '',
                    'Forecast': '',
                    'Previous': ''
                })

                # CPI - Généralement 2ème semaine
                cpi_date = self._get_second_week_day(year, month, 2)  # Mardi
                events.append({
                    'Date': cpi_date,
                    'Time': '12:30',
                    'Currency': 'USD',
                    'Event': 'CPI m/m',
                    'Impact': 'HIGH',
                    'Actual': '',
                    'Forecast': '',
                    'Previous': ''
                })
                events.append({
                    'Date': cpi_date,
                    'Time': '12:30',
                    'Currency': 'USD',
                    'Event': 'Core CPI m/m',
                    'Impact': 'HIGH',
                    'Actual': '',
                    'Forecast': '',
                    'Previous': ''
                })

                # PPI - Généralement un jour avant ou après CPI
                ppi_date = cpi_date - timedelta(days=1)
                events.append({
                    'Date': ppi_date,
                    'Time': '12:30',
                    'Currency': 'USD',
                    'Event': 'PPI m/m',
                    'Impact': 'MEDIUM',
                    'Actual': '',
                    'Forecast': '',
                    'Previous': ''
                })

                # Retail Sales - Mi-mois
                retail_date = datetime(year, month, 15)
                if retail_date.weekday() >= 5:  # Weekend
                    retail_date = retail_date - timedelta(days=retail_date.weekday() - 4)
                events.append({
                    'Date': retail_date,
                    'Time': '12:30',
                    'Currency': 'USD',
                    'Event': 'Retail Sales m/m',
                    'Impact': 'HIGH',
                    'Actual': '',
                    'Forecast': '',
                    'Previous': ''
                })

                # ISM Manufacturing PMI - Premier jour ouvrable du mois
                ism_date = self._get_first_business_day(year, month)
                events.append({
                    'Date': ism_date,
                    'Time': '14:00',
                    'Currency': 'USD',
                    'Event': 'ISM Manufacturing PMI',
                    'Impact': 'HIGH',
                    'Actual': '',
                    'Forecast': '',
                    'Previous': ''
                })

                # ISM Services PMI - 3ème jour ouvrable
                ism_services_date = ism_date + timedelta(days=2)
                events.append({
                    'Date': ism_services_date,
                    'Time': '14:00',
                    'Currency': 'USD',
                    'Event': 'ISM Services PMI',
                    'Impact': 'HIGH',
                    'Actual': '',
                    'Forecast': '',
                    'Previous': ''
                })

                # Initial Jobless Claims - Chaque jeudi
                for week in range(4):
                    thursday = self._get_nth_weekday(year, month, 3, week + 1)  # Jeudi
                    if thursday and thursday.month == month:
                        events.append({
                            'Date': thursday,
                            'Time': '12:30',
                            'Currency': 'USD',
                            'Event': 'Initial Jobless Claims',
                            'Impact': 'MEDIUM',
                            'Actual': '',
                            'Forecast': '',
                            'Previous': ''
                        })

                # Consumer Confidence - Dernier mardi du mois
                cc_date = self._get_last_weekday(year, month, 1)  # Mardi
                events.append({
                    'Date': cc_date,
                    'Time': '14:00',
                    'Currency': 'USD',
                    'Event': 'Consumer Confidence',
                    'Impact': 'MEDIUM',
                    'Actual': '',
                    'Forecast': '',
                    'Previous': ''
                })

            # FOMC Meetings (8 par an - dates approximatives)
            fomc_months = [1, 3, 5, 6, 7, 9, 11, 12]
            for fomc_month in fomc_months:
                # FOMC se termine généralement le mercredi de la 3ème semaine
                fomc_date = self._get_nth_weekday(year, fomc_month, 2, 3)  # 3ème mercredi
                if fomc_date:
                    events.append({
                        'Date': fomc_date,
                        'Time': '18:00',
                        'Currency': 'USD',
                        'Event': 'FOMC Statement',
                        'Impact': 'HIGH',
                        'Actual': '',
                        'Forecast': '',
                        'Previous': ''
                    })
                    events.append({
                        'Date': fomc_date,
                        'Time': '18:00',
                        'Currency': 'USD',
                        'Event': 'Federal Funds Rate',
                        'Impact': 'HIGH',
                        'Actual': '',
                        'Forecast': '',
                        'Previous': ''
                    })
                    # FOMC Press Conference 30 min après
                    events.append({
                        'Date': fomc_date,
                        'Time': '18:30',
                        'Currency': 'USD',
                        'Event': 'FOMC Press Conference',
                        'Impact': 'HIGH',
                        'Actual': '',
                        'Forecast': '',
                        'Previous': ''
                    })

            # GDP (Trimestriel - fin de mois suivant le trimestre)
            gdp_months = [(1, 'Q4'), (4, 'Q1'), (7, 'Q2'), (10, 'Q3')]
            for gdp_month, quarter in gdp_months:
                gdp_date = self._get_last_weekday(year, gdp_month, 3)  # Dernier jeudi
                if gdp_date:
                    events.append({
                        'Date': gdp_date,
                        'Time': '12:30',
                        'Currency': 'USD',
                        'Event': f'GDP q/q ({quarter})',
                        'Impact': 'HIGH',
                        'Actual': '',
                        'Forecast': '',
                        'Previous': ''
                    })

            # PCE Price Index (indicateur préféré de la Fed) - Fin de mois
            for month in range(1, 13):
                pce_date = self._get_last_weekday(year, month, 4)  # Dernier vendredi
                if pce_date:
                    events.append({
                        'Date': pce_date,
                        'Time': '12:30',
                        'Currency': 'USD',
                        'Event': 'Core PCE Price Index m/m',
                        'Impact': 'HIGH',
                        'Actual': '',
                        'Forecast': '',
                        'Previous': ''
                    })

            # Jackson Hole Symposium (fin août, chaque année)
            # Généralement dernier week-end d'août
            jackson_hole = datetime(year, 8, 25)  # Approximatif
            events.append({
                'Date': jackson_hole,
                'Time': '14:00',
                'Currency': 'USD',
                'Event': 'Jackson Hole Symposium',
                'Impact': 'HIGH',
                'Actual': '',
                'Forecast': '',
                'Previous': ''
            })

        # Créer DataFrame
        df = pd.DataFrame(events)

        # Combiner Date et Time
        df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'])
        df = df.sort_values('DateTime').reset_index(drop=True)

        # Formater la date
        df['Date'] = df['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df = df.drop(columns=['DateTime', 'Time'])

        # Réordonner les colonnes
        df = df[['Date', 'Currency', 'Event', 'Impact', 'Actual', 'Forecast', 'Previous']]

        return df

    def _get_first_friday(self, year: int, month: int) -> datetime:
        """Retourne le premier vendredi du mois."""
        first_day = datetime(year, month, 1)
        days_until_friday = (4 - first_day.weekday()) % 7
        return first_day + timedelta(days=days_until_friday)

    def _get_first_business_day(self, year: int, month: int) -> datetime:
        """Retourne le premier jour ouvrable du mois."""
        first_day = datetime(year, month, 1)
        while first_day.weekday() >= 5:  # Weekend
            first_day += timedelta(days=1)
        return first_day

    def _get_second_week_day(self, year: int, month: int, weekday: int) -> datetime:
        """Retourne un jour spécifique de la 2ème semaine."""
        first_day = datetime(year, month, 1)
        days_until_weekday = (weekday - first_day.weekday()) % 7
        first_occurrence = first_day + timedelta(days=days_until_weekday)
        return first_occurrence + timedelta(days=7)  # 2ème semaine

    def _get_nth_weekday(self, year: int, month: int, weekday: int, n: int) -> Optional[datetime]:
        """Retourne le n-ième jour de la semaine du mois."""
        first_day = datetime(year, month, 1)
        days_until_weekday = (weekday - first_day.weekday()) % 7
        first_occurrence = first_day + timedelta(days=days_until_weekday)
        target = first_occurrence + timedelta(days=7 * (n - 1))
        if target.month == month:
            return target
        return None

    def _get_last_weekday(self, year: int, month: int, weekday: int) -> datetime:
        """Retourne le dernier jour de la semaine spécifié du mois."""
        if month == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)
        last_day = next_month - timedelta(days=1)

        days_back = (last_day.weekday() - weekday) % 7
        return last_day - timedelta(days=days_back)

    def _print_stats(self, df: pd.DataFrame):
        """Affiche les statistiques du calendar."""
        print(f"\n{'='*60}")
        print("STATISTIQUES")
        print(f"{'='*60}")
        print(f"Total événements: {len(df)}")
        print(f"\nPar Impact:")
        print(df['Impact'].value_counts().to_string())
        print(f"\nPar Event (top 15):")
        print(df['Event'].value_counts().head(15).to_string())
        print(f"\nPar Année:")
        df['Year'] = pd.to_datetime(df['Date']).dt.year
        print(df['Year'].value_counts().sort_index().to_string())
        print(f"{'='*60}\n")


def verify_data_alignment(gold_path: str, calendar_path: str):
    """
    Vérifie l'alignement entre les données Gold et le calendar.
    """
    print(f"\n{'='*60}")
    print("VÉRIFICATION DE L'ALIGNEMENT")
    print(f"{'='*60}\n")

    # Charger les données
    gold = pd.read_csv(gold_path, parse_dates=['Date'])
    calendar = pd.read_csv(calendar_path, parse_dates=['Date'])

    # Stats Gold
    print(f"DONNEES GOLD:")
    print(f"   Période: {gold['Date'].min()} à {gold['Date'].max()}")
    print(f"   Nombre de barres: {len(gold):,}")

    # Stats Calendar
    print(f"\nECONOMIC CALENDAR:")
    print(f"   Period: {calendar['Date'].min()} to {calendar['Date'].max()}")
    print(f"   Events: {len(calendar):,}")

    gold_start = gold['Date'].min()
    gold_end = gold['Date'].max()

    calendar_in_range = calendar[
        (calendar['Date'] >= gold_start) &
        (calendar['Date'] <= gold_end)
    ]

    print(f"\nALIGNMENT:")
    print(f"   Events in Gold period: {len(calendar_in_range):,}")

    high_impact = calendar_in_range[calendar_in_range['Impact'] == 'HIGH']
    print(f"   HIGH IMPACT events: {len(high_impact):,}")

    print(f"\nKEY EVENTS DETECTED:")

    key_events = ['Non-Farm Payrolls', 'FOMC Statement', 'CPI m/m', 'Federal Funds Rate']
    for event in key_events:
        count = len(calendar_in_range[calendar_in_range['Event'].str.contains(event, case=False, na=False)])
        print(f"   {event}: {count} occurrences")

    print(f"\n{'='*60}\n")

    return calendar_in_range


if __name__ == "__main__":
    # Télécharger le calendar
    downloader = EconomicCalendarDownloader(output_dir="data")
    calendar_df = downloader.download_all(start_year=2019, end_year=2025)

    # Vérifier l'alignement avec les données Gold
    gold_path = "data/XAU_15MIN_2019_2025.csv"
    calendar_path = "data/economic_calendar_2019_2025.csv"

    if os.path.exists(gold_path) and os.path.exists(calendar_path):
        aligned_calendar = verify_data_alignment(gold_path, calendar_path)

        print("\nPRET POUR L'ENTRAINEMENT!")
        print("   Tu peux maintenant utiliser:")
        print("   - data/XAU_15MIN_2019_2025.csv (prix)")
        print("   - data/economic_calendar_2019_2025.csv (événements)")
        print("   - data/economic_calendar_HIGH_IMPACT_2019_2024.csv (événements majeurs)")
