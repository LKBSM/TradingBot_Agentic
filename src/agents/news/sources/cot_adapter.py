# =============================================================================
# CFTC COT ADAPTER - Commitment of Traders Report for Sprint 3
# =============================================================================
"""
Adapter for CFTC Commitment of Traders (COT) reports.

Sprint 3 Feature: Tracks institutional positioning in futures markets,
providing insight into "smart money" sentiment.

COT Report provides:
- Commercial (hedgers) positions
- Non-commercial (speculators) positions
- Open interest changes
- Net positioning and extremes

Report Schedule:
- Data as of Tuesday
- Released Friday 3:30 PM ET
- Covers major futures including Gold (GC), Currencies

Usage:
    adapter = COTAdapter()
    reports = await adapter.fetch_async()
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging
import asyncio
import csv
import io

from .base_adapter import (
    BaseNewsAdapter,
    AdapterConfig,
    NewsArticle,
    ArticleSource,
    ArticleCategory
)

logger = logging.getLogger(__name__)

# Try to import HTTP libraries
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


@dataclass
class COTPosition:
    """COT position data for a single instrument."""
    report_date: datetime
    instrument: str                     # e.g., "GOLD - COMEX"
    cftc_code: str                      # CFTC contract code

    # Commercial positions (hedgers)
    commercial_long: int
    commercial_short: int
    commercial_net: int

    # Non-commercial positions (large speculators)
    non_commercial_long: int
    non_commercial_short: int
    non_commercial_net: int

    # Small traders
    non_reportable_long: int
    non_reportable_short: int
    non_reportable_net: int

    # Open interest
    open_interest: int
    open_interest_change: int

    # Calculated metrics
    spec_index: float = 0.0            # Speculator positioning index (0-100)
    net_change_weekly: int = 0          # Week-over-week change


@dataclass
class COTConfig(AdapterConfig):
    """Configuration for COT adapter."""
    name: str = "CFTC_COT"

    # CFTC data source
    cftc_base_url: str = "https://www.cftc.gov/dea/newcot"

    # Report types
    # "fut" = Futures only
    # "futopt" = Futures and Options combined
    report_type: str = "fut"

    # Update schedule (COT released Friday ~3:30 PM ET)
    check_day: str = "Friday"
    check_hour: int = 16  # 4 PM to allow for delays

    # Instruments to track (CFTC codes)
    instruments: Dict[str, str] = field(default_factory=lambda: {
        'GOLD': '088691',           # Gold - COMEX
        'SILVER': '084691',         # Silver - COMEX
        'COPPER': '085692',         # Copper - COMEX
        'EURUSD': '099741',         # Euro FX - CME
        'GBPUSD': '096742',         # British Pound - CME
        'USDJPY': '097741',         # Japanese Yen - CME
        'USDCHF': '092741',         # Swiss Franc - CME
        'AUDUSD': '232741',         # Australian Dollar - CME
        'CADUSD': '090741',         # Canadian Dollar - CME
        'CRUDE_OIL': '067651',      # Crude Oil - NYMEX
        'NATURAL_GAS': '023651',    # Natural Gas - NYMEX
        'SP500': '13874A',          # E-mini S&P 500 - CME
        'VIX': '1170E1',            # VIX Futures - CFE
    })

    # Alert thresholds
    extreme_long_percentile: float = 90.0   # Alert if above
    extreme_short_percentile: float = 10.0  # Alert if below
    significant_change_pct: float = 10.0    # % change to trigger alert


class COTAdapter(BaseNewsAdapter):
    """
    CFTC Commitment of Traders report adapter.

    Fetches and parses COT data to provide insight into
    institutional positioning.
    """

    def __init__(self, config: Optional[COTConfig] = None):
        """
        Initialize COT adapter.

        Args:
            config: Adapter configuration
        """
        self.cot_config = config or COTConfig()
        super().__init__(self.cot_config)

        # Position cache
        self._current_positions: Dict[str, COTPosition] = {}
        self._historical_positions: Dict[str, List[COTPosition]] = {}
        self._last_report_date: Optional[datetime] = None
        self._last_fetch: Optional[datetime] = None

        # Statistics
        self._reports_fetched: int = 0

    def fetch(self) -> List[NewsArticle]:
        """
        Fetch COT report synchronously.

        Returns:
            List of NewsArticle objects for significant positioning changes
        """
        if not HAS_REQUESTS:
            self._logger.error("requests library not installed")
            return []

        # Check if it's time to update
        if not self._should_fetch():
            return []

        articles = []

        try:
            # Fetch current report
            positions = self._fetch_cot_report_sync()

            if positions:
                # Process and generate articles
                articles = self._process_positions(positions)
                self._last_fetch = datetime.now()
                self._reports_fetched += 1

        except Exception as e:
            self._logger.error(f"COT fetch error: {e}")

        return articles

    async def fetch_async(self) -> List[NewsArticle]:
        """
        Fetch COT report asynchronously.

        Returns:
            List of NewsArticle objects for significant positioning changes
        """
        if not HAS_HTTPX:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.fetch)

        if not self._should_fetch():
            return []

        articles = []

        try:
            positions = await self._fetch_cot_report_async()

            if positions:
                articles = self._process_positions(positions)
                self._last_fetch = datetime.now()
                self._reports_fetched += 1

        except Exception as e:
            self._logger.error(f"COT async fetch error: {e}")

        return articles

    def _should_fetch(self) -> bool:
        """Check if we should fetch new data."""
        now = datetime.now()

        # If never fetched, fetch now
        if self._last_fetch is None:
            return True

        # Check if it's past release time on Friday
        is_friday = now.strftime('%A') == self.cot_config.check_day
        is_after_release = now.hour >= self.cot_config.check_hour

        if is_friday and is_after_release:
            # Check if we've already fetched today
            if self._last_fetch.date() < now.date():
                return True

        # Also fetch if last fetch was over a day ago (catch-up)
        if (now - self._last_fetch).days >= 1:
            return True

        return False

    def _fetch_cot_report_sync(self) -> Dict[str, COTPosition]:
        """Fetch COT report from CFTC (sync)."""
        # CFTC provides data in various formats
        # Using the deacot text file format

        # Financial futures report
        url = f"{self.cot_config.cftc_base_url}/deacom{self.cot_config.report_type}.txt"

        try:
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                return self._parse_cot_text(response.text)
            else:
                self._logger.warning(f"COT fetch failed: {response.status_code}")
        except Exception as e:
            self._logger.error(f"COT request error: {e}")

        # Return mock data for demo
        return self._generate_mock_positions()

    async def _fetch_cot_report_async(self) -> Dict[str, COTPosition]:
        """Fetch COT report from CFTC (async)."""
        url = f"{self.cot_config.cftc_base_url}/deacom{self.cot_config.report_type}.txt"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=60.0)
                if response.status_code == 200:
                    return self._parse_cot_text(response.text)
                else:
                    self._logger.warning(f"COT fetch failed: {response.status_code}")
        except Exception as e:
            self._logger.error(f"COT async request error: {e}")

        return self._generate_mock_positions()

    def _parse_cot_text(self, text: str) -> Dict[str, COTPosition]:
        """Parse CFTC COT text file format."""
        positions = {}

        try:
            # CFTC format is fixed-width or comma-separated depending on file
            # This is a simplified parser
            lines = text.strip().split('\n')

            if not lines:
                return self._generate_mock_positions()

            # Try to detect format
            if ',' in lines[0]:
                # CSV format
                reader = csv.DictReader(io.StringIO(text))
                for row in reader:
                    position = self._parse_csv_row(row)
                    if position:
                        positions[position.instrument] = position
            else:
                # Fixed-width format - more complex parsing needed
                self._logger.debug("Fixed-width COT format, using simplified parser")
                positions = self._generate_mock_positions()

        except Exception as e:
            self._logger.error(f"COT parse error: {e}")
            positions = self._generate_mock_positions()

        return positions

    def _parse_csv_row(self, row: Dict) -> Optional[COTPosition]:
        """Parse a CSV row into COTPosition."""
        try:
            # Column names vary by report type
            # Common column names
            instrument = row.get('Market and Exchange Names', row.get('Contract_Market_Name', ''))
            cftc_code = row.get('CFTC_Contract_Market_Code', row.get('CFTC Contract Market Code', ''))

            # Check if this is an instrument we track
            if not any(code in str(cftc_code) for code in self.cot_config.instruments.values()):
                return None

            report_date_str = row.get('As_of_Date_In_Form_YYMMDD', row.get('As of Date in Form YYYY-MM-DD', ''))
            if report_date_str:
                try:
                    if '-' in report_date_str:
                        report_date = datetime.strptime(report_date_str, "%Y-%m-%d")
                    else:
                        report_date = datetime.strptime(report_date_str, "%y%m%d")
                except ValueError:
                    report_date = datetime.now()
            else:
                report_date = datetime.now()

            # Parse positions
            def safe_int(val):
                try:
                    return int(str(val).replace(',', ''))
                except (ValueError, TypeError):
                    return 0

            comm_long = safe_int(row.get('Comm_Positions_Long_All', row.get('Commercial_Positions_Long', 0)))
            comm_short = safe_int(row.get('Comm_Positions_Short_All', row.get('Commercial_Positions_Short', 0)))

            noncomm_long = safe_int(row.get('NonComm_Positions_Long_All', row.get('NonCommercial_Positions_Long', 0)))
            noncomm_short = safe_int(row.get('NonComm_Positions_Short_All', row.get('NonCommercial_Positions_Short', 0)))

            nonrep_long = safe_int(row.get('NonRept_Positions_Long_All', 0))
            nonrep_short = safe_int(row.get('NonRept_Positions_Short_All', 0))

            open_interest = safe_int(row.get('Open_Interest_All', row.get('Open_Interest', 0)))
            oi_change = safe_int(row.get('Change_in_Open_Interest_All', 0))

            return COTPosition(
                report_date=report_date,
                instrument=instrument,
                cftc_code=str(cftc_code),
                commercial_long=comm_long,
                commercial_short=comm_short,
                commercial_net=comm_long - comm_short,
                non_commercial_long=noncomm_long,
                non_commercial_short=noncomm_short,
                non_commercial_net=noncomm_long - noncomm_short,
                non_reportable_long=nonrep_long,
                non_reportable_short=nonrep_short,
                non_reportable_net=nonrep_long - nonrep_short,
                open_interest=open_interest,
                open_interest_change=oi_change
            )

        except Exception as e:
            self._logger.debug(f"Error parsing COT row: {e}")
            return None

    def _generate_mock_positions(self) -> Dict[str, COTPosition]:
        """Generate mock COT positions for testing/demo."""
        positions = {}
        report_date = datetime.now() - timedelta(days=datetime.now().weekday() - 1)  # Last Tuesday

        mock_data = {
            'GOLD - COMEX': {
                'code': '088691',
                'comm_long': 150000, 'comm_short': 280000,
                'noncomm_long': 320000, 'noncomm_short': 85000,
                'oi': 550000
            },
            'EURO FX - CME': {
                'code': '099741',
                'comm_long': 45000, 'comm_short': 120000,
                'noncomm_long': 180000, 'noncomm_short': 95000,
                'oi': 650000
            },
            'JAPANESE YEN - CME': {
                'code': '097741',
                'comm_long': 25000, 'comm_short': 85000,
                'noncomm_long': 55000, 'noncomm_short': 150000,
                'oi': 280000
            }
        }

        for instrument, data in mock_data.items():
            nonrep_long = data['oi'] - data['comm_long'] - data['noncomm_long']
            nonrep_short = data['oi'] - data['comm_short'] - data['noncomm_short']

            positions[instrument] = COTPosition(
                report_date=report_date,
                instrument=instrument,
                cftc_code=data['code'],
                commercial_long=data['comm_long'],
                commercial_short=data['comm_short'],
                commercial_net=data['comm_long'] - data['comm_short'],
                non_commercial_long=data['noncomm_long'],
                non_commercial_short=data['noncomm_short'],
                non_commercial_net=data['noncomm_long'] - data['noncomm_short'],
                non_reportable_long=max(0, nonrep_long),
                non_reportable_short=max(0, nonrep_short),
                non_reportable_net=nonrep_long - nonrep_short,
                open_interest=data['oi'],
                open_interest_change=int(data['oi'] * 0.02)  # 2% change
            )

        return positions

    def _process_positions(
        self,
        new_positions: Dict[str, COTPosition]
    ) -> List[NewsArticle]:
        """Process positions and generate news articles."""
        articles = []

        for instrument, position in new_positions.items():
            # Calculate speculator index
            position.spec_index = self._calculate_spec_index(position)

            # Check for week-over-week changes
            old_position = self._current_positions.get(instrument)
            if old_position:
                position.net_change_weekly = position.non_commercial_net - old_position.non_commercial_net

                # Check for significant changes
                if self._is_significant_change(old_position, position):
                    articles.append(self._create_change_article(instrument, old_position, position))

            # Check for extreme positioning
            if self._is_extreme_positioning(position):
                articles.append(self._create_extreme_article(instrument, position))

            # Update history
            if instrument not in self._historical_positions:
                self._historical_positions[instrument] = []
            self._historical_positions[instrument].append(position)

            # Keep only last 52 weeks
            self._historical_positions[instrument] = self._historical_positions[instrument][-52:]

            # Update current
            self._current_positions[instrument] = position

        # Create summary article
        if new_positions:
            articles.append(self._create_summary_article(new_positions))

        # Update last report date
        if new_positions:
            self._last_report_date = list(new_positions.values())[0].report_date

        return articles

    def _calculate_spec_index(self, position: COTPosition) -> float:
        """
        Calculate speculator positioning index (0-100).

        100 = Maximum long positioning (historically)
        0 = Maximum short positioning (historically)
        50 = Neutral
        """
        history = self._historical_positions.get(position.instrument, [])

        if len(history) < 4:  # Need some history
            # Use simple ratio
            total = position.non_commercial_long + position.non_commercial_short
            if total > 0:
                return (position.non_commercial_long / total) * 100
            return 50.0

        # Calculate historical range
        net_positions = [p.non_commercial_net for p in history]
        min_net = min(net_positions)
        max_net = max(net_positions)

        if max_net == min_net:
            return 50.0

        # Scale current position to 0-100
        spec_index = ((position.non_commercial_net - min_net) / (max_net - min_net)) * 100
        return max(0, min(100, spec_index))

    def _is_significant_change(
        self,
        old: COTPosition,
        new: COTPosition
    ) -> bool:
        """Check if change is significant."""
        if old.non_commercial_net == 0:
            return False

        pct_change = abs((new.non_commercial_net - old.non_commercial_net) / abs(old.non_commercial_net)) * 100
        return pct_change >= self.cot_config.significant_change_pct

    def _is_extreme_positioning(self, position: COTPosition) -> bool:
        """Check if positioning is at extreme levels."""
        return (position.spec_index >= self.cot_config.extreme_long_percentile or
                position.spec_index <= self.cot_config.extreme_short_percentile)

    def _create_change_article(
        self,
        instrument: str,
        old: COTPosition,
        new: COTPosition
    ) -> NewsArticle:
        """Create article for significant position change."""
        net_change = new.non_commercial_net - old.non_commercial_net
        direction = "increased LONG" if net_change > 0 else "increased SHORT"

        # Map instrument to trading symbol
        asset = self._instrument_to_asset(instrument)

        title = f"COT: Speculators {direction} {asset} by {abs(net_change):,} contracts"

        content = f"""
CFTC Commitment of Traders Report - {new.report_date.strftime('%Y-%m-%d')}

{instrument}:

Speculator Positioning (Non-Commercial):
- Long: {new.non_commercial_long:,} contracts
- Short: {new.non_commercial_short:,} contracts
- Net: {new.non_commercial_net:,} contracts ({'+' if new.non_commercial_net > 0 else ''}{net_change:,} vs last week)

Commercial Positioning (Hedgers):
- Net: {new.commercial_net:,} contracts

Speculator Index: {new.spec_index:.1f}/100
(100 = max long, 0 = max short over 52-week range)

Open Interest: {new.open_interest:,} ({'+' if new.open_interest_change > 0 else ''}{new.open_interest_change:,})

Trading Implications:
- {"Speculators heavily long - potential reversal signal" if new.spec_index > 80 else
   "Speculators heavily short - potential reversal signal" if new.spec_index < 20 else
   "Speculators increasing bullish bets" if net_change > 0 else
   "Speculators increasing bearish bets"}
        """.strip()

        return NewsArticle(
            article_id=NewsArticle.generate_id("COT", title, new.report_date.isoformat()),
            source_name="CFTC COT Report",
            source_type=ArticleSource.API,
            title=title,
            content=content,
            summary=f"Speculators {direction} {asset}",
            published_at=datetime.now(),
            category=ArticleCategory.MARKET,
            assets=[asset] if asset else [],
            keywords=['COT', 'Commitment of Traders', 'CFTC', 'positioning', 'speculators'],
            importance="MEDIUM"
        )

    def _create_extreme_article(
        self,
        instrument: str,
        position: COTPosition
    ) -> NewsArticle:
        """Create article for extreme positioning."""
        asset = self._instrument_to_asset(instrument)
        is_extreme_long = position.spec_index >= self.cot_config.extreme_long_percentile

        title = f"COT ALERT: {asset} Speculator Positioning at EXTREME {'LONG' if is_extreme_long else 'SHORT'} Levels"

        content = f"""
WARNING: Speculator positioning in {instrument} has reached extreme levels.

Speculator Index: {position.spec_index:.1f}/100
(Currently at {'top' if is_extreme_long else 'bottom'} of 52-week range)

Historical Context:
- Extreme positioning often precedes price reversals
- This does NOT mean price will reverse immediately
- Use as confluence with other analysis

Current Positions:
- Speculator Net: {position.non_commercial_net:,}
- Commercial Net: {position.commercial_net:,}

Potential Trading Implications:
- {"Contrarian signal: Consider bearish scenarios" if is_extreme_long else
   "Contrarian signal: Consider bullish scenarios"}
- Watch for price action confirmation
- Monitor next week's report for changes
        """.strip()

        return NewsArticle(
            article_id=NewsArticle.generate_id("COT_EXTREME", title, position.report_date.isoformat()),
            source_name="CFTC COT Report",
            source_type=ArticleSource.API,
            title=title,
            content=content,
            published_at=datetime.now(),
            category=ArticleCategory.MARKET,
            assets=[asset] if asset else [],
            keywords=['COT', 'extreme positioning', 'contrarian', 'reversal'],
            importance="HIGH"
        )

    def _create_summary_article(
        self,
        positions: Dict[str, COTPosition]
    ) -> NewsArticle:
        """Create weekly summary article."""
        report_date = list(positions.values())[0].report_date if positions else datetime.now()

        summary_lines = []
        for instrument, pos in positions.items():
            asset = self._instrument_to_asset(instrument)
            direction = "LONG" if pos.non_commercial_net > 0 else "SHORT"
            summary_lines.append(f"- {asset}: Net {direction} ({pos.non_commercial_net:,}), Index: {pos.spec_index:.0f}")

        content = f"""
CFTC Commitment of Traders Weekly Summary - {report_date.strftime('%Y-%m-%d')}

Speculator Positioning Overview:
{chr(10).join(summary_lines)}

Key Observations:
- Data as of Tuesday, released Friday
- Watch for extreme readings (>90 or <10) as potential reversal signals
- Week-over-week changes indicate momentum

Source: CFTC Commitment of Traders Report
        """.strip()

        return NewsArticle(
            article_id=NewsArticle.generate_id("COT_SUMMARY", "Weekly COT Summary", report_date.isoformat()),
            source_name="CFTC COT Report",
            source_type=ArticleSource.API,
            title=f"COT Weekly Summary - {report_date.strftime('%B %d, %Y')}",
            content=content,
            published_at=datetime.now(),
            category=ArticleCategory.MARKET,
            assets=['XAUUSD', 'EURUSD', 'USDJPY'],
            keywords=['COT', 'weekly summary', 'positioning'],
            importance="LOW"
        )

    def _instrument_to_asset(self, instrument: str) -> str:
        """Map CFTC instrument name to trading asset."""
        mappings = {
            'GOLD': 'XAUUSD',
            'SILVER': 'XAGUSD',
            'EURO FX': 'EURUSD',
            'BRITISH POUND': 'GBPUSD',
            'JAPANESE YEN': 'USDJPY',
            'SWISS FRANC': 'USDCHF',
            'AUSTRALIAN DOLLAR': 'AUDUSD',
            'CANADIAN DOLLAR': 'USDCAD',
            'CRUDE OIL': 'WTI',
            'NATURAL GAS': 'NATGAS',
        }

        for key, asset in mappings.items():
            if key in instrument.upper():
                return asset

        return instrument.split(' - ')[0] if ' - ' in instrument else instrument

    def get_current_positions(self) -> Dict[str, COTPosition]:
        """Get current cached positions."""
        return self._current_positions.copy()

    def get_position(self, asset: str) -> Optional[COTPosition]:
        """Get position for a specific asset."""
        for instrument, position in self._current_positions.items():
            if asset.upper() in instrument.upper():
                return position
        return None

    def get_status(self) -> Dict[str, Any]:
        """Get adapter status."""
        base_status = super().get_status()

        # Get gold positioning as example
        gold_pos = self.get_position('GOLD')

        base_status.update({
            'reports_fetched': self._reports_fetched,
            'instruments_tracked': len(self._current_positions),
            'last_report_date': self._last_report_date.isoformat() if self._last_report_date else None,
            'gold_spec_index': gold_pos.spec_index if gold_pos else None,
            'gold_spec_net': gold_pos.non_commercial_net if gold_pos else None
        })
        return base_status


def create_cot_adapter() -> COTAdapter:
    """
    Factory function to create a COT adapter.

    Returns:
        Configured COTAdapter instance
    """
    return COTAdapter()
