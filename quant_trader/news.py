"""High-impact economic news filter.

Gold reacts violently to US news (NFP, CPI, FOMC, ...): spreads widen, price
spikes through stops and fills slip. The bot reads the free weekly economic
calendar published for ForexFactory, opens no new trades from
``minutes_before`` to ``minutes_after`` around matching events and can
optionally close open trades before them.

The calendar is cached on disk, so a temporary download problem falls back
to the last copy. Backtests cannot apply this filter (the feed only covers
the current week).
"""

from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from .config import NewsConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class NewsEvent:
    time: datetime  # UTC
    currency: str
    title: str
    impact: str

    def describe(self) -> str:
        return f"{self.currency} {self.title} at {self.time:%a %H:%M} UTC"


def parse_calendar(items: list[dict]) -> list[NewsEvent]:
    """Events from the ForexFactory weekly JSON (times with UTC offsets)."""
    events = []
    for item in items or []:
        try:
            when = datetime.fromisoformat(str(item["date"]).replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        events.append(
            NewsEvent(
                time=when.astimezone(timezone.utc),
                currency=str(item.get("country", "")).upper(),
                title=str(item.get("title", "")).strip(),
                impact=str(item.get("impact", "")).strip(),
            )
        )
    return sorted(events, key=lambda e: e.time)


def download(url: str, timeout: float = 15.0) -> list[dict]:
    req = urllib.request.Request(url, headers={"User-Agent": "QuantTrader/1.1 (+economic calendar filter)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class NewsCalendar:
    def __init__(
        self,
        cfg: NewsConfig,
        cache_path: Path,
        fetch: Callable[[str], list[dict]] = download,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self.cfg = cfg
        self.cache_path = Path(cache_path)
        self.fetch = fetch
        self.clock = clock
        self.events: list[NewsEvent] = []
        self.loaded_at: datetime | None = None  # when the events were downloaded
        self.last_attempt: datetime | None = None
        self.error = ""
        self._load_cache()

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled)

    def _load_cache(self) -> None:
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            self.events = parse_calendar(data["items"])
            self.loaded_at = datetime.fromisoformat(data["downloaded"])
        except (OSError, KeyError, ValueError):
            pass

    def refresh(self, force: bool = False) -> None:
        """Download the calendar when the cached copy is older than ``refresh_hours``."""
        if not self.enabled:
            return
        now = self.clock()
        fresh = self.loaded_at is not None and now - self.loaded_at < timedelta(hours=self.cfg.refresh_hours)
        retry_wait = self.last_attempt is not None and now - self.last_attempt < timedelta(minutes=30)
        if not force and (fresh or retry_wait):
            return
        self.last_attempt = now
        try:
            items = self.fetch(self.cfg.feed_url)
            events = parse_calendar(items)
        except Exception as exc:
            self.error = f"news calendar download failed ({exc})"
            log.warning("%s; using %s", self.error, "the cached copy" if self.events else "no calendar")
            return
        self.events, self.loaded_at, self.error = events, now, ""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps({"downloaded": now.isoformat(), "items": items}), encoding="utf-8")
        except OSError:
            pass
        log.info("News calendar loaded: %d high-impact events this week", len(self.relevant()))

    def relevant(self) -> list[NewsEvent]:
        currencies = {c.upper() for c in self.cfg.currencies}
        impacts = {i.lower() for i in self.cfg.impacts}
        return [e for e in self.events if e.currency in currencies and e.impact.lower() in impacts]

    def blackout(self, now: datetime | None = None) -> NewsEvent | None:
        """The event whose no-trade window contains ``now``, if any."""
        if not self.enabled:
            return None
        now = now or self.clock()
        before = timedelta(minutes=self.cfg.minutes_before)
        after = timedelta(minutes=self.cfg.minutes_after)
        for e in self.relevant():
            if e.time - before <= now <= e.time + after:
                return e
        return None

    def next_event(self, now: datetime | None = None) -> NewsEvent | None:
        now = now or self.clock()
        return next((e for e in self.relevant() if e.time >= now), None)

    def status(self, now: datetime | None = None) -> dict:
        """Summary for the app."""
        now = now or self.clock()
        if not self.enabled:
            return {"enabled": False}
        blk = self.blackout(now)
        nxt = self.next_event(now)
        return {
            "enabled": True,
            "loaded": self.loaded_at is not None,
            "error": self.error,
            "blackout": blk.describe() if blk else None,
            "blackout_until": (blk.time + timedelta(minutes=self.cfg.minutes_after)).isoformat() if blk else None,
            "next": nxt.describe() if nxt else None,
            "next_time": nxt.time.isoformat() if nxt else None,
        }
