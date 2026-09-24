from datetime import datetime, timedelta, timezone

import pytest

from quant_trader.config import NewsConfig
from quant_trader.news import NewsCalendar, parse_calendar

NOW = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)

# Same shape as the ForexFactory weekly JSON feed (times carry a UTC offset).
FEED = [
    {"title": "Non-Farm Employment Change", "country": "USD", "date": "2026-09-25T08:30:00-04:00", "impact": "High",
     "forecast": "150K", "previous": "142K"},
    {"title": "Unemployment Claims", "country": "USD", "date": "2026-09-25T08:30:00-04:00", "impact": "Medium"},
    {"title": "German ifo Business Climate", "country": "EUR", "date": "2026-09-25T04:00:00-04:00", "impact": "High"},
    {"title": "FOMC Statement", "country": "USD", "date": "2026-09-30T14:00:00-04:00", "impact": "High"},
    {"title": "broken", "country": "USD", "date": "not a date", "impact": "High"},
]


def calendar(tmp_path, fetch=lambda url: FEED, **cfg):
    clock = {"now": NOW}
    cal = NewsCalendar(NewsConfig(**cfg), tmp_path / "news.json", fetch=fetch, clock=lambda: clock["now"])
    return cal, clock


def test_parse_converts_to_utc_and_skips_bad_rows():
    events = parse_calendar(FEED)
    assert len(events) == 4
    nfp = next(e for e in events if "Non-Farm" in e.title)
    assert nfp.time == datetime(2026, 9, 25, 12, 30, tzinfo=timezone.utc)
    assert nfp.currency == "USD" and nfp.impact == "High"


def test_only_high_impact_usd_events_block_trading(tmp_path):
    cal, _ = calendar(tmp_path)
    cal.refresh()
    assert [e.title for e in cal.relevant()] == ["Non-Farm Employment Change", "FOMC Statement"]
    nfp = datetime(2026, 9, 25, 12, 30, tzinfo=timezone.utc)
    assert cal.blackout(nfp - timedelta(minutes=31)) is None
    assert cal.blackout(nfp - timedelta(minutes=30)).title == "Non-Farm Employment Change"
    assert cal.blackout(nfp + timedelta(minutes=30)) is not None
    assert cal.blackout(nfp + timedelta(minutes=31)) is None
    # EUR news is ignored for gold by default.
    assert cal.blackout(datetime(2026, 9, 25, 8, 0, tzinfo=timezone.utc)) is None
    assert cal.next_event(NOW).title == "Non-Farm Employment Change"


def test_disabled_filter_never_blocks(tmp_path):
    cal, _ = calendar(tmp_path, enabled=False)
    cal.refresh()
    assert cal.blackout(datetime(2026, 9, 25, 12, 30, tzinfo=timezone.utc)) is None
    assert cal.status(NOW) == {"enabled": False}


def test_download_is_cached_and_reused_when_the_feed_fails(tmp_path):
    calls = []

    def fetch(url):
        calls.append(url)
        return FEED

    cal, clock = calendar(tmp_path, fetch=fetch)
    cal.refresh()
    cal.refresh()  # still fresh: no second download
    assert len(calls) == 1 and (tmp_path / "news.json").exists()

    def broken(url):
        raise OSError("no internet")

    later, clock2 = calendar(tmp_path, fetch=broken)
    clock2["now"] = NOW + timedelta(hours=5)
    later.refresh()
    assert later.error and len(later.relevant()) == 2  # cached copy still protects
    assert later.status(NOW)["next"].startswith("USD Non-Farm")


def test_failed_download_is_retried_after_a_pause(tmp_path):
    calls = []

    def flaky(url):
        calls.append(url)
        raise OSError("timeout")

    cal, clock = calendar(tmp_path, fetch=flaky)
    cal.refresh()
    cal.refresh()
    assert len(calls) == 1
    clock["now"] = NOW + timedelta(minutes=31)
    cal.refresh()
    assert len(calls) == 2


class FakeNews:
    """A calendar with one high-impact event at a fixed time."""

    def __init__(self, event_time, close_before=False):
        from quant_trader.news import NewsEvent

        self.cfg = NewsConfig(close_positions_before=close_before)
        self.event = NewsEvent(event_time, "USD", "CPI m/m", "High")

    def refresh(self):
        pass

    def blackout(self, now=None):
        before, after = timedelta(minutes=30), timedelta(minutes=30)
        return self.event if self.event.time - before <= now <= self.event.time + after else None

    def status(self, now=None):
        return {"enabled": True, "blackout": None, "next": self.event.describe()}


@pytest.fixture
def bot_with_news(cfg, noise_bars):
    from test_bot import StubModel, make_bot

    def build(minutes_to_event, close_before=False):
        bot, broker, journal = make_bot(cfg, noise_bars, StubModel())
        bot.news = FakeNews(datetime.now(timezone.utc) + timedelta(minutes=minutes_to_event), close_before)
        cfg.news.close_positions_before = close_before
        return bot, broker, journal

    return build


def test_bot_skips_entries_around_news(bot_with_news):
    bot, broker, _ = bot_with_news(minutes_to_event=10)
    report = bot.run_cycle()
    assert report["action"] == "none" and "high-impact news" in report["reason"] and "CPI" in report["reason"]
    assert broker.sent == []


def test_bot_trades_when_news_is_far_away(bot_with_news):
    bot, _, _ = bot_with_news(minutes_to_event=300)
    assert bot.run_cycle()["action"] == "open"


def test_bot_can_close_positions_before_news(bot_with_news):
    bot, broker, journal = bot_with_news(minutes_to_event=300, close_before=True)
    assert bot.run_cycle()["action"] == "open"
    bot.news = FakeNews(datetime.now(timezone.utc) + timedelta(minutes=10), close_before=True)
    broker.advance()
    bot.run_cycle()
    assert broker.positions() == []
    (t,) = journal.closed_trades()
    assert t["close_reason"] == "news"
