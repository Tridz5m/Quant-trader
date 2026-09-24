import numpy as np

from quant_trader.labeling import triple_barrier


def _run(o, h, l, c, spread=0.0, horizon=3):
    n = len(c)
    return triple_barrier(
        np.array(o, float), np.array(h, float), np.array(l, float), np.array(c, float),
        np.ones(n), np.full(n, spread), sl_mult=1.0, tp_mult=2.0, horizon=horizon, cost=0.0,
    )


def test_long_target_hit():
    out = _run([100, 100, 101], [100, 102.5, 101], [100, 99.5, 100], [100, 101, 100.5], horizon=2)
    assert out["long_win"][0] == 1.0
    assert out["long_r"][0] == 2.0
    assert out["long_exit"][0] == 1


def test_stop_wins_when_both_levels_in_one_bar():
    out = _run([100, 100, 100], [100, 102.5, 100], [100, 98.5, 100], [100, 100, 100], horizon=2)
    assert out["long_win"][0] == 0.0
    assert out["long_r"][0] == -1.0
    assert out["short_r"][0] == -1.0


def test_gap_through_stop_fills_at_open():
    out = _run([100, 98.0, 98], [100, 98.2, 98], [100, 97.5, 98], [100, 98, 98], horizon=2)
    assert np.isclose(out["long_r"][0], -2.0)


def test_time_exit_and_unknown_tail():
    o = [100, 100.2, 100.4, 100.5, 100.5]
    c = [100, 100.2, 100.4, 100.5, 100.5]
    h = [x + 0.1 for x in c]
    l = [x - 0.1 for x in c]
    out = _run(o, h, l, c, horizon=3)
    assert out["long_exit"][0] == 3
    assert np.isclose(out["long_r"][0], 0.5)
    assert np.isclose(out["short_r"][0], -0.5)
    assert np.isnan(out["long_r"][-1]) and np.isnan(out["long_win"][-1])


def test_short_uses_ask_with_spread():
    # Bid high 100.8 + spread 0.3 = ask 101.1 >= stop 101 -> short stopped out.
    out = _run([100, 100, 100], [100, 100.8, 100], [100, 99.9, 100], [100, 100, 100], spread=0.3, horizon=2)
    assert out["short_r"][0] == -1.0
    # Long pays the spread on entry: entry 100.3, stop 99.3 not hit, target 102.3 not hit.
    assert out["long_exit"][0] == 2
