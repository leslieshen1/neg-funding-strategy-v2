"""Deterministic contract checks. Invented prices/quantities are never performance evidence."""
import ast
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'repro/lib'))
import engine_v12s as E
from inventory import convert_inventory, load_caps
from run_backtest import Data


def replay(qty=10., dry_from=None, pre=False, impact=0., obs=0., volumes=None,
           events=None, rates=None, intervals=None, start=0, end=12,
           entry_rates=None, allowed=None, missing_price_hours=(), price_increase_from=None,
           account_transfers=False, **overrides):
    T, N = 16, 1
    if rates is not None: N = np.asarray(rates).shape[1]
    a = lambda x: np.full((T, N), x, dtype=float)
    sv, fv, volume = a(100), a(99.5), a(1e6)
    if price_increase_from is not None:
        sv[price_increase_from:] += 50; fv[price_increase_from:] += 50
    for hour in missing_price_hours:
        sv[hour] = 0; fv[hour] = 0
    if volumes is not None: volume[:] = volumes
    if dry_from is not None: volume[dry_from:] = 0
    ev = a(0); ev[2] = -.0001
    if pre: ev[0] = .001
    if events is not None: ev[:] = events
    ivh = np.where(ev != 0, 8., 0.) if intervals is None else np.asarray(intervals, dtype=float)
    r8 = a(-.001) if rates is None else np.asarray(rates, dtype=float)
    ok = np.zeros((T, N), np.uint8); ok[1] = 1
    if allowed is not None: ok[:] = allowed
    H = end-start
    used, cum = np.zeros(H), np.zeros(H)
    ep, comp, diag = np.zeros((100, 11)), np.zeros((H, 6)), np.zeros(32)
    pos, xd = np.zeros((H, N)), np.zeros(64)
    accounts = np.zeros((H, 9))
    kw = dict(entry_fav_n=.0005, entry_basis_n=-.003, exit_fav_n=-1e9,
              wait_max_n=0, pov=.03, exec_mode=1, impact=0, fee_s=0, fee_f=0,
              capital=1e6, derisk=0, f_exitlag=1, f_markfund=1, f_coin=2,
              f_budget=1, f_prior=1, f_hardcap=1, f_exitpov=1, k_imp=impact,
              fx_fund=2, xmode=5, x_obs=obs, conf_h=1, wind_down_h=1,
              res_mode=3, fill_cap=1)
    delivery = overrides.pop("delivery", 1e9)
    kw.update(overrides)
    v, z = np.ones(N), np.zeros(N)
    # The production JIT path, not only the Python implementation.
    res, _, _ = E.sim(start, end, sv, sv, volume, fv, fv, volume, ev, r8, a(0), a(1e6), a(1e6), a(1e8), a(1e6),
        z, z, v*2, v*5, ok, np.zeros_like(ok), v*1000, z, fv, np.broadcast_to(qty, (N,)).copy(),
        a(10000), ivh, a(.01), a(1), E.mkp(**kw), used, cum, ep, comp, diag, pos,
        sv, sv*1.02, sv*.98, fv, fv*1.02, fv*.98, z, z, v*1e9,
        v*delivery,
        xd, np.cumsum(ev, axis=0), a(1), np.ones((3, T, N)), v, entry_rates, accounts, account_transfers)
    return dict(ep=ep[:int(res[:, 6].sum())], res=res, diag=diag, pos=pos,
                cum=cum, used=used, xd=xd, comp=comp, accounts=accounts)


class LedgerTests(unittest.TestCase):
    def test_hedged_profit_cannot_fund_spot_repurchase_across_accounts(self):
        r = replay(price_increase_from=3, fee_s=.0006, fee_f=.00032)
        a = r['accounts']
        # Equal absolute moves cancel direction PnL, but A must find the extra
        # spot-repurchase cash while B retains the offsetting perpetual gain.
        self.assertEqual(r['diag'][24], 0)
        self.assertLess(a[-1, 0], -500)
        self.assertGreater(a[-1, 1], 1e6 + 499)
        self.assertAlmostEqual(a[-1, 5], a[-1, 0])
        joint = a[:, 0] - a[:, 4] + a[:, 1] + a[:, 2] - 1e6
        np.testing.assert_allclose(joint, r['cum'], atol=1e-8)
        funded = replay(price_increase_from=3, fee_s=.0006, fee_f=.00032, account_transfers=True)
        f = funded['accounts']
        self.assertAlmostEqual(f[-1, 0], 0)
        self.assertGreater(f[-1, 7], 500)
        np.testing.assert_allclose(f[:, 0]-f[:, 4]+f[:, 1]+f[:, 2]-1e6, funded['cum'], atol=1e-8)
        np.testing.assert_allclose(funded['cum'], r['cum'], atol=1e-8)

    def test_entry_threshold_is_strict_and_uses_forecast(self):
        # Settled rates cannot replace the independently archived entry forecast.
        settled = np.full((16, 1), .001)
        for forecast, enters in [(-.000999, False), (-.001, False), (-.001000001, True)]:
            with self.subTest(forecast=forecast):
                r = replay(rates=settled, entry_rates=np.full((16, 1), forecast), entry_fav_n=.001)
                self.assertEqual(r['pos'].max() > 0, enters)
        r = replay(rates=-settled, entry_rates=settled, entry_fav_n=.001)
        self.assertEqual(r['pos'].max(), 0)

    def test_entries_and_adds_only_at_recheck_hours(self):
        allowed = np.zeros((16, 1), dtype=np.uint8); allowed[[1, 9]] = 1
        # Small hourly participation permits two distinct fills, rather than one full fill.
        r = replay(qty=10, volumes=np.full((16, 1), 1000.), allowed=allowed,
                   entry_rates=np.full((16, 1), -.002), entry_fav_n=.001,
                   events=np.zeros((16, 1)), add_n=1)
        increases = np.flatnonzero(np.diff(np.r_[0., r['pos'][:, 0]]) > 1e-8)
        np.testing.assert_array_equal(increases, [1, 9])
        decreases = np.flatnonzero(np.diff(np.r_[0., r['pos'][:, 0]]) < -1e-8)
        np.testing.assert_array_equal(decreases, [11])  # unwind proceeds outside recheck hours

    def test_zero_inventory_cannot_trade(self):
        r = replay(qty=0)
        self.assertEqual(r['pos'].max(), 0)
        self.assertEqual(r['res'][:, 5].sum(), 0)

    def test_raw_token_cap(self):
        r = replay(qty=2.5)
        self.assertAlmostEqual(r['pos'].max()/100, 2.5)

    def test_prior_volume_does_not_allow_fills_in_dry_hour(self):
        r = replay(dry_from=1)
        self.assertEqual(r['pos'].max(), 0)

    def test_current_volume_cannot_raise_submission_budget(self):
        vols = np.full((16,1), 1e6); vols[0] = 100
        r = replay(volumes=vols)
        self.assertLessEqual(r['pos'][1,0], 3 + 1e-8)

    def test_dry_terminal_is_open_liability(self):
        r = replay(dry_from=2)
        self.assertEqual(len(r['ep']), 0)
        self.assertAlmostEqual(r['diag'][8], 1000)
        self.assertEqual(r['diag'][24], 1)
        self.assertEqual(r['diag'][9], 0)  # no invented close costs
        self.assertEqual(r['pos'][-1,0], 1000)
        self.assertEqual(r['res'][:,5].sum(), 1)
        self.assertAlmostEqual(r['cum'][-1], r['comp'][-1,0] + r['comp'][-1,1] - r['comp'][-1,2] + r['comp'][-1,5])

    def test_unwind_tail_charges_actual_funding_then_closes(self):
        vols = np.full((16,1),1e6); vols[2:12] = 0
        ev = np.zeros((16,1)); ev[12] = .001
        r = replay(volumes=vols, events=ev, trade_end=12, end=16)
        self.assertEqual(r['diag'][24], 0)
        self.assertEqual(r['ep'][0,2], 13)  # prior-hour budget recovers one bar later
        self.assertAlmostEqual(r['res'][:,1].sum(), -10*99.5*.001)
        self.assertEqual(r['pos'][-1,0], 0)

    def test_pre_entry_funding_is_not_held_funding(self):
        r = replay(pre=True)
        self.assertEqual(r['ep'][0,2], 11)
        self.assertGreater(r['res'][:,1].sum(), 0)

    def test_own_impact_can_trigger_net_payment_exit(self):
        r = replay(impact=4)
        self.assertEqual(r['ep'][0,2], 2)
        self.assertLess(r['res'][:,1].sum(), 0)

    def test_zero_rate_settlement_still_charges_own_impact(self):
        events = np.zeros((16,1)); intervals = np.zeros((16,1)); intervals[2] = 8
        no_event = replay(events=events, impact=4)
        settled = replay(events=events, intervals=intervals, impact=4)
        self.assertEqual(no_event['res'][:,1].sum(), 0)
        self.assertEqual(settled['ep'][0,2], 2)
        self.assertAlmostEqual(settled['res'][:,1].sum(), -10*99.5*(4e-4*(10*99.5/10000)*100))

    def test_delivery_cannot_fabricate_spot_repurchase(self):
        with self.assertRaisesRegex(ValueError, "delivery"):
            replay(dry_from=2, delivery=5)

    def test_shared_tiered_capital(self):
        rates = np.tile([-.001, -.01], (16,1))
        r = replay(rates=rates, capital=1000, tiered_exec=1, mkm=1, fshare=1, advw=0)
        self.assertLessEqual(r['pos'].sum(axis=1).max(), 1000 + 1e-8)
        self.assertLessEqual(r['used'].max(), 1000 + 1e-8)

    def test_tiered_deep_uses_taker_shallow_uses_maker(self):
        shallow = replay(tiered_exec=1, mkm=1, fshare=1, advw=0)
        deep = replay(rates=np.full((16,1),-.01), tiered_exec=1, mkm=1, fshare=1, advw=0)
        self.assertGreater(shallow['xd'][30], 0)
        self.assertEqual(deep['xd'][30], 0)

    def test_episode_execution_policy_stays_at_entry_tier(self):
        rates = np.full((16,1), -.01); rates[2:] = -.001
        r = replay(rates=rates, tiered_exec=1, mkm=1, fshare=1, advw=0)
        self.assertEqual(r['xd'][30], 0)


class InputTests(unittest.TestCase):
    def test_zero_rate_event_resets_funding_streak(self):
        data = object.__new__(Data)
        data.arr = [None]*8
        data.arr[6] = np.array([[-.001], [0.], [-.001]])
        data.arr[7] = np.array([[-.002], [0.], [-.002]])
        data._event_iv = np.full((3,1),8.)
        np.testing.assert_array_equal(data.ivh(), data._event_iv)
        np.testing.assert_array_equal(data.streak_arr(), [[1.], [0.], [1.]])
    def test_inventory_30percent_is_not_discounted_again(self):
        with tempfile.TemporaryDirectory() as td:
            src, dst = Path(td)/'in.tsv', Path(td)/'caps.csv'
            src.write_text('Token\tBorrowable Qty @ 30% (Token)\tBinance Has Contract\tBinance Symbols\tBN Multiplier (Token/Base)\tBN Borrowable Qty (Contract Base)\nPEPE\t123000\tY\t1000PEPEUSDT\t1000\t123\nZERO\t0\tY\tZEROUSDT\t1\t0\n')
            report = convert_inventory(src, dst)
            cap = load_caps(dst)
            self.assertEqual(cap.cap_qty_token.tolist(), [123000,0])
            self.assertEqual(cap.cap_usdt.tolist(), [10_000_000]*2)
            self.assertEqual(report['annualization_denominator_usdt'], 10_000_000)
            self.assertEqual(dst.stat().st_mode & 0o777, 0o600)

    def test_duplicate_or_invalid_caps_fail(self):
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td)/'caps.csv'
            for rows in ['AAAUSDT,10,1\n AAAUSDT ,10,1', 'AAAUSDT,10,-1', 'AAAUSDT,10,inf']:
                dst.write_text('fut_symbol,cap_usdt,cap_qty_token\n'+rows+'\n')
                with self.assertRaises(ValueError): load_caps(dst)

    def test_sigma_is_prefix_invariant(self):
        def sigma(prices):
            data = object.__new__(Data); data.arr = [None, prices]
            return data.sigma()
        p = np.exp(np.arange(80)[:,None]*.01 + np.sin(np.arange(80)[:,None])*.03)
        p = np.hstack([p, np.full_like(p, np.nan)])
        np.testing.assert_allclose(sigma(p[:40]), sigma(p)[:40])
        self.assertEqual(sigma(p)[0,0], .05)

    def test_calibration_does_not_backfill_future(self):
        # Isolate pure function from legacy import-time full-data construction.
        tree = ast.parse((ROOT/'repro/lib/r7_lib.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='stepwise')
        ns = dict(np=np, T=16, N=2)
        exec(compile(ast.Module(body=[fn],type_ignores=[]), '<stepwise>', 'exec'), ns)
        f = ns['stepwise']
        initial = f([], lo=.1, hi=3.)
        later = f([(0,8,.2)], lo=.1, hi=3.)
        np.testing.assert_array_equal(initial[:8], later[:8])
        self.assertTrue((later[:8] == 3).all())
        self.assertLess(later[8,0], 3)
        # Samples available before the dataset begin are already known at hour zero.
        self.assertAlmostEqual(f([(0,-24,.4)], lo=.1, hi=3.)[0,0],.4)


if __name__ == '__main__': unittest.main()
