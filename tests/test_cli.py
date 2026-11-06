"""Run the actual headline command on explicitly synthetic market inputs."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def fixture(root, dry_end=False):
    (root/'ds').mkdir(); (root/'calib').mkdir()
    T,N = 9576,3
    ts = np.arange(T)[:,None]
    sc = np.broadcast_to(100+np.sin(ts/12)*.2,(T,N)).copy(); fc = sc*.995
    qty = np.full((T,N),1e8)
    # Zero volume through the FY end and all available unwind bars.
    if dry_end: qty[-400:] = 0
    r8 = np.broadcast_to(np.array([-.002,-.01,-.01]),(T,N)).copy()
    ev = np.where(ts%8 == 0,r8,0)
    z = dict(sv=sc,sc=sc,sq=qty,fv=fc,fc=fc,fq=qty,ev=ev,r8=r8,r8s=r8,
             d02=qty,d1=qty*5,oi=np.full_like(sc,1e9),oih=np.full_like(sc,1e9),
             mk=fc,ivh=np.broadcast_to(np.where(ts%8 == 0,8.,0.),(T,N)),
             so=sc,sh=sc*1.003,sl=sc*.997,fo=fc,fh=fc*1.003,fl=fc*.997)
    np.savez(root/'ds/dataset.npz',**z)
    symbols = ['FAKEUSDT','1000TESTUSDT','ZEROUSDT']
    # Explicitly synthetic point-in-time quotes, independent of realised ev.
    hour_ms = 3_600_000
    start_ms = pd.Timestamp('2025-10-01', tz='UTC').value//1_000_000
    decision = start_ms + np.arange(T)*hour_ms
    next_settle = start_ms + ((np.arange(T)//8+1)*8)*hour_ms
    np.savez(root/'ds/entry_snapshots.npz', symbols=np.array(symbols), decision_time_ms=decision,
             observed_at_ms=np.broadcast_to(decision[:, None], (T,N)),
             next_settlement_at_ms=np.broadcast_to(next_settle[:, None], (T,N)),
             expected_funding_rate=r8, funding_interval_hours=np.full((T,N),8.))
    meta = pd.DataFrame(dict(fut=symbols,spot=['FAKEUSDT','TESTUSDT','ZEROUSDT'],mult=[1,1000,1],
                             hs_s=[.00005]*N,hs_f=[.00005]*N,r2=[2.]*N,r5=[5.]*N,
                             tick_s=[.00001]*N,tick_f=[.00001]*N,dl_exit=[1e9]*N,dl_fin=[1e9]*N,
                             delisted=[False]*N,grp=['pool']*N))
    meta.to_csv(root/'ds/dataset_meta.csv',index=False)
    ag = pd.DataFrame([dict(sym=sym,date=date,y_kyle=1.,hs=.00005) for sym in symbols for date in ['2025-10-01','2025-11-01']])
    ag.to_csv(root/'calib/ag.csv',index=False)
    pd.DataFrame(columns=['sym','date']).to_csv(root/'calib/bd.csv',index=False)
    caps = root/'caps.csv'
    pd.DataFrame(dict(fut_symbol=symbols,cap_usdt=[1e7]*N,cap_qty_token=[1e6,1e6,0])).to_csv(caps,index=False)
    return caps


class HeadlineTests(unittest.TestCase):
    def test_missing_settlement_mark_is_rejected(self):
        sys.path.insert(0,str(ROOT/'repro/lib'))
        from inputs import validate_inputs
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); caps=fixture(root)
            path=root/'ds/dataset.npz'
            with np.load(path) as archive:
                arrays={k:archive[k] for k in archive.files}
            arrays['mk'][8,0]=np.nan
            np.savez(path,**arrays)
            with self.assertRaisesRegex(ValueError,'settlement marks missing'):
                validate_inputs(root,caps)

    def test_contract_prices_must_use_raw_token_units(self):
        sys.path.insert(0,str(ROOT/'repro/lib'))
        from inputs import validate_inputs
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); caps=fixture(root)
            path=root/'ds/dataset.npz'
            with np.load(path) as archive:
                arrays={k:archive[k] for k in archive.files}
            arrays['fc'][:,1]*=1000
            np.savez(path,**arrays)
            with self.assertRaisesRegex(ValueError,'normalised to raw-token'):
                validate_inputs(root,caps)

    def run_cli(self, dry=False, mode='taker', coverage='continuous'):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); caps=fixture(root,dry)
            # The actual strategy must run without any forecast archive.
            (root/'ds/entry_snapshots.npz').unlink()
            output=root/'result.json'
            env=dict(os.environ,NFB_DATA=str(root),NFB_CAP_FILE=str(caps),NFB_ROOT=str(root),NFB_S='999',NFB_DERISK='1')
            result=subprocess.run([sys.executable,str(ROOT/'repro/run_headline.py'),'--windows','FY','--exec',mode,'--signal-coverage',coverage,'--json',str(output)],env=env,text=True,capture_output=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            report=json.loads(output.read_text())
            return report

    def test_fixed_denominator_and_full_unwind(self):
        report=self.run_cli(mode='tiered_perp'); r=report['FY']
        self.assertEqual(r['capital_usdt'],10_000_000)
        self.assertEqual(r['annualization_denominator_usdt'],10_000_000)
        self.assertEqual(report['account_model']['borrow_interest'],0)
        self.assertFalse(report['account_model']['inventory_value_added_to_denominator'])
        self.assertEqual(report['account_model']['leverage'],1)
        self.assertEqual(report['account_model']['initial_notional_budget_usdt'],10_000_000)
        self.assertEqual(report['fee_model']['tier'],'vip3')
        self.assertEqual(report['engine_params']['fee_s'],.0006)
        self.assertEqual(report['engine_params']['fee_f'],.00032)
        self.assertAlmostEqual(report['fee_model']['expected_income_roundtrip_fee'],.00184)
        self.assertFalse(report['execution_model']['historical_l2_required'])
        self.assertEqual(report['engine_params']['res_mode'],3)
        self.assertEqual(report['engine_params']['f_prior'],1)
        self.assertEqual(report['engine_params']['x_obs'],1)
        self.assertEqual(report['engine_params']['conf_h'],1)
        self.assertEqual(report['engine_params']['derisk'],0)
        self.assertEqual(report['engine_params']['fx_eq'],0)
        self.assertEqual(report['engine_params']['derisk_force'],2)
        self.assertEqual(report['engine_params']['entry_fav_n'],.001)
        self.assertEqual(report['entry_policy']['comparator'],'<')
        self.assertEqual(report['entry_policy']['recheck_before_hours'],1)
        self.assertEqual(report['entry_policy']['applies_to'],['entry'])
        self.assertEqual(report['entry_policy']['add_timing'],'qualified')
        self.assertEqual(r['entry_fill_audit']['first_fills_outside_recheck'],0)
        self.assertGreater(r['entry_fill_audit']['adds_outside_recheck'],0)
        self.assertEqual(report['entry_policy']['signal_source'],'settled')
        self.assertFalse(report['entry_policy']['extra_adl_filter'])
        self.assertTrue(report['exit_policy']['includes_pre_entry_settlements'])
        self.assertFalse(report['account_model']['separate_account_cash_paths_verified'])
        self.assertTrue(report['account_model']['mutual_transfers_allowed'])
        self.assertTrue(report['account_model']['hourly_cash_paths_audited'])
        self.assertLess(r['account_cash_audit']['cash_pnl_identity_max_error_usdt'], .001)
        self.assertEqual(r['signal_hours'],8760)
        self.assertEqual(r['tail_hours'],168)
        self.assertEqual(r['status'],'fully_unwound')
        self.assertGreater(r['n'],0)
        self.assertAlmostEqual(r['ann'], r['net_wan']*1e4/1e7*8760/r['H']*100)
        self.assertAlmostEqual(r['net_wan'],r['closed_episode_net_wan'])
        # Fee/spread/impact have already changed execution prices, never subtract twice.
        self.assertAlmostEqual(r['net_wan'],r['fund_wan']+r['price_wan']-r['fees_wan'])

    def test_unclosed_position_cannot_report_completed_return(self):
        r=self.run_cli(dry=True)['FY']
        self.assertEqual(r['status'],'open_inventory_liability')
        self.assertIsNone(r['ann'])
        self.assertGreater(r['open_symbols'],0)
        self.assertGreater(r['residual_spot_notional_usdt'],0)
        self.assertAlmostEqual(r['net_wan'],r['closed_episode_net_wan']+r['open_episode_net_mtm_wan'])
        self.assertAlmostEqual(r['net_wan'],r['fund_wan']+r['price_wan']-r['fees_wan']+r['unrealized_pair_price_wan'])

    def test_sampled_quotes_cannot_report_an_annual_return(self):
        report=self.run_cli(coverage='sampled'); r=report['FY']
        self.assertEqual(report['signal_coverage'],'sampled')
        self.assertTrue(r['partial_observation_pnl_only'])
        self.assertIsNone(r['ann'])
        self.assertIsNone(r['ann_mtm'])
        self.assertGreater(r['n'],0)
        self.assertAlmostEqual(r['period_return_on_10m_pct'],r['net_wan']*1e4/1e7*100)

    def test_missing_inputs_fail_before_backtest(self):
        with tempfile.TemporaryDirectory() as td:
            env=dict(os.environ,NFB_DATA=td,NFB_CAP_FILE=str(Path(td)/'absent.csv'))
            result=subprocess.run([sys.executable,str(ROOT/'repro/run_headline.py')],env=env,text=True,capture_output=True,timeout=15)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Missing required backtest inputs', result.stderr)
            self.assertNotIn('Traceback',result.stderr)

    def test_missing_forecasts_fail_before_backtest(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); caps=fixture(root)
            (root/'ds/entry_snapshots.npz').unlink()
            env=dict(os.environ,NFB_DATA=str(root),NFB_CAP_FILE=str(caps),NFB_ROOT=str(root))
            output=root/'result.json'
            result=subprocess.run([sys.executable,str(ROOT/'repro/run_headline.py'),'--signal-source','archived','--json',str(output)],env=env,text=True,capture_output=True,timeout=15)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Missing pre-settlement snapshots',result.stderr)
            self.assertNotIn('Traceback',result.stderr)
            self.assertFalse(output.exists())


if __name__ == '__main__': unittest.main()
