"""Two-account negative-funding replay; annualisation denominator fixed at 10M USDT."""
import os, sys, json, argparse, math
ap = argparse.ArgumentParser()
ap.add_argument("--fee", default="vip3", choices=["vip0", "vip3"])
ap.add_argument("--exec", default="taker", choices=["taker", "perp_maker", "spot_maker", "tiered_perp", "tiered_spot"])
ap.add_argument("--fill", type=float, default=0.5); ap.add_argument("--adv", type=float, default=0.5)
ap.add_argument("--entry-tiers", default="T1,T2,T3,T4"); ap.add_argument("--json", default=None)
ap.add_argument("--windows", default="IS,HO,FY")
ap.add_argument("--tail-hours", type=int, default=168, help="Use available future bars for actual unwind; never synthesise missing bars")
ap.add_argument("--entry-snapshots", default=None, help="Only for explicitly selected forecast research variants")
ap.add_argument("--signal-source", default="settled", choices=["settled", "archived", "reconstructed"],
                help="Default: latest already-settled rate; forecast variants change the strategy")
ap.add_argument("--signal-coverage", default="continuous", choices=["continuous", "sampled"],
                help="Sampled signals produce partial-observation PnL only; annualisation is disabled")
ap.add_argument("--adv-mode", default="range", choices=["range", "nextclose", "basis"]); ap.add_argument("--hedge", default="taker", choices=["taker", "worst"]); ap.add_argument("--gap", default="off", choices=["off", "on"])
A = ap.parse_args()
for k in ("NFB_DATA", "NFB_CAP_FILE"):
    if not os.environ.get(k): sys.exit(f"请先设置环境变量 {k}（见 README）")
if not math.isfinite(A.fill) or not math.isfinite(A.adv) or not 0 <= A.fill <= 1 or A.adv < 0 or A.tail_hours < 0:
    ap.error("fill must be in [0,1]; adv and tail-hours must be nonnegative")
if not set(A.windows.split(",")).issubset({"IS", "HO", "FY"}): ap.error("windows must be IS,HO,FY")
if not set(A.entry_tiers.split(",")).issubset({"T1", "T2", "T3", "T4"}): ap.error("unknown entry tier")
# Pin both the engine capital and the report denominator before importing research modules.
os.environ["NFB_S"] = "10000000"
os.environ["NFB_DERISK"] = "2"  # every trim obeys the shared participation budget
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
from inputs import validate_inputs
from entry_policy import load_entry_policy, settled_entry_policy, ENTRY_THRESHOLD_R8
try:
    INPUTS = validate_inputs(os.environ["NFB_DATA"], os.environ["NFB_CAP_FILE"])
    import pandas as pd
    symbols = pd.read_csv(os.path.join(os.environ['NFB_DATA'], 'ds/dataset_meta.csv')).fut.tolist()
    if A.signal_source == 'settled':
        if A.entry_snapshots: raise ValueError('--entry-snapshots requires an explicit forecast --signal-source')
        import numpy as np
        with np.load(os.path.join(os.environ['NFB_DATA'], 'ds/dataset.npz')) as z:
            if 'ivh' not in z.files: raise ValueError('settled strategy requires explicit ivh event intervals, including zero-rate settlements')
            ENTRY_POLICY = settled_entry_policy(z['ev'], z['ivh'])
    else:
        ENTRY_POLICY = load_entry_policy(A.entry_snapshots or os.path.join(os.environ['NFB_DATA'], 'ds/entry_snapshots.npz'),
                                         (INPUTS['dataset_hours'], INPUTS['dataset_symbols']), symbols, A.signal_source)
except (ValueError, FileNotFoundError) as exc:
    sys.exit(str(exc))
ROOT = os.environ.get("NFB_ROOT", ".").rstrip("/")
os.environ["NFB_OUT"] = ROOT + "/out/"; os.environ["NFB_DAILY"] = ROOT + "/out/daily/"
os.makedirs(os.environ["NFB_DAILY"], exist_ok=True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
import engine_v12s, lib12s           # 带逐笔记账字段的引擎副本，必须先于其它模块导入
lib12s.ENTRY_POLICY = ENTRY_POLICY
FEES = {"vip0": dict(fee_s=0.001, fee_s_mk=0.001, fee_f=0.0005, fee_f_mk=0.0002),
        "vip3": dict(fee_s=0.0006, fee_s_mk=0.0004, fee_f=0.00032, fee_f_mk=0.00012)}
# Import-time EV/exit tables also need the selected fee tier.
lib12s.MKO.update(FEES[A.fee])
sys.argv = sys.argv[:1]
import numpy as np, v13lib
from v13lib import run_fix
from run_signals import r8n

TIERS = [("T1", -0.001, 9.0), ("T2", -0.005, -0.001), ("T3", -0.01, -0.005), ("T4", -9.0, -0.01)]   # (lo, hi]
def tier_of(x):
    out = np.empty(x.shape, object)
    for nm, lo, hi in TIERS: out[(x > lo) & (x <= hi)] = nm
    return out
MKM = {"perp_maker": 1.0, "spot_maker": 3.0}
S = 1e7

def cash_audit(r):
    a = r['accounts']
    if a is None: raise ValueError('headline requires separate account cash records')
    joint_pnl = a[:, 0] - a[:, 4] + a[:, 1] + a[:, 2] - S
    err = float(np.max(np.abs(joint_pnl-r['cum'])))
    if err > .001: raise ValueError('two-account cash/PnL identity failed')
    b_equity = a[:, 1] + a[:, 2]
    return dict(initial_a_usdt=0., initial_b_usdt=S,
                ending_a_cash_usdt=float(a[-1, 0]), ending_b_wallet_usdt=float(a[-1, 1]),
                ending_b_unrealized_usdt=float(a[-1, 2]), ending_a_token_liability_mtm_usdt=float(a[-1, 4]),
                cumulative_b_to_a_usdt=float(a[-1, 7]), cumulative_a_to_b_usdt=float(a[-1, 8]),
                min_a_cash_usdt=float(a[-1, 5]), min_b_wallet_usdt=float(a[-1, 6]),
                unfunded_cash_deficit_usdt=float(max(0., -a[-1, 5], -a[-1, 6])),
                min_hourly_b_equity_usdt=float(min(S, b_equity.min())),
                min_hourly_b_equity_minus_1x_notional_usdt=float(min(S, (b_equity-a[:, 3]).min())),
                cash_pnl_identity_max_error_usdt=err,
                transfers='as needed; immediate and zero-fee model; no external cash injection',
                withdrawal_availability_verified=False,
                liquidation_path_verified=False,
                equity_valuation='hour-end perpetual close proxy; no intrahour margin replay')

def run(mode, win, mask):
    mko = dict(FEES[A.fee])
    maker_mode = mode.replace("tiered_", "") + "_maker" if mode.startswith("tiered_") else mode
    if mode != "taker": mko.update(mkm=MKM[maker_mode], pen=0.002, fshare=A.fill, advw=A.adv, advmode={"range": 3.0, "nextclose": 0.0, "basis": 2.0}[A.adv_mode], hx_off=(1.0 if A.hedge == "taker" else 0.0), gap_sym=(2.0 if A.gap == "off" else 0.0))
    v13lib._ns["MKO"].clear(); v13lib._ns["MKO"].update(mko)
    extra = dict(x_obs=1.0, conf_h=1.0, derisk=0.0, f_mtmderisk=0.0, fx_eq=0.0,
                 f_prior=1.0, res_mode=3.0, fill_cap=1.0, entry_fav_n=ENTRY_THRESHOLD_R8,
                 tiered_exec=float(mode.startswith("tiered_")), tail_hours=A.tail_hours)
    r = run_fix("cons_c0", win, mask=mask, extra=extra); ep = r["ep"]
    net = ep[:, 3] - ep[:, 6]; tr = tier_of(r8n[ep[:, 1].astype(int), ep[:, 0].astype(int)]); hold = ep[:, 2] - ep[:, 1]
    tiers = {nm: dict(n=int((tr == nm).sum()), net_wan=float(net[tr == nm].sum() / 1e4), hold_h=float(hold[tr == nm].mean()) if (tr == nm).any() else 0.0) for nm, _, _ in TIERS}
    H = len(r["cum"]); xd = r["xd"]; residual = float(r["diag"][8])
    complete = r["diag"][24] == 0
    net_u = float(r["cum"][-1])
    ann_mtm = net_u / S / (H / 8760) * 100
    closed_net = float(net.sum())
    continuous = A.signal_coverage == 'continuous'
    return dict(ann=ann_mtm if complete and continuous else None, ann_mtm=ann_mtm if continuous else None,
                partial_observation_pnl_only=not continuous, period_return_on_10m_pct=net_u/S*100,
                status="fully_unwound" if complete else "open_inventory_liability",
                capital_usdt=S, annualization_denominator_usdt=S,
                borrow_interest_usdt=0.0, n=len(ep), H=H, signal_hours=r["signal_hours"], tail_hours=r["tail_hours"],
                net_wan=net_u / 1e4, closed_episode_net_wan=closed_net / 1e4,
                open_episode_net_mtm_wan=(net_u-closed_net) / 1e4,
                account_cash_audit=cash_audit(r),
                residual_spot_notional_usdt=residual, open_symbols=int(r["diag"][24]),
                fund_wan=r["funding"] / 1e4, fees_wan=r["fees"] / 1e4, price_wan=r["price"] / 1e4, unrealized_pair_price_wan=float(r["comp"][-1, 5]) / 1e4,
                cost_wan=r["cost"] / 1e4, traded_wan=r["traded"] / 1e4, gap_wan=xd[35] / 1e4, tiers=tiers)

allow = set(A.entry_tiers.split(","))
mask = ENTRY_POLICY.mask.copy()
if allow != {"T1", "T2", "T3", "T4"}:
    tt = tier_of(r8n); mask &= np.isin(tt.astype(str), list(allow))
out = dict(schema_version=5, args=vars(A), inputs=INPUTS, entry_policy=ENTRY_POLICY.metadata,
           exit_policy=dict(signal='raw market settled funding sum over hours t-23..t > 0',
                            includes_pre_entry_settlements=True, normalise_to_8h=False,
                            confirmation_hours=1, locked_until_flat=True,
                            entry_failure_action='stop adds; retain existing hedge',
                            automatic_volume_oi_trims=False,
                            administrative_exceptions=['terminal wind-down'],
                            delivery_handling='fail if still open; no fabricated token repurchase'),
           account_model=dict(inventory="borrowed raw tokens; return same token quantity", borrow_interest=0,
                              inventory_history_assumption="user-provided raw-token quantities available throughout replay",
                              strategy_capital_usdt=S, annualization_denominator_usdt=S,
                              inventory_value_added_to_denominator=False,
                              leverage=1.0, leverage_basis="fixed entry notional budget; internal transfers do not expand it",
                              initial_notional_budget_usdt=S,
                              spot_sale_proceeds_available_to_b=True,
                              mutual_transfers_allowed=True,
                              transfer_assumption="as needed, immediate, zero fee; withdrawal limits unverified",
                              pnl_recycling=False,
                              separate_account_cash_paths_verified=False,
                              hourly_cash_paths_audited=True,
                              margin_path="hourly account cash/equity and transfers; intrahour margin, liquidation and withdrawal limits not replayed"),
           fee_model=dict(tier=A.fee, rates=FEES[A.fee], bnb_discount=False,
                          historical_schedule="fixed user-selected scenario",
                          expected_income_roundtrip_fee=2*__import__('r7_lib').pair_fee()),
           signal_coverage=A.signal_coverage,
           research_status=("settled_signal_modelled_schedule_and_execution" if A.signal_source == 'settled' else
                            "modelled_signal_and_execution" if A.signal_source == 'reconstructed' else "archived_signal_simulated_execution"),
           execution_model=dict(status="simulated", historical_l2_required=False,
                                calibration="volume/depth proxy and declared impact assumptions; not private fills"))
for w in A.windows.split(","):
    # All execution policies run once with shared capital, inventory, and exit states.
    o = run(A.exec, w, mask)
    out[w] = o
    if A.signal_coverage == 'sampled':
        label = "抽样信号重放（不计算年化）"
    else:
        label = f"完成平仓年化 {o['ann']:.2f}%" if o["ann"] is not None else f"含未平仓估值年化 {o['ann_mtm']:.2f}%（未完成平仓）"
        if A.signal_source == 'reconstructed': label = "重建信号模型：" + label
    print(f"{w}: {label}  已平仓笔数 {o['n']}  净值变动 {o['net_wan']:.1f} 万U  剩余敞口 {o['residual_spot_notional_usdt']:.2f} U", flush=True)
if A.json:
    out["engine_params"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in v13lib._ns["LASTP"].items() if not hasattr(v, "shape")}
    with open(A.json, "w") as f:
        json.dump(out, f, indent=2, allow_nan=False)
