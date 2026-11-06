"""The user's settled-rate strategy, independent of forecast reconstruction."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'repro/lib'))
import entry_policy
from test_two_accounts import replay


class SettledStrategyTests(unittest.TestCase):
    def test_causal_normalisation_and_next_settlement_recheck(self):
        ev = np.zeros((17, 3)); iv = np.zeros_like(ev)
        for n, hours in enumerate((1, 4, 8)):
            iv[::hours, n] = hours
            ev[::hours, n] = -.002 * hours / 8
        p = entry_policy.settled_entry_policy(ev, iv)
        # For a 1h contract the first T-1h instant coincides with the signal;
        # wait until the following cycle instead of entering immediately.
        np.testing.assert_array_equal(np.flatnonzero(p.mask[:, 0]), np.arange(1, 17))
        np.testing.assert_array_equal(np.flatnonzero(p.mask[:, 1]), [3, 7, 11, 15])
        np.testing.assert_array_equal(np.flatnonzero(p.mask[:, 2]), [7, 15])
        np.testing.assert_allclose(p.r8, -.002)
        ev[8:] = np.where(iv[8:] > 0, .01, 0.)
        later = entry_policy.settled_entry_policy(ev, iv)
        np.testing.assert_array_equal(p.mask[:8], later.mask[:8])
        np.testing.assert_array_equal(p.r8[:8], later.r8[:8])

    def test_zero_settlement_resets_signal_and_threshold_is_strict(self):
        ev = np.full((16, 1), -.001/8); iv = np.ones_like(ev)
        p = entry_policy.settled_entry_policy(ev, iv)
        self.assertFalse(p.mask.any())
        ev[:] = -.002/8; ev[5] = 0
        p = entry_policy.settled_entry_policy(ev, iv)
        self.assertFalse(p.mask[5, 0])
        self.assertFalse(p.mask[6, 0])  # a new signal arms, no immediate entry
        self.assertTrue(p.mask[7, 0])

    def test_raw_exit_includes_pre_entry_settlements(self):
        r = replay(pre=True, obs=1.)
        self.assertEqual(r['ep'][0, 2], 2)
        # Pre-entry settlement affects the exit signal, never the cash ledger.
        self.assertAlmostEqual(r['res'][:, 1].sum(), 10*99.5*.0001)

    def test_own_impact_does_not_change_raw_exit_signal(self):
        r = replay(impact=4., obs=1.)
        self.assertLess(r['res'][:, 1].sum(), 0)
        self.assertEqual(r['ep'][0, 2], 11)  # only terminal unwind, raw sum < 0

    def test_missing_bar_still_locks_exit_until_flat(self):
        ev = np.zeros((16, 1)); ev[2] = .001; ev[3] = -.002
        vols = np.full_like(ev, 1e6); vols[2] = 0
        r = replay(events=ev, volumes=vols, obs=1., missing_price_hours=(2,))
        # Trigger at 2 even though it cannot fill; the reversal at 3 cannot cancel.
        self.assertEqual(r['ep'][0, 2], 4)

    def test_supplied_settled_signal_is_not_rewritten_by_own_impact(self):
        allowed = np.zeros((16, 1), np.uint8); allowed[[1, 2]] = 1
        r = replay(allowed=allowed, volumes=np.full((16, 1), 1000.),
                   entry_rates=np.full((16, 1), -.00101), impact=4., obs=1.,
                   entry_fav_n=.001, events=np.zeros((16, 1)), add_n=1)
        self.assertGreater(r['pos'][2, 0], r['pos'][1, 0])


if __name__ == '__main__':
    unittest.main()
