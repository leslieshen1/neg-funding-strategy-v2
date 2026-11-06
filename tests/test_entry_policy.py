"""Point-in-time snapshots, never reconstructed from subsequent settlements."""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'repro/lib'))
from entry_policy import load_entry_policy

HOUR = 3_600_000
START = 1759276800000  # 2025-10-01 00:00 UTC


def snapshots():
    decision = START + np.arange(16) * HOUR
    shape = (16, 1)
    return dict(symbols=np.array(['TESTUSDT']), decision_time_ms=decision,
                observed_at_ms=decision[:, None].astype(float),
                next_settlement_at_ms=np.broadcast_to(START + ((np.arange(16)//8+1)*8)[:, None]*HOUR, shape).copy(),
                expected_funding_rate=np.full(shape, -.0002),
                funding_interval_hours=np.full(shape, 1.))


class EntryPolicyTests(unittest.TestCase):
    def load(self, arrays, source='archived'):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'entry.npz'
            np.savez(path, **arrays)
            return load_entry_policy(path, (16, 1), ['TESTUSDT'], source)

    def test_reconstruction_requires_explicit_opt_in_and_is_labelled(self):
        arrays = snapshots(); arrays['signal_source'] = np.array('reconstructed')
        arrays['reconstruction_method'] = np.array('closed premium minute scenario')
        with self.assertRaisesRegex(ValueError, 'explicitly select'):
            self.load(arrays)
        p = self.load(arrays, 'reconstructed')
        self.assertTrue(p.metadata['reconstructed_forecasts'])
        self.assertEqual(p.metadata['reconstruction_method'], 'closed premium minute scenario')
        with self.assertRaisesRegex(ValueError, 'explicitly select'):
            self.load(snapshots(), 'reconstructed')

    def test_normalises_interval_and_uses_announced_next_time(self):
        arrays = snapshots()
        # A changed schedule announced at hour 3 is allowed; no look-ahead at ev.
        arrays['next_settlement_at_ms'][3, 0] = START + 4*HOUR
        p = self.load(arrays)
        np.testing.assert_array_equal(np.flatnonzero(p.mask[:, 0]), [3, 7, 15])
        self.assertAlmostEqual(p.r8[3, 0], -.0016)
        self.assertTrue(np.isnan(p.r8[4, 0]))

    def test_stale_and_missing_quotes_do_not_open(self):
        arrays = snapshots()
        arrays['observed_at_ms'][7, 0] -= HOUR
        arrays['expected_funding_rate'][15, 0] = np.nan
        p = self.load(arrays)
        self.assertFalse(p.mask.any())

    def test_recent_asynchronous_quote_and_freshness_boundary(self):
        arrays = snapshots()
        arrays['observed_at_ms'][7, 0] -= 1_068.25
        arrays['observed_at_ms'][15, 0] -= 10_000
        p = self.load(arrays)
        self.assertTrue(p.mask[7, 0])
        self.assertTrue(p.mask[15, 0])
        arrays['observed_at_ms'][15, 0] -= 1
        self.assertFalse(self.load(arrays).mask[15, 0])

    def test_later_snapshots_cannot_change_earlier_decisions(self):
        arrays = snapshots(); before = self.load(arrays)
        arrays['expected_funding_rate'][8:] = -.5
        arrays['next_settlement_at_ms'][8:] += HOUR
        after = self.load(arrays)
        np.testing.assert_array_equal(before.r8[:8], after.r8[:8])
        np.testing.assert_array_equal(before.mask[:8], after.mask[:8])

    def test_future_observation_is_rejected(self):
        arrays = snapshots(); arrays['observed_at_ms'][7, 0] += 1
        with self.assertRaisesRegex(ValueError, 'after decision'):
            self.load(arrays)

    def test_wrong_symbol_order_is_rejected(self):
        arrays = snapshots(); arrays['symbols'][0] = 'WRONG'
        with self.assertRaisesRegex(ValueError, 'symbol order'):
            self.load(arrays)

    def test_wrong_time_grid_is_rejected(self):
        arrays = snapshots(); arrays['decision_time_ms'] += HOUR
        with self.assertRaisesRegex(ValueError, 'time grid'):
            self.load(arrays)

    def test_missing_archive_is_not_replaced_with_settled_rates(self):
        with self.assertRaisesRegex(FileNotFoundError, 'pre-settlement snapshots'):
            load_entry_policy('/nonexistent/entry.npz', (16, 1), ['TESTUSDT'])


if __name__ == '__main__': unittest.main()
