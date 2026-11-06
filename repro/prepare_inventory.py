"""Convert the private inventory export without printing any quantities."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
from inventory import convert_inventory

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source")
    p.add_argument("destination")
    p.add_argument("--capital", type=float, default=10_000_000)
    a = p.parse_args()
    print(json.dumps(convert_inventory(a.source, a.destination, a.capital), ensure_ascii=False, indent=2))
