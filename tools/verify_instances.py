"""Verify the official AssemblyGrid catalogue against the runtime implementation."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Experiments' / 'env'))
from official import load_official_config
from standards import official_instance_suite

path = ROOT / 'official' / 'official_realizations.json'
catalogue = json.loads(path.read_text(encoding='utf-8'))
records = catalogue['realizations']
assert catalogue['schema'] == 'assemblygrid-official-realizations-v4'
assert len(records) == 45
assert len({r['instance_id'] for r in records}) == 45
assert {r['family'] for r in records} == {'flow', 'coalition', 'concurrency'}
assert len(official_instance_suite(0)) == 9
for record in records:
    load_official_config(path, record['instance_id'])
print(f'PASS: {len(records)} official realizations verified.')
