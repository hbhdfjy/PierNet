#!/usr/bin/env python3
"""Benchmark heavyweight PiERN API read endpoints against the local backend."""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE_URL = 'http://127.0.0.1:8000'
OUTPUT_DIR = Path('/home/tpx/piern/.runlogs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def request_json(path: str) -> tuple[float, int, object]:
    url = BASE_URL + path
    started = time.perf_counter()
    with urllib.request.urlopen(url, timeout=120) as resp:
        payload = resp.read()
        elapsed_ms = (time.perf_counter() - started) * 1000
        return elapsed_ms, len(payload), json.loads(payload.decode('utf-8'))


def choose_paths() -> dict[str, str]:
    root = Path('/home/tpx/piern/data')
    sample = next((p.stem for p in sorted((root / 'text2comp').glob('*.jsonl')) if p.name != 'all_training_data.jsonl'), '')
    template = next((p.name.replace('_templates.jsonl', '') for p in sorted((root / 'templates').glob('*_templates.jsonl'))), '')
    router = next((p.stem for p in sorted((root / 'router' / 'by_scenario').glob('*.jsonl'))), '')
    return {
        'sample_scenario': sample,
        'template_scenario': template,
        'router_scenario': router,
    }


def timed_runs(name: str, path: str, runs: int = 3) -> dict:
    samples = []
    last_size = 0
    last_meta = {}
    for _ in range(runs):
        elapsed_ms, size_bytes, payload = request_json(path)
        samples.append(round(elapsed_ms, 2))
        last_size = size_bytes
        if isinstance(payload, dict):
            last_meta = {
                'keys': sorted(payload.keys())[:20],
                'total': payload.get('total', payload.get('total_samples')),
                'page_size': payload.get('page_size'),
            }
        time.sleep(0.2)
    ordered = sorted(samples)
    return {
        'name': name,
        'path': path,
        'runs_ms': samples,
        'p50_ms': ordered[len(ordered) // 2],
        'p95_ms': ordered[-1],
        'response_bytes': last_size,
        'payload_meta': last_meta,
    }


def main() -> None:
    picks = choose_paths()
    endpoints = [
        ('datasets', '/api/datasets'),
        ('stats', '/api/stats'),
        ('files.templates', '/api/files/templates'),
        ('router.status', '/api/router/status'),
    ]
    if picks['sample_scenario']:
        q = urllib.parse.urlencode({'scenario': picks['sample_scenario'], 'page': 0, 'page_size': 10})
        endpoints.append(('samples.page0', f'/api/samples?{q}'))
    if picks['template_scenario']:
        q = urllib.parse.urlencode({'page': 0, 'page_size': 10})
        endpoints.append(('template_items.page0', f"/api/files/templates/{urllib.parse.quote(picks['template_scenario'])}/items?{q}"))
    if picks['router_scenario']:
        q = urllib.parse.urlencode({'split': 'train', 'scenario': picks['router_scenario'], 'page': 0, 'page_size': 10, 'label': -1})
        endpoints.append(('router_samples.page0', f'/api/router/samples?{q}'))

    results = {
        'generated_at': time.time(),
        'base_url': BASE_URL,
        'selected_scenarios': picks,
        'endpoints': [timed_runs(name, path) for name, path in endpoints],
    }

    stamp = time.strftime('%Y%m%d-%H%M%S')
    out_path = OUTPUT_DIR / f'performance-baseline-{stamp}.json'
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(out_path)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
