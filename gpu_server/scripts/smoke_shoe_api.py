"""Verify real shoe recommendation, generation, image download and cache over HTTP."""
import argparse
import json
import time
from pathlib import Path

from smoke_web_api import _download, _json_request, _multipart


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--base-url', required=True)
    cli.add_argument('--image', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--timeout', type=int, default=900)
    args = cli.parse_args()
    base = args.base_url.rstrip('/')
    args.output.mkdir(parents=True, exist_ok=True)

    def save(name, payload):
        (args.output / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    health = _json_request(base + '/api/health')
    save('health.json', health)
    assert health['device'] == 'cuda' and 'shoes' in health['tryon_categories'], health
    profile = dict(purpose='데일리', gender='남성', desired_style='캐주얼', change_categories=['shoes'],
                   min_budget=30000, max_budget=200000, budget=100000, season='가을', activity_level='보통')
    body, content_type = _multipart(args.image, profile)
    created = _json_request(base + '/api/analyze', method='POST', data=body,
                            headers={'Content-Type': content_type})
    job = created['job_id']
    save('job.json', created)
    print('job', job, flush=True)
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        state = _json_request(f'{base}/api/jobs/{job}')
        if state['status'] != 'running':
            break
        time.sleep(2)
    save('analysis.json', state)
    assert state['status'] == 'done', state.get('error')
    products = state['result']['shopping_results']
    ready = [p for p in products if p['category'] == 'shoes' and p.get('tryon_available')]
    assert ready, products
    print('shoe_products', len(ready), flush=True)
    while time.monotonic() < deadline:
        batch = _json_request(f'{base}/api/jobs/{job}/shopping-tryon-batch')
        save('batch.json', batch)
        if batch and batch['status'] not in ('queued', 'running'):
            break
        time.sleep(3)
    assert batch['status'] == 'done', batch
    assert len(batch['items']) == len(ready), batch
    for item in batch['items']:
        assert item['categories'] == ['shoes'] and item['status'] == 'done', item
        kind, image = _download(f"{base}/api/jobs/{job}/images/{item['image']}")
        assert kind == 'image/jpeg' and image.startswith(b'\xff\xd8\xff')
        (args.output / Path(item['image']).name).write_bytes(image)
    cached = _json_request(f'{base}/api/jobs/{job}/tryon-products', method='POST',
                           data=json.dumps({'product_ids': [ready[0]['product_id']]}).encode(),
                           headers={'Content-Type': 'application/json'})
    save('cached.json', cached)
    assert cached['cached'], cached
    print('SHOE_API_SMOKE_OK', len(batch['items']), 'images; cache verified', flush=True)


if __name__ == '__main__':
    main()
