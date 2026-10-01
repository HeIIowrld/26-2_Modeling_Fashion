"""Exercise the real polling functions in Node, without model inference."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

SOURCE = Path(os.environ.get("FITTA_POLLING_TEST_SOURCE", str(Path(__file__).resolve().parents[1] / "static" / "app.js")))
pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="Node is required")


def run_js(body):
    source = SOURCE.read_text(encoding="utf-8")
    functions = source[source.index("function stopShoppingTryonBatchPolling("):source.index("/* 상위 탭:")]
    setup = """
const state = {jobId: 'job', shoppingTryonPoll: null};
const API_BASE = '';
const timers = new Map();
let started = 0;
let rendered = [];
const setInterval = (fn) => { const id = ++started; timers.set(id, fn); return id; };
const clearInterval = (id) => timers.delete(id);
const shoppingTryonResult = (item) => item;
const renderShoppingTryonPanel = () => {};
const refreshOutfitTryonRenders = () => { rendered = state.shoppingTryonResults.map(item => item.image); };
const toast = () => {};
"""
    result = subprocess.run(["node", "-e", setup + functions + "\n(async () => {" + body + "})().catch(e => {console.error(e); process.exit(1);});"],
                            capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(result.stdout)


@pytest.mark.parametrize("status", ["queued", "running"])
def test_initial_snapshot_polls_then_displays_completed_image(status):
    result = run_js("""
applyShoppingTryonBatch({status: STATUS, items: [{status: 'running'}]});
const initialTimers = timers.size;
global.fetch = async () => ({ok: true, json: async () => ({status: 'done', items: [
  {status: 'done', image: 'product_tryon_123.jpg'}
]})});
const tick = [...timers.values()][0];
if (tick) await tick();
console.log(JSON.stringify({initialTimers, remaining: timers.size, rendered}));
""".replace("STATUS", json.dumps(status)))
    assert result == {"initialTimers": 1, "remaining": 0, "rendered": ["product_tryon_123.jpg"]}


def test_repeated_running_snapshots_keep_one_timer():
    result = run_js("""
for (let i = 0; i < 3; i++) applyShoppingTryonBatch({status: 'running', items: []});
console.log(JSON.stringify({started, active: timers.size}));
""")
    assert result == {"started": 1, "active": 1}


@pytest.mark.parametrize("status", ["done", "partial", "failed", "unavailable"])
def test_terminal_snapshot_stops_polling(status):
    result = run_js("""
applyShoppingTryonBatch({status: 'running', items: []});
applyShoppingTryonBatch({status: STATUS, items: []});
console.log(JSON.stringify({active: timers.size, handle: state.shoppingTryonPoll}));
""".replace("STATUS", json.dumps(status)))
    assert result == {"active": 0, "handle": None}


def test_start_request_uses_same_polling_without_duplicate_timer():
    result = run_js("""
global.fetch = async () => ({ok: true, json: async () => ({status: 'running', items: []})});
await startShoppingTryonBatch();
console.log(JSON.stringify({started, active: timers.size}));
""")
    assert result == {"started": 1, "active": 1}


def test_missing_job_does_not_start_polling():
    result = run_js("""
state.jobId = null;
applyShoppingTryonBatch({status: 'running', items: []});
console.log(JSON.stringify(timers.size));
""")
    assert result == 0
