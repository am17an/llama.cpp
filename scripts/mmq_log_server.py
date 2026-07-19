#!/usr/bin/env python3

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_LOG = Path(__file__).resolve().parent.parent / "MMQ_PERF_LOG.md"


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GB10 Q4_K optimization logbook</title>
<style>
:root { color-scheme: dark; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
body { margin: 0; background: #0b1020; color: #dbeafe; }
main { max-width: 1120px; margin: auto; padding: 28px; }
h1 { margin: 0 0 6px; font-size: 24px; }
.status { color: #93c5fd; margin-bottom: 20px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
.card, .panel { background: #111a31; border: 1px solid #263656; border-radius: 10px; }
.card { padding: 14px; }
.card span { display: block; color: #8fa7c9; font-size: 12px; margin-bottom: 6px; }
.card strong { font-size: 20px; }
.panel { margin-top: 14px; padding: 16px; }
svg { display: block; width: 100%; height: 360px; }
pre { margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; color: #cbd5e1; line-height: 1.45; }
.axis { stroke: #61708c; stroke-width: 1; }
.grid { stroke: #273754; stroke-width: 1; }
.target { stroke: #f59e0b; stroke-width: 2; stroke-dasharray: 8 6; }
.series { fill: none; stroke: #38bdf8; stroke-width: 3; }
.point { fill: #38bdf8; }
.label { fill: #cbd5e1; font-size: 12px; }
.target-label { fill: #fbbf24; font-size: 12px; }
</style>
</head>
<body>
<main>
  <h1>GB10 Q4_K optimization logbook</h1>
  <div id="status" class="status">Connecting...</div>
  <section class="cards">
    <div class="card"><span>Baseline</span><strong id="baseline">-</strong></div>
    <div class="card"><span>Current best</span><strong id="best">-</strong></div>
    <div class="card"><span>Speedup</span><strong id="speedup">-</strong></div>
    <div class="card"><span>Reference target</span><strong id="target">-</strong></div>
  </section>
  <section class="panel"><svg id="chart" viewBox="0 0 1040 360" role="img"></svg></section>
  <section class="panel"><pre id="log"></pre></section>
</main>
<script>
const NS = "http://www.w3.org/2000/svg";
let lastMtime = null;

function node(name, attrs, text) {
  const item = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attrs || {})) item.setAttribute(key, value);
  if (text !== undefined) item.textContent = text;
  return item;
}

function renderChart(results, target) {
  const svg = document.getElementById("chart");
  svg.replaceChildren();
  const left = 72, right = 24, top = 24, bottom = 62;
  const width = 1040 - left - right, height = 360 - top - bottom;
  const values = results.map(item => item.throughput);
  const ymax = Math.max(target * 1.1, ...values.map(value => value * 1.1), 1);
  const x = index => left + (results.length === 1 ? width / 2 : index * width / (results.length - 1));
  const y = value => top + height - value * height / ymax;

  for (let tick = 0; tick <= 4; ++tick) {
    const value = ymax * tick / 4;
    const yy = y(value);
    svg.append(node("line", { x1: left, y1: yy, x2: left + width, y2: yy, class: "grid" }));
    svg.append(node("text", { x: left - 10, y: yy + 4, "text-anchor": "end", class: "label" }, value.toFixed(0)));
  }
  svg.append(node("line", { x1: left, y1: top, x2: left, y2: top + height, class: "axis" }));
  svg.append(node("line", { x1: left, y1: top + height, x2: left + width, y2: top + height, class: "axis" }));

  const targetY = y(target);
  svg.append(node("line", { x1: left, y1: targetY, x2: left + width, y2: targetY, class: "target" }));
  svg.append(node("text", { x: left + width - 4, y: targetY - 8, "text-anchor": "end", class: "target-label" }, `target ${target.toFixed(2)}`));

  if (results.length) {
    const points = results.map((item, index) => `${x(index)},${y(item.throughput)}`).join(" ");
    svg.append(node("polyline", { points, class: "series" }));
    results.forEach((item, index) => {
      const xx = x(index), yy = y(item.throughput);
      svg.append(node("circle", { cx: xx, cy: yy, r: 5, class: "point" }));
      svg.append(node("text", { x: xx, y: yy - 12, "text-anchor": "middle", class: "label" }, item.throughput.toFixed(2)));
      svg.append(node("text", { x: xx, y: top + height + 24, "text-anchor": "middle", class: "label" }, item.variant));
    });
  }
}

async function refresh() {
  const status = document.getElementById("status");
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const state = await response.json();
    status.textContent = `Live - refreshes every 2 seconds - ${new Date(state.mtime * 1000).toLocaleString()}`;
    if (state.mtime === lastMtime) return;
    lastMtime = state.mtime;
    const baseline = state.results.length ? state.results[0].throughput : 0;
    const best = state.results.length ? Math.max(...state.results.map(item => item.throughput)) : 0;
    document.getElementById("baseline").textContent = `${baseline.toFixed(2)} t/s`;
    document.getElementById("best").textContent = `${best.toFixed(2)} t/s`;
    document.getElementById("speedup").textContent = baseline ? `${(best / baseline).toFixed(3)}x` : "-";
    document.getElementById("target").textContent = `${state.target.toFixed(2)} t/s`;
    document.getElementById("log").textContent = state.log;
    renderChart(state.results, state.target);
  } catch (error) {
    status.textContent = `Disconnected: ${error.message}`;
  }
}

refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
"""


def parse_results(log_text):
    current_section = "## W4A16 sidecar experiment (July 17, 2026)"
    if current_section not in log_text:
        current_section = "## Progress"

    results = []
    high_water_mark = 0.0
    in_progress = False
    for line in log_text.splitlines():
        if line == current_section:
            in_progress = True
            continue
        if in_progress and line.startswith("## "):
            break
        if not in_progress or not line.startswith("|"):
            continue
        columns = [column.strip().strip("`") for column in line.strip("|").split("|")]
        if len(columns) != 5:
            continue
        throughput_match = re.match(r"([0-9]+(?:\.[0-9]+)?)", columns[2])
        relative_match = re.match(r"([0-9]+(?:\.[0-9]+)?)x", columns[3])
        if not throughput_match or not relative_match:
            continue
        throughput = float(throughput_match.group(1))
        if "rejected" in columns[4].lower() or throughput <= high_water_mark:
            continue
        high_water_mark = throughput
        results.append({
            "revision": columns[0],
            "variant": columns[1],
            "throughput": throughput,
            "relative": float(relative_match.group(1)),
            "result": columns[4],
        })
    return results


def read_state(log_path):
    log_text = log_path.read_text(encoding="utf-8")
    target_match = re.search(r"Dashboard target: ([0-9]+(?:\.[0-9]+)?) tokens/s", log_text)
    if not target_match:
        target_match = re.search(r"Reach at least ([0-9]+(?:\.[0-9]+)?) tokens/s", log_text)
    if not target_match:
        raise ValueError("target throughput is missing from the logbook")
    return {
        "log": log_text,
        "mtime": log_path.stat().st_mtime,
        "results": parse_results(log_text),
        "target": float(target_match.group(1)),
    }


def make_handler(log_path):
    class LogHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self.send_body(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/state":
                try:
                    body = json.dumps(read_state(log_path)).encode("utf-8")
                    self.send_body(body, "application/json; charset=utf-8")
                except (OSError, ValueError) as error:
                    self.send_error(500, str(error))
                return
            self.send_error(404)

        def send_body(self, body, content_type):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format_string, *args):
            print("%s - %s" % (self.address_string(), format_string % args), flush=True)

    return LogHandler


def main():
    parser = argparse.ArgumentParser(description="Serve the live MMQ performance logbook")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = parser.parse_args()

    log_path = args.log.resolve()
    read_state(log_path)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(log_path))
    print("Serving %s at http://%s:%d" % (log_path, args.host, args.port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
