"""
Web-based real-time robot state visualizer for Qmini.
Works in WSL — open http://localhost:8080 in your Windows browser.

No extra dependencies beyond numpy (already in requirements.txt).

Usage:
    python3 visualizer_web.py [--udp-port 9870] [--http-port 8080] [--history 200]

Then open: http://localhost:8080
"""

import argparse
import json
import socket
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

# ── data layout ─────────────────────────────────────────────────────────────
JOINT_NAMES = ['l_hyaw','l_hrol','l_hpit','l_knee','l_apit',
               'r_hyaw','r_hrol','r_hpit','r_knee','r_apit']
LEFT_IDX  = list(range(0, 5))
RIGHT_IDX = list(range(5, 10))

# ── embedded HTML/JS ─────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Qmini Monitor</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body { margin: 0; background: #1a1a2e; color: #eee; font-family: monospace; }
  #header { padding: 10px 20px; background: #16213e; display: flex;
            align-items: center; justify-content: space-between; }
  #header h2 { margin: 0; color: #e94560; }
  #status { font-size: 12px; color: #aaa; }
  #grid { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; padding: 4px; }
  .plot { background: #16213e; border-radius: 4px; }
</style>
</head>
<body>
<div id="header">
  <h2>Qmini Real-time Monitor</h2>
  <span id="status">Waiting for data...</span>
</div>
<div id="grid">
  <div id="p0" class="plot"></div>
  <div id="p1" class="plot"></div>
  <div id="p2" class="plot"></div>
  <div id="p3" class="plot"></div>
  <div id="p4" class="plot"></div>
  <div id="p5" class="plot"></div>
  <div id="p6" class="plot"></div>
  <div id="p7" class="plot"></div>
</div>
<script>
const JOINT_NAMES = ['l_hyaw','l_hrol','l_hpit','l_knee','l_apit',
                     'r_hyaw','r_hrol','r_hpit','r_knee','r_apit'];
const LEFT  = [0,1,2,3,4];
const RIGHT = [5,6,7,8,9];
const RPY_NAMES = ['roll','pitch','yaw'];
const COLORS5 = ['#e94560','#0f9b8e','#f5a623','#7b68ee','#50fa7b'];
const RPY_COLORS = ['#e94560','#0f9b8e','#7b68ee'];
const PLOTBG  = '#1a1a2e';
const PAPERBG = '#16213e';
const GRID    = '#2a2a4a';
const H = 300;

function baseLayout(title) {
  return {
    title: { text: title, font: { color: '#ccc', size: 12 } },
    paper_bgcolor: PAPERBG, plot_bgcolor: PLOTBG,
    font: { color: '#ccc', size: 10 },
    margin: { l: 45, r: 10, t: 35, b: 30 },
    height: H,
    xaxis: { gridcolor: GRID, zeroline: false },
    yaxis: { gridcolor: GRID, zeroline: true, zerolinecolor: GRID },
    legend: { font: { size: 9 }, bgcolor: 'rgba(0,0,0,0)' },
    showlegend: true,
  };
}

function makeTraces(names, colors, dash) {
  return names.map((n, i) => ({
    x: [], y: [], name: n, type: 'scatter', mode: 'lines',
    line: { color: colors[i % colors.length], width: 1.5, dash: dash || 'solid' }
  }));
}

// ── initialise 8 plots ───────────────────────────────────────────────────────
// p0: joint pos left   (actual solid + commanded dashed)
// p1: joint pos right
// p2: joint vel left
// p3: joint vel right
// p4: RPY
// p5: RPY rate
// p6: torque left
// p7: torque right

const leftNames  = LEFT.map(i  => JOINT_NAMES[i]);
const rightNames = RIGHT.map(i => JOINT_NAMES[i]);

Plotly.newPlot('p0', [
  ...makeTraces(leftNames,  COLORS5, 'solid'),   // actual (0-4)
  ...makeTraces(leftNames.map(n=>n+'_act'), COLORS5, 'dot'),  // commanded (5-9)
], baseLayout('Joint Pos — Left Leg (rad)'));

Plotly.newPlot('p1', [
  ...makeTraces(rightNames, COLORS5, 'solid'),
  ...makeTraces(rightNames.map(n=>n+'_act'), COLORS5, 'dot'),
], baseLayout('Joint Pos — Right Leg (rad)'));

Plotly.newPlot('p2', makeTraces(leftNames,  COLORS5), baseLayout('Joint Vel — Left Leg (rad/s)'));
Plotly.newPlot('p3', makeTraces(rightNames, COLORS5), baseLayout('Joint Vel — Right Leg (rad/s)'));
Plotly.newPlot('p4', makeTraces(RPY_NAMES,  RPY_COLORS), baseLayout('Base RPY (rad)'));
Plotly.newPlot('p5', makeTraces(RPY_NAMES.map(n=>n+'_rate'), RPY_COLORS), baseLayout('Base RPY Rate (rad/s)'));
Plotly.newPlot('p6', makeTraces(leftNames,  COLORS5), baseLayout('Torque — Left Leg (N·m)'));
Plotly.newPlot('p7', makeTraces(rightNames, COLORS5), baseLayout('Torque — Right Leg (N·m)'));

// ── poll + update ────────────────────────────────────────────────────────────
let initialized = false;

async function poll() {
  try {
    const res = await fetch('/data');
    if (!res.ok) return;
    const d = await res.json();
    if (!d.n || d.n < 2) return;

    const t   = d.t;
    const act = d.act;   // [10][N]
    const pos = d.pos;
    const vel = d.vel;
    const tau = d.tau;
    const rpy = d.rpy;   // [3][N]
    const rr  = d.rr;
    const n   = d.n;

    // joint pos left: traces 0-4 = actual, 5-9 = commanded
    Plotly.react('p0',
      [...LEFT.map((ji,k)  => ({x:t, y:pos[ji], name:leftNames[k],  type:'scatter', mode:'lines', line:{color:COLORS5[k],width:1.5}})),
       ...LEFT.map((ji,k)  => ({x:t, y:act[ji], name:leftNames[k]+'_act', type:'scatter', mode:'lines', line:{color:COLORS5[k],width:1,dash:'dot'}}))],
      baseLayout('Joint Pos — Left Leg (rad)'));

    Plotly.react('p1',
      [...RIGHT.map((ji,k) => ({x:t, y:pos[ji], name:rightNames[k], type:'scatter', mode:'lines', line:{color:COLORS5[k],width:1.5}})),
       ...RIGHT.map((ji,k) => ({x:t, y:act[ji], name:rightNames[k]+'_act', type:'scatter', mode:'lines', line:{color:COLORS5[k],width:1,dash:'dot'}}))],
      baseLayout('Joint Pos — Right Leg (rad)'));

    Plotly.react('p2', LEFT.map( (ji,k) => ({x:t, y:vel[ji], name:leftNames[k],  type:'scatter', mode:'lines', line:{color:COLORS5[k],width:1.5}})), baseLayout('Joint Vel — Left Leg (rad/s)'));
    Plotly.react('p3', RIGHT.map((ji,k) => ({x:t, y:vel[ji], name:rightNames[k], type:'scatter', mode:'lines', line:{color:COLORS5[k],width:1.5}})), baseLayout('Joint Vel — Right Leg (rad/s)'));
    Plotly.react('p4', RPY_NAMES.map((n,i) => ({x:t, y:rpy[i], name:n, type:'scatter', mode:'lines', line:{color:RPY_COLORS[i],width:1.5}})), baseLayout('Base RPY (rad)'));
    Plotly.react('p5', RPY_NAMES.map((n,i) => ({x:t, y:rr[i],  name:n+'_rate', type:'scatter', mode:'lines', line:{color:RPY_COLORS[i],width:1.5}})), baseLayout('Base RPY Rate (rad/s)'));
    Plotly.react('p6', LEFT.map( (ji,k) => ({x:t, y:tau[ji], name:leftNames[k],  type:'scatter', mode:'lines', line:{color:COLORS5[k],width:1.5}})), baseLayout('Torque — Left Leg (N·m)'));
    Plotly.react('p7', RIGHT.map((ji,k) => ({x:t, y:tau[ji], name:rightNames[k], type:'scatter', mode:'lines', line:{color:COLORS5[k],width:1.5}})), baseLayout('Torque — Right Leg (N·m)'));

    const last_rpy = [rpy[0][n-1], rpy[1][n-1], rpy[2][n-1]];
    document.getElementById('status').textContent =
      `roll=${last_rpy[0].toFixed(3)}  pitch=${last_rpy[1].toFixed(3)}  yaw=${last_rpy[2].toFixed(3)}  [${n} frames @ 20Hz]`;

  } catch(e) { /* ignore fetch errors during shutdown */ }
}

setInterval(poll, 100);
</script>
</body>
</html>
"""


# ── UDP receiver ─────────────────────────────────────────────────────────────
class UDPReceiver:
    def __init__(self, port: int, history: int):
        self.buf   = deque(maxlen=history)
        self.lock  = threading.Lock()
        self.port  = port
        self._stop = threading.Event()

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('', self.port))
        sock.settimeout(1.0)
        print(f"Listening on UDP port {self.port} ...")
        while not self._stop.is_set():
            try:
                data, _ = sock.recvfrom(4096)
                values = np.array([float(x) for x in data.decode().split(',')],
                                  dtype=np.float32)
                if len(values) >= 53:
                    with self.lock:
                        self.buf.append(values)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[UDP] parse error: {e}")
        sock.close()

    def get_json(self) -> dict:
        """Return latest buffer as a JSON-serialisable dict, column-major."""
        with self.lock:
            if len(self.buf) < 2:
                return {"n": 0}
            arr = np.stack(self.buf, axis=0)   # (N, 53)

        n = len(arr)
        return {
            "n":   n,
            "t":   list(range(n)),
            "act": [arr[:, i].tolist() for i in range(10)],
            "pos": [arr[:, 10+i].tolist() for i in range(10)],
            "vel": [arr[:, 20+i].tolist() for i in range(10)],
            "tau": [arr[:, 30+i].tolist() for i in range(10)],
            "rpy": [arr[:, 40+i].tolist() for i in range(3)],
            "rr":  [arr[:, 43+i].tolist() for i in range(3)],
            "acc": [arr[:, 46+i].tolist() for i in range(3)],
        }

    def stop(self):
        self._stop.set()


# ── HTTP server ───────────────────────────────────────────────────────────────
def make_handler(receiver: UDPReceiver):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # silence request logs

        def do_GET(self):
            if self.path == '/':
                body = HTML.encode()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            elif self.path == '/data':
                body = json.dumps(receiver.get_json()).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                self.wfile.write(body)

            else:
                self.send_response(404)
                self.end_headers()

    return Handler


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Qmini web visualizer (WSL-friendly)')
    parser.add_argument('--udp-port',  type=int, default=9870,
                        help='UDP port to receive robot data (default: 9870)')
    parser.add_argument('--http-port', type=int, default=8080,
                        help='HTTP port for the browser UI (default: 8080)')
    parser.add_argument('--history',   type=int, default=200,
                        help='Frames to display (default: 200 = 10s at 20Hz)')
    args = parser.parse_args()

    rx = UDPReceiver(port=args.udp_port, history=args.history)
    rx.start()

    server = HTTPServer(('0.0.0.0', args.http_port), make_handler(rx))
    print(f"Open in your Windows browser: http://localhost:{args.http_port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        rx.stop()
        server.server_close()


if __name__ == '__main__':
    main()
