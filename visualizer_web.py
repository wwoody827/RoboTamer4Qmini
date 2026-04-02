"""
Web-based real-time robot state visualizer for Qmini.
Works in WSL — open http://localhost:8080 in your Windows browser.

Tab 1 — 3D Robot: live FK rendering, toggle Live / Manual Pose mode.
Tab 2 — Time Plots: 8 signal charts from UDP data.

In Manual Pose mode, sliders drive the 3D view instantly (client-side FK).
Run mock_robot.py --mode pose to also broadcast those positions via UDP
so the plots update too.

Usage:
    python3 visualizer_web.py [--udp-port 9870] [--http-port 8080] [--history 200]
"""

import argparse
import json
import os
import socket
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── embedded HTML ─────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Qmini Monitor</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script type="importmap">
{
  "imports": {
    "three":          "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/":  "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
  }
}
</script>
<style>
*  { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #1a1a2e; color: #eee; font-family: monospace;
       display: flex; flex-direction: column; height: 100vh; overflow: hidden; }

/* ── header ── */
#header { padding: 5px 14px; background: #16213e; display: flex;
          align-items: center; justify-content: space-between; flex-shrink: 0; }
#header h2 { color: #e94560; font-size: 13px; }
#status { font-size: 11px; color: #aaa; }

/* ── tab bar ── */
#tabbar { display: flex; background: #0f0f1e; flex-shrink: 0; border-bottom: 1px solid #333; }
.tab-btn { padding: 7px 22px; background: none; border: none; color: #888;
           font-family: monospace; font-size: 12px; cursor: pointer; border-bottom: 2px solid transparent; }
.tab-btn.active { color: #e94560; border-bottom-color: #e94560; }

/* ── tab content ── */
.tab-pane { display: none; flex: 1; min-height: 0; flex-direction: column; }
.tab-pane.active { display: flex; }

/* ── 3D tab ── */
#tab-3d { position: relative; }
#canvas3d { display: block; width: 100%; flex: 1; min-height: 0; }
#mode-bar  { position: absolute; top: 10px; right: 14px; z-index: 10; display: flex; gap: 6px; }
.mode-btn  { padding: 5px 14px; border: 1px solid #555; border-radius: 3px; background: #1a1a2e;
             color: #aaa; font-family: monospace; font-size: 11px; cursor: pointer; }
.mode-btn.active { background: #e94560; border-color: #e94560; color: #fff; }
#loading3d { position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);
             color: #aaa; font-size: 12px; pointer-events: none; }

/* ── pose panel ── */
#pose-panel { background: #0f0f1e; border-top: 1px solid #333; overflow-y: auto;
              max-height: 0; transition: max-height 0.3s ease; flex-shrink: 0; }
#pose-panel.open { max-height: 260px; }
#pose-inner { display: grid; grid-template-columns: 1fr 1fr; gap: 0; }
.pose-col   { padding: 8px 14px; }
.pose-col h4 { font-size: 11px; color: #e94560; margin-bottom: 6px; }
.slider-row { display: flex; align-items: center; gap: 8px; margin-bottom: 5px; }
.slider-row label { width: 90px; font-size: 10px; color: #ccc; flex-shrink: 0; }
.slider-row input[type=range] { flex: 1; accent-color: #e94560; }
.slider-row .val { width: 48px; font-size: 10px; color: #0f9b8e; text-align: right; }
.slider-row input[type=number] { width: 54px; background: #1a1a2e; border: 1px solid #444;
                                  color: #0f9b8e; font-size: 10px; padding: 1px 3px; }
#pose-actions { padding: 6px 14px; display: flex; gap: 8px; align-items: center; }
.pose-act-btn { padding: 4px 12px; background: #16213e; border: 1px solid #555; color: #ccc;
                font-size: 10px; font-family: monospace; cursor: pointer; border-radius: 2px; }
.pose-act-btn:hover { border-color: #e94560; color: #e94560; }
#pose-hint { font-size: 10px; color: #555; margin-left: auto; }

/* ── plots tab ── */
#tab-plots { overflow-y: auto; }
#grid { display: grid; grid-template-columns: 1fr 1fr; gap: 3px; padding: 3px; }
.plot { background: #16213e; border-radius: 3px; }
</style>
</head>
<body>

<div id="header">
  <h2>Qmini Real-time Monitor</h2>
  <span id="status">Waiting for data...</span>
</div>

<div id="tabbar">
  <button class="tab-btn active" onclick="showTab('3d',this)">3D Robot</button>
  <button class="tab-btn"        onclick="showTab('plots',this)">Time Plots</button>
</div>

<!-- 3D tab -->
<div id="tab-3d" class="tab-pane active">
  <canvas id="canvas3d"></canvas>
  <div id="mode-bar">
    <button class="mode-btn active" id="btn-live"   onclick="setMode('live')">Live</button>
    <button class="mode-btn"        id="btn-manual" onclick="setMode('manual')">Manual Pose</button>
  </div>
  <div id="loading3d">Loading meshes...</div>

  <div id="pose-panel">
    <div id="pose-actions">
      <button class="pose-act-btn" onclick="resetToStand()">Reset to Stand</button>
      <button class="pose-act-btn" onclick="resetToZero()">Zero All</button>
    </div>
    <div id="pose-inner">
      <div class="pose-col" id="col-left"><h4>Left Leg</h4></div>
      <div class="pose-col" id="col-right"><h4>Right Leg</h4></div>
    </div>
  </div>
</div>

<!-- Plots tab -->
<div id="tab-plots" class="tab-pane">
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
</div>

<script>
// ── tab switching ────────────────────────────────────────────────────────────
function showTab(name, btn) {
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  if (btn) btn.classList.add('active');
  if (name === '3d') window._resizeRenderer && window._resizeRenderer();
}
</script>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader }     from 'three/addons/loaders/STLLoader.js';

// ── joint definitions from q1.urdf ──────────────────────────────────────────
const JOINTS = [
  { name:'hip_yaw_l',    parent:null,           xyz:[0,        0.055,   0       ], rpy:[0,0,   0.4], axis:[0,0,-1], udp:0 },
  { name:'hip_roll_l',   parent:'hip_yaw_l',    xyz:[0,        0.054,  -0.1073  ], rpy:[0,0,   0  ], axis:[1,0, 0], udp:1 },
  { name:'hip_pitch_l',  parent:'hip_roll_l',   xyz:[0.0165,   0.028,   0       ], rpy:[0,1.5, 0  ], axis:[0,1, 0], udp:2 },
  { name:'knee_pitch_l', parent:'hip_pitch_l',  xyz:[-0.081317,-0.0003,-0.081317], rpy:[0,1.05,0  ], axis:[0,-1,0], udp:3 },
  { name:'ankle_pitch_l',parent:'knee_pitch_l', xyz:[0.053013, 0.0228, -0.14565 ], rpy:[0,1.22,0  ], axis:[0,1, 0], udp:4 },
  { name:'hip_yaw_r',    parent:null,           xyz:[0,       -0.055,   0       ], rpy:[0,0,  -0.4], axis:[0,0,-1], udp:5 },
  { name:'hip_roll_r',   parent:'hip_yaw_r',    xyz:[0,       -0.054,  -0.1073  ], rpy:[0,0,   0  ], axis:[1,0, 0], udp:6 },
  { name:'hip_pitch_r',  parent:'hip_roll_r',   xyz:[0.0165,  -0.028,   0       ], rpy:[0,1.5, 0  ], axis:[0,-1,0], udp:7 },
  { name:'knee_pitch_r', parent:'hip_pitch_r',  xyz:[-0.081317,0.0003, -0.081317], rpy:[0,1.05,0  ], axis:[0,1, 0], udp:8 },
  { name:'ankle_pitch_r',parent:'knee_pitch_r', xyz:[0.053013,-0.0228, -0.14565 ], rpy:[0,1.22,0  ], axis:[0,-1,0], udp:9 },
];

// Joint limits from URDF (for sliders)
const JOINT_LIMITS = [
  { label:'Hip Yaw L',   min:-0.1, max:0.7,  def: 0.4  },
  { label:'Hip Roll L',  min:-0.3, max:0.6,  def:-0.1  },
  { label:'Hip Pitch L', min:-2.1, max:0.0,  def:-1.5  },
  { label:'Knee L',      min: 0.0, max:2.1,  def: 1.0  },
  { label:'Ankle L',     min:-2.5, max:0.0,  def:-1.3  },
  { label:'Hip Yaw R',   min:-0.7, max:0.1,  def:-0.4  },
  { label:'Hip Roll R',  min:-0.6, max:0.3,  def: 0.1  },
  { label:'Hip Pitch R', min: 0.0, max:2.1,  def: 1.5  },
  { label:'Knee R',      min:-2.1, max:0.0,  def:-1.0  },
  { label:'Ankle R',     min: 0.0, max:2.5,  def: 1.3  },
];

// ── Three.js scene ───────────────────────────────────────────────────────────
const canvas    = document.getElementById('canvas3d');
const renderer  = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x0d0d1a);

const scene  = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.001, 20);
camera.position.set(0.6, 0.5, 0.8);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.2, 0);
controls.enableDamping = true;

scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.2);
sun.position.set(1, 3, 2); scene.add(sun);
const fill = new THREE.DirectionalLight(0x4466bb, 0.4);
fill.position.set(-2, 0, -1); scene.add(fill);
scene.add(new THREE.GridHelper(2, 20, 0x334466, 0x222244));

// Z-up → Y-up frame
const frameNode = new THREE.Object3D();
frameNode.rotation.x = -Math.PI / 2;
frameNode.position.y = 0.55;
scene.add(frameNode);

const baseOrientNode = new THREE.Object3D();
frameNode.add(baseOrientNode);

const matPurple = new THREE.MeshPhongMaterial({ color:0x9977cc, specular:0x222, shininess:40 });
const matBlue   = new THREE.MeshPhongMaterial({ color:0x6699ee, specular:0x222, shininess:40 });
const matBase   = new THREE.MeshPhongMaterial({ color:0x5588bb, specular:0x222, shininess:40 });

let meshesLeft = JOINTS.length + 1;
const stlLoader = new STLLoader();
function loadSTL(url, parent, mat) {
  stlLoader.load(url, geo => {
    geo.computeVertexNormals();
    parent.add(new THREE.Mesh(geo, mat));
    if (--meshesLeft === 0) document.getElementById('loading3d').style.display = 'none';
  }, undefined, () => { if (--meshesLeft === 0) document.getElementById('loading3d').style.display = 'none'; });
}

loadSTL('/assets/q1/meshes/base_link.STL', baseOrientNode, matBase);

const jointNodes = {};
for (const j of JOINTS) {
  const par = j.parent ? jointNodes[j.parent].rotNode : baseOrientNode;
  const orig = new THREE.Object3D();
  orig.position.set(...j.xyz);
  orig.rotation.set(j.rpy[0], j.rpy[1], j.rpy[2]);
  par.add(orig);
  const rot = new THREE.Object3D();
  orig.add(rot);
  loadSTL(`/assets/q1/meshes/${j.name}.STL`, rot, j.name.includes('ankle') ? matBlue : matPurple);
  jointNodes[j.name] = { rotNode: rot, axis: new THREE.Vector3(...j.axis) };
}

// ── FK ───────────────────────────────────────────────────────────────────────
function applyPose(angles) {
  for (const j of JOINTS) {
    const { rotNode, axis } = jointNodes[j.name];
    rotNode.quaternion.setFromAxisAngle(axis, angles[j.udp]);
  }
}
function applyBaseRPY(rpy) {
  baseOrientNode.rotation.set(rpy[0], rpy[1], rpy[2], 'XYZ');
}

// Apply default stand pose immediately
applyPose(JOINT_LIMITS.map(j => j.def));

// ── resize ───────────────────────────────────────────────────────────────────
function resizeRenderer() {
  const c = canvas.parentElement;
  const w = c.clientWidth, h = canvas.clientHeight || c.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window._resizeRenderer = resizeRenderer;
window.addEventListener('resize', resizeRenderer);
resizeRenderer();

// ── render loop ──────────────────────────────────────────────────────────────
(function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
})();

// ── mode ─────────────────────────────────────────────────────────────────────
let mode = 'live';
window.setMode = function(m) {
  mode = m;
  document.getElementById('btn-live').classList.toggle('active',   m === 'live');
  document.getElementById('btn-manual').classList.toggle('active', m === 'manual');
  document.getElementById('pose-panel').classList.toggle('open', m === 'manual');
  setTimeout(resizeRenderer, 350);
};

// ── pose state ───────────────────────────────────────────────────────────────
const manualPose = JOINT_LIMITS.map(j => j.def);

window.onSliderChange = function(i, val) {
  manualPose[i] = parseFloat(val);
  document.getElementById('sv' + i).value = parseFloat(val).toFixed(3);
  if (mode === 'manual') applyPose(manualPose);
};
window.onNumberChange = function(i, val) {
  const v = Math.max(JOINT_LIMITS[i].min, Math.min(JOINT_LIMITS[i].max, parseFloat(val) || 0));
  manualPose[i] = v;
  document.getElementById('sl' + i).value = v;
  if (mode === 'manual') applyPose(manualPose);
};
window.resetToStand = function() {
  JOINT_LIMITS.forEach((j, i) => {
    manualPose[i] = j.def;
    document.getElementById('sl' + i).value = j.def;
    document.getElementById('sv' + i).value = j.def.toFixed(3);
  });
  if (mode === 'manual') applyPose(manualPose);
};
window.resetToZero = function() {
  JOINT_LIMITS.forEach((j, i) => {
    manualPose[i] = 0;
    document.getElementById('sl' + i).value = 0;
    document.getElementById('sv' + i).value = '0.000';
  });
  if (mode === 'manual') applyPose(manualPose);
};

// ── build sliders ─────────────────────────────────────────────────────────────
const colL = document.getElementById('col-left');
const colR = document.getElementById('col-right');
JOINT_LIMITS.forEach((j, i) => {
  const col = i < 5 ? colL : colR;
  col.insertAdjacentHTML('beforeend', `
    <div class="slider-row">
      <label>${j.label}</label>
      <input id="sl${i}" type="range" min="${j.min}" max="${j.max}" step="0.01" value="${j.def}"
             oninput="onSliderChange(${i}, this.value)">
      <input id="sv${i}" type="number" min="${j.min}" max="${j.max}" step="0.01" value="${j.def.toFixed(3)}"
             onchange="onNumberChange(${i}, this.value)">
    </div>`);
});

// ── Plotly 2D charts ─────────────────────────────────────────────────────────
const Plotly = window.Plotly;
const JNAMES = ['l_hyaw','l_hrol','l_hpit','l_knee','l_apit',
                'r_hyaw','r_hrol','r_hpit','r_knee','r_apit'];
const L = [0,1,2,3,4], R = [5,6,7,8,9];
const C5   = ['#e94560','#0f9b8e','#f5a623','#7b68ee','#50fa7b'];
const CRPY = ['#e94560','#0f9b8e','#7b68ee'];
const BG = '#1a1a2e', PBG = '#16213e', GC = '#2a2a4a';

function layout(title) {
  return {
    title:{ text:title, font:{color:'#ccc',size:10} },
    paper_bgcolor:PBG, plot_bgcolor:BG, font:{color:'#ccc',size:9},
    margin:{l:38,r:6,t:24,b:22}, height:180,
    xaxis:{gridcolor:GC, zeroline:false},
    yaxis:{gridcolor:GC, zeroline:true, zerolinecolor:GC},
    legend:{font:{size:8}, bgcolor:'rgba(0,0,0,0)', orientation:'h', y:-0.2},
    showlegend:true,
  };
}
function tr(name, x, y, color, dash) {
  return { x, y, name, type:'scatter', mode:'lines',
           line:{ color, width:1.4, dash: dash||'solid' } };
}

['p0','p1','p2','p3','p4','p5','p6','p7'].forEach((id, i) => {
  Plotly.newPlot(id, [], layout(['Joint Pos L','Joint Pos R',
    'Joint Vel L','Joint Vel R','Base RPY','RPY Rate','Torque L','Torque R'][i] + ' '));
});

// ── data poll ────────────────────────────────────────────────────────────────
setInterval(async () => {
  try {
    const res = await fetch('/data');
    if (!res.ok) return;
    const d = await res.json();
    if (!d.n || d.n < 2) return;
    const { t, act, pos, vel, tau, rpy, rr, n } = d;

    // 3D: only update from UDP in live mode
    if (mode === 'live') {
      applyPose(pos.map(arr => arr[n-1]));
      applyBaseRPY(rpy.map(arr => arr[n-1]));
    }

    // Plots: always update
    Plotly.react('p0', [...L.map((ji,k)=>tr(JNAMES[ji],t,pos[ji],C5[k])),
                        ...L.map((ji,k)=>tr(JNAMES[ji]+'_act',t,act[ji],C5[k],'dot'))],
                 layout('Joint Pos — Left (rad)'));
    Plotly.react('p1', [...R.map((ji,k)=>tr(JNAMES[ji],t,pos[ji],C5[k])),
                        ...R.map((ji,k)=>tr(JNAMES[ji]+'_act',t,act[ji],C5[k],'dot'))],
                 layout('Joint Pos — Right (rad)'));
    Plotly.react('p2', L.map((ji,k)=>tr(JNAMES[ji],t,vel[ji],C5[k])), layout('Joint Vel — Left (rad/s)'));
    Plotly.react('p3', R.map((ji,k)=>tr(JNAMES[ji],t,vel[ji],C5[k])), layout('Joint Vel — Right (rad/s)'));
    Plotly.react('p4', ['roll','pitch','yaw'].map((n,i)=>tr(n,t,rpy[i],CRPY[i])), layout('Base RPY (rad)'));
    Plotly.react('p5', ['roll','pitch','yaw'].map((n,i)=>tr(n+'_r',t,rr[i],CRPY[i])), layout('RPY Rate (rad/s)'));
    Plotly.react('p6', L.map((ji,k)=>tr(JNAMES[ji],t,tau[ji],C5[k])), layout('Torque — Left (N·m)'));
    Plotly.react('p7', R.map((ji,k)=>tr(JNAMES[ji],t,tau[ji],C5[k])), layout('Torque — Right (N·m)'));

    document.getElementById('status').textContent =
      `roll=${rpy[0][n-1].toFixed(3)}  pitch=${rpy[1][n-1].toFixed(3)}  yaw=${rpy[2][n-1].toFixed(3)}  [${n} frames @ 20Hz]`;
  } catch(e) {}
}, 100);
</script>
</body>
</html>
"""


# ── UDP receiver ──────────────────────────────────────────────────────────────
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
        with self.lock:
            if len(self.buf) < 2:
                return {"n": 0}
            arr = np.stack(self.buf, axis=0)
        n = len(arr)
        return {
            "n":   n,
            "t":   list(range(n)),
            "act": [arr[:, i].tolist()    for i in range(10)],
            "pos": [arr[:, 10+i].tolist() for i in range(10)],
            "vel": [arr[:, 20+i].tolist() for i in range(10)],
            "tau": [arr[:, 30+i].tolist() for i in range(10)],
            "rpy": [arr[:, 40+i].tolist() for i in range(3)],
            "rr":  [arr[:, 43+i].tolist() for i in range(3)],
            "acc": [arr[:, 46+i].tolist() for i in range(3)],
        }

    def stop(self):
        self._stop.set()


# ── HTTP handler ──────────────────────────────────────────────────────────────
def make_handler(receiver: UDPReceiver):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            if self.path == '/':
                self._send(200, 'text/html', HTML.encode())

            elif self.path == '/data':
                body = json.dumps(receiver.get_json()).encode()
                self._send(200, 'application/json', body,
                           extra=[('Cache-Control', 'no-cache')])

            elif self.path.startswith('/assets/'):
                fp = os.path.join(REPO_ROOT, self.path.lstrip('/'))
                if os.path.isfile(fp):
                    with open(fp, 'rb') as f:
                        body = f.read()
                    ext = os.path.splitext(fp)[1].lower()
                    ct  = {'.stl': 'model/stl'}.get(ext, 'application/octet-stream')
                    self._send(200, ct, body)
                else:
                    self.send_response(404); self.end_headers()
            else:
                self.send_response(404); self.end_headers()

        def _send(self, code, ct, body, extra=()):
            self.send_response(code)
            self.send_header('Content-Type', ct)
            self.send_header('Content-Length', str(len(body)))
            for k, v in extra:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

    return Handler


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Qmini web visualizer')
    parser.add_argument('--udp-port',  type=int, default=9870)
    parser.add_argument('--http-port', type=int, default=8080)
    parser.add_argument('--history',   type=int, default=200)
    args = parser.parse_args()

    rx = UDPReceiver(port=args.udp_port, history=args.history)
    rx.start()

    server = HTTPServer(('0.0.0.0', args.http_port), make_handler(rx))
    print(f"Open in your Windows browser → http://localhost:{args.http_port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        rx.stop()
        server.server_close()


if __name__ == '__main__':
    main()
