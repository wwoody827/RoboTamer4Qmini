"""
Mock robot UDP broadcaster for debugging visualizer_web.py without a real robot.

Sends 53-float packets at 20Hz on the same port/format as RoboTamerSdk4Qmini.

Usage:
    python3 tools/mock_robot.py [--port 9870] [--mode walk|stand|sin]

Modes:
    stand  — joints hold zero position, slow RPY sway
    walk   — sinusoidal gait pattern on all joints
    sin    — single-joint sin sweep across all joints sequentially
"""

import argparse
import math
import socket
import time

PORT    = 9870
ADDR    = '255.255.255.255'
RATE_HZ = 20

# Joint index labels (for reference)
# 0:l_hyaw  1:l_hrol  2:l_hpit  3:l_knee  4:l_apit
# 5:r_hyaw  6:r_hrol  7:r_hpit  8:r_knee  9:r_apit


def make_packet_stand(t: float) -> list:
    """Gentle sway — joints near zero, RPY oscillates slowly."""
    joint_act = [0.0] * 10
    joint_pos = [0.02 * math.sin(t * 0.5 + i * 0.3) for i in range(10)]
    joint_vel = [0.02 * math.cos(t * 0.5 + i * 0.3) for i in range(10)]
    joint_tau = [0.5  * math.sin(t * 0.5 + i * 0.3) for i in range(10)]

    roll  =  0.05 * math.sin(t * 0.8)
    pitch =  0.03 * math.sin(t * 0.6 + 1.0)
    yaw   =  0.01 * math.sin(t * 0.2)
    base_rpy      = [roll, pitch, yaw]
    base_rpy_rate = [0.04 * math.cos(t * 0.8),
                     0.018 * math.cos(t * 0.6 + 1.0),
                     0.002 * math.cos(t * 0.2)]
    base_acc  = [0.1 * math.sin(t), 0.05 * math.cos(t), 9.81 + 0.02 * math.sin(t * 2)]
    base_quat = _rpy_to_quat(roll, pitch, yaw)

    return joint_act + joint_pos + joint_vel + joint_tau + base_rpy + base_rpy_rate + base_acc + base_quat


def make_packet_walk(t: float) -> list:
    """Alternating gait: left/right legs out of phase."""
    freq = 1.5  # Hz

    def leg(phase_offset, t):
        hyaw =  0.0
        hrol =  0.05 * math.sin(2 * math.pi * freq * t + phase_offset)
        hpit =  0.4  * math.sin(2 * math.pi * freq * t + phase_offset)
        knee = -0.8  * abs(math.sin(2 * math.pi * freq * t + phase_offset))
        apit =  0.4  * math.sin(2 * math.pi * freq * t + phase_offset + math.pi)
        return [hyaw, hrol, hpit, knee, apit]

    left  = leg(0,       t)
    right = leg(math.pi, t)

    joint_act = left + right
    noise     = [0.01 * math.sin(t * 13.7 + i) for i in range(10)]
    joint_pos = [joint_act[i] + noise[i] for i in range(10)]
    joint_vel = [2 * math.pi * freq * math.cos(2 * math.pi * freq * t + (0 if i < 5 else math.pi)) * 0.4
                 for i in range(10)]
    joint_tau = [5.0 * math.sin(2 * math.pi * freq * t + (0 if i < 5 else math.pi) + i * 0.1)
                 for i in range(10)]

    roll  =  0.08 * math.sin(2 * math.pi * freq * t)
    pitch =  0.04 * math.sin(2 * math.pi * freq * t * 2)
    yaw   =  0.005 * t % (2 * math.pi)
    base_rpy      = [roll, pitch, yaw]
    base_rpy_rate = [0.08 * 2 * math.pi * freq * math.cos(2 * math.pi * freq * t),
                     0.04 * 4 * math.pi * freq * math.cos(4 * math.pi * freq * t),
                     0.005]
    base_acc  = [0.3 * math.sin(t * freq * 2 * math.pi),
                 0.1 * math.cos(t * freq * 2 * math.pi),
                 9.81 + 0.1 * math.sin(t * freq * 4 * math.pi)]
    base_quat = _rpy_to_quat(roll, pitch, yaw)

    return joint_act + joint_pos + joint_vel + joint_tau + base_rpy + base_rpy_rate + base_acc + base_quat


def make_packet_sin(t: float) -> list:
    """Slow sin sweep through all joints one at a time."""
    period    = 3.0
    amp       = 0.5
    joint_idx = int(t / period) % 10
    phase     = (t % period) / period * 2 * math.pi

    joint_act = [0.0] * 10
    joint_act[joint_idx] = amp * math.sin(phase)
    joint_pos = [joint_act[i] + 0.01 * math.sin(t * 7 + i) for i in range(10)]
    joint_vel = [0.0] * 10
    joint_vel[joint_idx] = amp * (2 * math.pi / period) * math.cos(phase)
    joint_tau = [0.0] * 10
    joint_tau[joint_idx] = 3.0 * math.sin(phase)

    base_rpy      = [0.0, 0.0, 0.0]
    base_rpy_rate = [0.0, 0.0, 0.0]
    base_acc      = [0.0, 0.0, 9.81]
    base_quat     = [1.0, 0.0, 0.0, 0.0]

    return joint_act + joint_pos + joint_vel + joint_tau + base_rpy + base_rpy_rate + base_acc + base_quat


def _rpy_to_quat(r, p, y) -> list:
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return [qw, qx, qy, qz]


MODES = {
    'stand': make_packet_stand,
    'walk':  make_packet_walk,
    'sin':   make_packet_sin,
}


def main():
    parser = argparse.ArgumentParser(description='Mock robot UDP broadcaster')
    parser.add_argument('--port', type=int, default=PORT,
                        help=f'UDP broadcast port (default: {PORT})')
    parser.add_argument('--mode', choices=list(MODES), default='walk',
                        help='Motion pattern to simulate (default: walk)')
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    interval    = 1.0 / RATE_HZ
    make_packet = MODES[args.mode]

    print(f"Mock robot broadcasting → {ADDR}:{args.port}  mode={args.mode}  rate={RATE_HZ}Hz")
    print("Ctrl-C to stop.")

    t0 = time.time()
    try:
        while True:
            t      = time.time() - t0
            values = make_packet(t)
            assert len(values) == 53, f"Packet length {len(values)} != 53"
            msg = ','.join(f'{v:.6f}' for v in values)
            sock.sendto(msg.encode(), (ADDR, args.port))
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sock.close()


if __name__ == '__main__':
    main()
