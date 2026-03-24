"""Deploy a trained policy to a real Petoi Bittle robot.

Reads sensor data from the Bittle (joint positions + IMU via serial),
runs the neural network controller, and sends joint angle commands back.

Requirements:
  - Petoi Bittle X V2 with feedback servos (post-May 2024)
  - USB serial connection (or WiFi via ESP32)
  - pyserial: uv add pyserial

Usage:
  # List available serial ports
  uv run python sim2real/deploy_bittle.py --list-ports

  # Deploy trained weights
  uv run python sim2real/deploy_bittle.py --weights /data/evo-embodied/YYYYMMDD/best_weights.npy

  # Dry run (print commands, don't send to robot)
  uv run python sim2real/deploy_bittle.py --weights best_weights.npy --dry-run

  # Specify serial port
  uv run python sim2real/deploy_bittle.py --weights best_weights.npy --port /dev/ttyUSB0
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np


# ── Petoi Bittle Joint Map ─────────────────────────────────────────
# MuJoCo actuator index → Petoi servo index
# Petoi protocol: m <joint_index> <angle_degrees>
MUJOCO_TO_PETOI = {
    0: 8,   # servo_shoulder_fl → Petoi joint 8
    1: 12,  # servo_knee_fl     → Petoi joint 12
    2: 9,   # servo_shoulder_fr → Petoi joint 9
    3: 13,  # servo_knee_fr     → Petoi joint 13
    4: 11,  # servo_shoulder_bl → Petoi joint 11
    5: 15,  # servo_knee_bl     → Petoi joint 15
    6: 10,  # servo_shoulder_br → Petoi joint 10
    7: 14,  # servo_knee_br     → Petoi joint 14
}

# Default servo offsets (calibrate per robot)
SERVO_OFFSETS = {8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}


class BittleSerial:
    """Communication with Petoi Bittle via serial protocol."""

    def __init__(self, port="/dev/ttyUSB0", baudrate=115200, dry_run=False):
        self.dry_run = dry_run
        self.port_name = port
        self.ser = None

        if not dry_run:
            try:
                import serial
                self.ser = serial.Serial(port, baudrate, timeout=0.5)
                time.sleep(2)  # wait for ESP32 boot
                self._flush()
                print(f"Connected to Bittle on {port}")
            except ImportError:
                print("ERROR: pyserial not installed. Run: uv add pyserial")
                raise
            except Exception as e:
                print(f"ERROR: Cannot open {port}: {e}")
                raise
        else:
            print(f"DRY RUN mode (no serial connection)")

    def _flush(self):
        if self.ser:
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()

    def send(self, cmd):
        """Send a command string to Bittle."""
        if self.dry_run:
            return
        self.ser.write(f"{cmd}\n".encode())
        time.sleep(0.005)  # small delay for ESP32

    def set_joint(self, petoi_idx, angle_degrees):
        """Set a single joint to an angle (degrees)."""
        angle = int(round(angle_degrees))
        angle = max(-125, min(125, angle))  # Bittle servo range
        self.send(f"m {petoi_idx} {angle}")

    def set_all_joints(self, angles_dict):
        """Set multiple joints simultaneously using 'i' command.

        angles_dict: {petoi_joint_idx: angle_degrees, ...}
        """
        # Petoi 'i' command: i <idx0> <angle0> <idx1> <angle1> ...
        parts = []
        for idx, angle in sorted(angles_dict.items()):
            a = int(round(max(-125, min(125, angle))))
            parts.append(f"{idx} {a}")
        cmd = "i " + " ".join(parts)
        if self.dry_run:
            print(f"  CMD: {cmd}")
        self.send(cmd)

    def read_feedback(self):
        """Read feedback servo positions (Bittle X V2 only).

        Returns dict of {petoi_joint_idx: angle_degrees} or None.
        """
        if self.dry_run:
            return None
        self.send("f")
        time.sleep(0.05)
        response = self.ser.readline().decode().strip()
        if response:
            try:
                angles = [int(x) for x in response.split()]
                # Feedback returns all 16 servo positions
                return {i: angles[i] for i in range(len(angles))
                        if i in MUJOCO_TO_PETOI.values()}
            except (ValueError, IndexError):
                return None
        return None

    def read_imu(self):
        """Read IMU data (roll, pitch, yaw in degrees).

        Returns (roll, pitch, yaw) or None.
        """
        if self.dry_run:
            return None
        self.send("v")  # 'v' command returns IMU data
        time.sleep(0.05)
        response = self.ser.readline().decode().strip()
        if response:
            try:
                vals = [float(x) for x in response.split()]
                if len(vals) >= 3:
                    return vals[:3]
            except ValueError:
                pass
        return None

    def rest(self):
        """Send rest pose."""
        self.send("d")
        print("Sent rest pose")

    def balance(self):
        """Send balance pose."""
        self.send("kbalance")
        print("Sent balance pose")

    def close(self):
        if self.ser:
            self.rest()
            time.sleep(0.5)
            self.ser.close()


class NeuralController:
    """Neural network controller matching the sim training architecture."""

    def __init__(self, weights_path, n_hidden=64, clock_freq=2.0):
        weights = np.load(weights_path)
        # 8 actuators, infer n_sensors from weight dimensions
        self.n_motors = 8
        self.n_hidden = n_hidden
        self.clock_freq = clock_freq

        # Network: (n_sensors + 2) → n_hidden → n_motors
        n_inputs_hidden = weights.size - n_hidden * self.n_motors
        self.n_inputs = n_inputs_hidden // n_hidden
        n_sensors = self.n_inputs - 2

        self.w1 = weights[:self.n_inputs * n_hidden].reshape(
            self.n_inputs, n_hidden)
        self.w2 = weights[self.n_inputs * n_hidden:].reshape(
            n_hidden, self.n_motors)

        print(f"Loaded controller: {n_sensors} sensors + 2 clock → "
              f"{n_hidden} hidden → {self.n_motors} motors")
        print(f"Weight shapes: w1={self.w1.shape}, w2={self.w2.shape}")

    def __call__(self, sensor_data, t):
        """Compute motor commands from sensor data and time.

        Args:
            sensor_data: numpy array of sensor readings
            t: current time in seconds

        Returns:
            numpy array of 8 motor commands in radians
        """
        clock_sin = math.sin(2.0 * math.pi * self.clock_freq * t)
        clock_cos = math.cos(2.0 * math.pi * self.clock_freq * t)

        inputs = np.concatenate([
            sensor_data[:self.n_inputs - 2],
            [clock_sin, clock_cos]
        ])

        hidden = np.tanh(inputs @ self.w1)
        commands = np.tanh(hidden @ self.w2)
        return commands


def deploy(weights_path, port, dry_run, n_hidden, clock_freq, ctrl_hz, duration):
    """Main deployment loop."""
    controller = NeuralController(weights_path, n_hidden, clock_freq)
    bittle = BittleSerial(port, dry_run=dry_run)

    ctrl_dt = 1.0 / ctrl_hz
    n_steps = int(duration / ctrl_dt)

    print(f"\nDeploying at {ctrl_hz} Hz for {duration}s ({n_steps} steps)")
    print("Press Ctrl+C to stop\n")

    # Start from balance pose
    bittle.balance()
    time.sleep(1.0)

    try:
        t_start = time.time()
        for step in range(n_steps):
            t = step * ctrl_dt

            # Read sensors (use zeros if unavailable)
            feedback = bittle.read_feedback()
            imu = bittle.read_imu()

            if feedback is not None:
                # Build sensor array: joint positions from feedback servos
                joint_angles_rad = np.zeros(8)
                for mj_idx, petoi_idx in MUJOCO_TO_PETOI.items():
                    if petoi_idx in feedback:
                        joint_angles_rad[mj_idx] = math.radians(
                            feedback[petoi_idx])
                sensor_data = joint_angles_rad
            else:
                sensor_data = np.zeros(8)

            if imu is not None:
                # Append IMU readings (converted to radians)
                imu_rad = np.array([math.radians(x) for x in imu])
                sensor_data = np.concatenate([sensor_data, imu_rad])

            # Run controller
            commands_rad = controller(sensor_data, t)

            # Convert radians to degrees and send
            angles_deg = {}
            for mj_idx in range(8):
                petoi_idx = MUJOCO_TO_PETOI[mj_idx]
                angle_deg = math.degrees(commands_rad[mj_idx])
                angle_deg += SERVO_OFFSETS.get(petoi_idx, 0)
                angles_deg[petoi_idx] = angle_deg

            bittle.set_all_joints(angles_deg)

            if dry_run and step % 10 == 0:
                angles_str = " ".join(f"{v:+.0f}" for v in angles_deg.values())
                print(f"  t={t:.2f}s  angles: {angles_str}")

            # Maintain control rate
            elapsed = time.time() - t_start
            target = (step + 1) * ctrl_dt
            if target > elapsed:
                time.sleep(target - elapsed)

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        bittle.close()
        print("Done")


def list_ports():
    """List available serial ports."""
    try:
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        if ports:
            print("Available serial ports:")
            for p in ports:
                print(f"  {p.device:20s} {p.description}")
        else:
            print("No serial ports found.")
            print("  - Is the Bittle connected via USB?")
            print("  - Try: sudo dmesg | grep tty")
    except ImportError:
        print("pyserial not installed. Run: uv add pyserial")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy policy to Petoi Bittle")
    parser.add_argument("--weights", type=str, help="Path to best_weights.npy")
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0",
                        help="Serial port (default: /dev/ttyUSB0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without sending to robot")
    parser.add_argument("--list-ports", action="store_true",
                        help="List available serial ports")
    parser.add_argument("--n-hidden", type=int, default=64,
                        help="Hidden layer size (must match training)")
    parser.add_argument("--clock-freq", type=float, default=2.0,
                        help="Clock frequency Hz (must match training)")
    parser.add_argument("--ctrl-hz", type=float, default=20.0,
                        help="Control loop frequency (default: 20 Hz)")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Duration in seconds (default: 30)")
    args = parser.parse_args()

    if args.list_ports:
        list_ports()
    elif args.weights:
        deploy(args.weights, args.port, args.dry_run,
               args.n_hidden, args.clock_freq, args.ctrl_hz, args.duration)
    else:
        parser.print_help()
