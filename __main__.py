#!/usr/bin/env python3
"""
h5loop — scan ไฟล์ HDF5 ในโฟลเดอร์ แล้วส่งภาพออก socket ทีละภาพ วนซ้ำไปเรื่อย ๆ จนกด ESC

    pip install h5py numpy opencv-python

ตัวอย่าง:
    python h5loop.py recv  --port 5555 --show
    python h5loop.py send  ./data --host 127.0.0.1 --port 5555 --interval-ms 50

GUI (recv --show):  Space = pause/play,  ←/→ = ทีละเฟรม,  ESC = ปิด
send:              --interval-ms 0 = ยิงเต็มที่ไม่หน่วง
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import queue
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass

import h5py
import numpy as np

MAGIC = b"H5IM"
HEADER = struct.Struct("!4sII")          # magic, meta_len, payload_len
SOCK_BUF = 4 * 1024 * 1024               # SO_SNDBUF / SO_RCVBUF


# ----------------------------------------------------------------- protocol ---

def send_frame(sock: socket.socket, meta: dict, payload: bytes) -> None:
    raw = json.dumps(meta).encode()
    sock.sendall(HEADER.pack(MAGIC, len(raw), len(payload)))
    sock.sendall(raw)
    if payload:
        sock.sendall(payload)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            raise ConnectionError("socket ถูกปิดก่อนข้อมูลจะครบ")
        buf += chunk
    return bytes(buf)


def recv_frame(sock: socket.socket) -> tuple[dict, bytes]:
    magic, mlen, plen = HEADER.unpack(_recv_exact(sock, HEADER.size))
    if magic != MAGIC:
        raise ValueError("magic ไม่ตรง — ปลายทางไม่ใช่ h5loop?")
    meta = json.loads(_recv_exact(sock, mlen))
    payload = _recv_exact(sock, plen) if plen else b""
    return meta, payload


# ---------------------------------------------------------------- timing ---

class HighResTimer:
    """จูนตัวนับเวลา Windows เป็น 1ms — sleep จะได้ไม่เผา 15.6ms ต่อครั้ง"""

    def __init__(self) -> None:
        self._ok = False
        if os.name == "nt":
            try:
                import ctypes
                self._wmm = ctypes.windll.winmm
                self._ok = self._wmm.timeBeginPeriod(1) == 0
            except Exception:
                self._ok = False

    def close(self) -> None:
        if self._ok:
            try:
                self._wmm.timeEndPeriod(1)
            except Exception:
                pass
            self._ok = False


def sleep_until(t_end: float) -> None:
    """นอนคร่าว ๆ แล้ว spin ช่วงท้ายเพื่อให้ตื่นตรงคาบ"""
    while True:
        remain = t_end - time.monotonic()
        if remain <= 0:
            return
        if remain > 0.004:
            time.sleep(remain - 0.003)
        # ช่วง ~3ms สุดท้ายวนเปล่า ๆ (ยอมใช้ CPU เล็กน้อยเพื่อจังหวะตรง)


# -------------------------------------------------------------------- scan ---

@dataclass
class Source:
    path: str
    dset: str
    n: int                    # จำนวนภาพใน dataset นี้
    shape: tuple
    dtype: str                # numpy dtype string เช่น '|u1', '<f4'

    @property
    def label(self) -> str:
        return f"{os.path.basename(self.path)}:{self.dset or '/'}"


def _auto_datasets(f: h5py.File) -> list[str]:
    names: list[str] = []
    f.visititems(lambda n, o: names.append(n)
                 if isinstance(o, h5py.Dataset) and o.ndim >= 2 else None)
    return sorted(names)


def scan(root: str, pattern: str, dataset: str | None) -> list[Source]:
    found: list[Source] = []
    paths = sorted(glob.glob(os.path.join(root, pattern)))
    for p in paths:
        try:
            with h5py.File(p, "r") as f:
                names = [dataset] if dataset else _auto_datasets(f)
                for name in names:
                    if name not in f:
                        continue
                    ds = f[name]
                    if ds.ndim < 2:
                        continue
                    n = ds.shape[0] if ds.ndim >= 3 else 1
                    found.append(Source(p, name, n,
                                        tuple(ds.shape), ds.dtype.str))
        except (OSError, KeyError) as e:
            print(f"[warn] เปิด {p} ไม่ได้: {e}", file=sys.stderr)
    return found


def grab(f: h5py.File, src: Source, i: int) -> np.ndarray:
    ds = f[src.dset]
    return ds[i] if src.n > 1 else ds[()]


# ------------------------------------------------------------- จับปุ่ม ESC ---

class EscListener:
    """อ่านคีย์แบบ non-blocking — ต้องรันใน terminal จริง"""

    def __init__(self) -> None:
        self._win = os.name == "nt"
        self._fd: int | None = None
        self._old = None
        if not sys.stdin.isatty():
            return
        if self._win:
            import msvcrt
            self._msvcrt = msvcrt
        else:
            import termios, tty
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)

    def pressed(self) -> bool:
        try:
            if self._win and hasattr(self, "_msvcrt"):
                while self._msvcrt.kbhit():
                    if self._msvcrt.getwch() == "\x1b":
                        return True
            elif self._fd is not None:
                import select
                r, _, _ = select.select([self._fd], [], [], 0)
                if r and os.read(self._fd, 1) == b"\x1b":
                    return True
        except Exception:
            return False
        return False

    def close(self) -> None:
        if self._fd is not None and self._old is not None:
            import termios
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
        self._fd = None


# --------------------------------------------------------------------- GUI ---

class Viewer:
    """หน้าต่าง OpenCV: HUD + pause/step ด้วยคีย์"""

    K_ESC, K_SPACE = 27, 32
    K_LEFT = (2, 81, 2424832, 65361)          # win / legacy / x11
    K_RIGHT = (3, 83, 2555904, 65363)

    def __init__(self, ring: int = 300, win: str = "h5loop") -> None:
        import cv2
        self.cv2 = cv2
        self.win = win
        self.ring = max(2, ring)
        self.buf: list[tuple[np.ndarray, dict]] = []   # (BGR, meta) เฟรมย้อนหลัง
        self.idx = 0
        self.paused = False
        self._waitKey = getattr(cv2, "waitKeyEx", cv2.waitKey)
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 960, 720)

    # ---- แปลงภาพอะไรก็ได้เป็น BGR8 ที่โชว์ได้ ----
    def _to_bgr(self, img: np.ndarray) -> np.ndarray:
        v = img
        if v.ndim == 2:
            v = np.stack([v] * 3, -1)
        elif v.ndim == 3 and v.shape[-1] == 1:
            v = np.repeat(v, 3, -1)
        elif v.ndim == 3 and v.shape[-1] == 4:
            v = v[..., :3]
        if v.dtype != np.uint8:                      # normalize ชั่วคราว
            f = v.astype(np.float32)
            s = f[::8, ::8]                          # ประเมิน min/max จากส่วนย่อย
            lo, hi = float(s.min()), float(s.max())
            v = ((f - lo) / (hi - lo + 1e-9) * 255).astype(np.uint8)
        if v.ndim == 3 and v.shape[-1] == 3:        # RGB -> BGR (cache ค่า)
            code = getattr(self, "_rgb2bgr", None)
            if code is None:
                code = self._rgb2bgr = self.cv2.COLOR_RGB2BGR
            v = self.cv2.cvtColor(v, code)
        return np.ascontiguousarray(v)

    def push(self, img: np.ndarray, meta: dict) -> None:
        self.buf.append((self._to_bgr(img), dict(meta)))
        if len(self.buf) > self.ring:                # หมุน buffer
            self.buf.pop(0)
            self.idx = max(0, self.idx - 1)          # คงตำแหน่งเฟรมที่ดูอยู่
        if not self.paused:                          # live → ตามเฟรมใหม่เสมอ
            self.idx = len(self.buf) - 1

    def _hud(self, frame: np.ndarray, m: dict) -> np.ndarray:
        cv2 = self.cv2
        out = frame.copy()
        h, w = out.shape[:2]
        cv2.rectangle(out, (0, 0), (w, 58), (30, 30, 30), -1)
        fps = m.get("_fps", 0.0)
        behind = len(self.buf) - 1 - self.idx
        state = ("LIVE" if not self.paused and behind == 0
                 else f"PAUSE  -{behind} frame")
        lines = [
            f"{m.get('file','-')}  {m.get('dataset','-')}  "
            f"[{m.get('img_idx','-')}/{m.get('n_imgs','-')}]  "
            f"cycle {m.get('cycle','-')}  seq {m.get('seq','-')}",
            f"{tuple(m.get('shape', ()))}  {m.get('dtype','-')}   "
            f"{fps:5.1f} fps   {state}",
        ]
        for i, s in enumerate(lines):
            cv2.putText(out, s, (10, 22 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (220, 220, 220), 1, cv2.LINE_AA)
        return out

    def show(self) -> bool:
        """วาดหน้าต่าง 1 เฟรม → คืน False ถ้าผู้ใช้กด ESC หรือปิดหน้าต่าง"""
        cv2 = self.cv2
        if not self.buf:
            return True

        key = self._waitKey(1)
        if key == self.K_ESC:
            return False
        if key == self.K_SPACE:
            self.paused = not self.paused
            if not self.paused:                      # เล่นต่อ → กลับไปเฟรมล่าสุด
                self.idx = len(self.buf) - 1
        elif key in self.K_LEFT and self.idx > 0:
            self.idx -= 1
            self.paused = True
        elif key in self.K_RIGHT and self.idx < len(self.buf) - 1:
            self.idx += 1
            self.paused = True

        frame, meta = self.buf[self.idx]
        cv2.imshow(self.win, self._hud(frame, meta))

        # ผู้ใช้กดปุ่ม X ปิดหน้าต่าง
        if cv2.getWindowProperty(self.win, cv2.WND_PROP_VISIBLE) < 1:
            return False
        return True

    def close(self) -> None:
        try:
            self.cv2.destroyAllWindows()
        except Exception:
            pass


class ViewerThread:
    """รัน GUI คนละ thread กับลูปรับเน็ต — imshow จะได้ไม่หน่วง throughput"""

    def __init__(self, ring: int = 300, maxlen: int = 128) -> None:
        self.q: queue.Queue = queue.Queue(maxsize=maxlen)
        self.stop = False
        self.viewer: Viewer | None = None
        self._ring = ring
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def feed(self, img: np.ndarray, meta: dict) -> None:
        try:
            self.q.put_nowait((img, meta))
        except queue.Full:                       # ทิ้งเฟรมเก่าสุด 1 ตัว
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.q.put_nowait((img, meta))
            except queue.Full:
                pass

    def _run(self) -> None:
        self.viewer = Viewer(ring=self._ring)    # สร้างหน้าต่างใน thread นี้เท่านั้น
        v = self.viewer
        while not self.stop:
            try:
                img, meta = self.q.get(timeout=0.03)
                v.push(img, meta)
            except queue.Empty:
                pass
            if not v.show():                     # วนเสมอ → ปุ่มไม่ค้างแม้ไม่มีเฟรมใหม่
                self.stop = True
        v.close()

    def close(self) -> None:
        self.stop = True
        self.th.join(timeout=2)


# ------------------------------------------------------------------- send ---

def cmd_send(args: argparse.Namespace) -> None:
    srcs = scan(args.dir, args.pattern, args.dataset)
    if not srcs:
        sys.exit(f"ไม่พบ dataset รูปภาพใน {args.dir} (pattern={args.pattern})")
    print(f"[send] พบ {len(srcs)} dataset จาก "
          f"{len({s.path for s in srcs})} ไฟล์")

    # --- socket: เป็น client (ปกติ) หรือรอรับการเชื่อมต่อ (--listen) ---
    if args.listen:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((args.host, args.port))
        srv.listen(1)
        print(f"[send] รอ receiver ที่ {args.host}:{args.port} ...")
        sock, addr = srv.accept()
        srv.close()
        print(f"[send] receiver: {addr[0]}:{addr[1]}")
    else:
        sock = socket.create_connection((args.host, args.port),
                                        timeout=args.timeout)
        print(f"[send] เชื่อมต่อ {args.host}:{args.port} แล้ว")
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SOCK_BUF)

    timer = HighResTimer()                       # Windows: จูน timer เป็น 1ms
    esc = EscListener()
    interval = max(0, args.interval_ms) / 1000.0    # 0 = ยิงเต็มที่ไม่หน่วง
    seq = 0
    sent = 0
    t0 = time.monotonic()
    next_t = t0
    cycle = 0

    try:
        with sock:
            while True:
                # ---- วนทุกไฟล์ ทุก dataset ทุกภาพ ----
                for fi, src in enumerate(srcs):
                    with h5py.File(src.path, "r") as f:
                        for i in range(src.n):
                            img = np.ascontiguousarray(grab(f, src, i))
                            meta = {
                                "seq": seq,
                                "file": os.path.basename(src.path),
                                "file_idx": fi,
                                "n_files": len(srcs),
                                "dataset": src.dset,
                                "img_idx": i,
                                "n_imgs": src.n,
                                "shape": list(img.shape),
                                "dtype": img.dtype.str,
                                "cycle": cycle,
                                "t": time.time(),
                            }
                            send_frame(sock, meta, img.tobytes())
                            seq += 1
                            sent += 1

                            # ---- คุมจังหวะแบบไม่ drift ----
                            if interval > 0:
                                next_t += interval
                                if next_t > time.monotonic():
                                    sleep_until(next_t)
                                else:
                                    next_t = time.monotonic()   # ช้าไปแล้ว → รีเซ็ต

                            if seq % max(1, args.log_every) == 0:
                                dt = time.monotonic() - t0
                                hz = sent / dt if dt else 0
                                print(f"[send] seq={seq:<8} {src.label}[{i}] "
                                      f"{tuple(img.shape)} {img.dtype} "
                                      f"{hz:5.1f} img/s")

                            if esc.pressed():
                                print("\n[send] กด ESC → หยุด")
                                return
                cycle += 1
                if args.rescan:                       # เผื่อมีไฟล์ใหม่เข้ามา
                    fresh = scan(args.dir, args.pattern, args.dataset)
                    if fresh:
                        srcs = fresh
                print(f"[send] --- จบ cycle {cycle} → กลับไปไฟล์แรก ---")
    except (KeyboardInterrupt, ConnectionError):
        print("\n[send] หยุด")
    finally:
        esc.close()
        timer.close()


# ------------------------------------------------------------------ recv ---

def cmd_recv(args: argparse.Namespace) -> None:
    vt = None
    if args.show:
        try:
            import cv2  # noqa: F401
            vt = ViewerThread(ring=args.ring)
        except ImportError:
            print("[warn] ไม่มี opencv-python — ข้ามการแสดงผล", file=sys.stderr)
    if args.save:
        os.makedirs(args.save, exist_ok=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    print(f"[recv] รอที่ {args.host}:{args.port} ...")

    conn, addr = srv.accept()
    conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCK_BUF)
    srv.close()
    print(f"[recv] sender: {addr[0]}:{addr[1]}")

    n = 0
    t0 = time.monotonic()
    try:
        with conn:
            while True:
                meta, payload = recv_frame(conn)
                img = np.frombuffer(payload,
                                    dtype=np.dtype(meta["dtype"])
                                    ).reshape(tuple(meta["shape"]))
                n += 1

                if args.save:
                    np.save(os.path.join(args.save,
                                         f"f{meta['seq']:06d}.npy"), img)

                if vt is not None:
                    if vt.stop:                  # ผู้ใช้กด ESC / ปิดหน้าต่าง
                        break
                    dt = time.monotonic() - t0
                    meta["_fps"] = n / dt if dt else 0.0
                    vt.feed(img, meta)           # ไม่บล็อก — เต็มก็ทิ้งเฟรมเก่า

                if n % max(1, args.log_every) == 0:
                    dt = time.monotonic() - t0
                    print(f"[recv] seq={meta['seq']:<8} "
                          f"{meta['file']}:{meta['dataset']}[{meta['img_idx']}] "
                          f"{tuple(img.shape)} {n / dt:5.1f} img/s")
    except (ConnectionError, KeyboardInterrupt):
        pass
    finally:
        if vt is not None:
            vt.close()
    print(f"[recv] จบ — รับทั้งหมด {n} ภาพ")


# ------------------------------------------------------------------- main ---

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("send", help="scan HDF5 ในโฟลเดอร์แล้วยิงภาพออก socket วนซ้ำ")
    s.add_argument("dir", help="โฟลเดอร์ที่มีไฟล์ .h5")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=5555)
    s.add_argument("--pattern", default="*.h5",
                   help="glob pattern ของไฟล์ (default: *.h5)")
    s.add_argument("--dataset", default=None,
                   help="ชื่อ dataset ที่จะส่ง (ไม่ใส่ = auto ทุก dataset ที่เป็นภาพ)")
    s.add_argument("--interval-ms", type=int, default=100,
                   help="หน่วงเวลาระหว่างภาพ (ms) — 0 = ยิงเต็มที่")
    s.add_argument("--listen", action="store_true",
                   help="รอ receiver เชื่อมมาหา (ปกติ sender จะเป็นฝ่ายต่อออก)")
    s.add_argument("--rescan", action="store_true",
                   help="สแกนโฟลเดอร์ใหม่ทุกครั้งที่ครบรอบ")
    s.add_argument("--log-every", type=int, default=20)
    s.add_argument("--timeout", type=float, default=30.0)
    s.set_defaults(func=cmd_send)

    r = sub.add_parser("recv", help="รับภาพจาก socket")
    r.add_argument("--host", default="0.0.0.0")
    r.add_argument("--port", type=int, default=5555)
    r.add_argument("--show", action="store_true", help="แสดงผลด้วย OpenCV")
    r.add_argument("--ring", type=int, default=300,
                   help="จำนวนเฟรมย้อนหลังที่ดูด้วยปุ่ม ←/→ ได้")
    r.add_argument("--save", default=None, help="โฟลเดอร์เก็บ .npy")
    r.add_argument("--log-every", type=int, default=20)
    r.set_defaults(func=cmd_recv)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()