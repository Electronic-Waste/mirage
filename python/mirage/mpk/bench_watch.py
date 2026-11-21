"""Watch a persistent-kernel launch from the host: progress of the requests'
positions, and on a stall what the workers last did (the device-side
profiler buffer). Used by the demos' MPK_BENCH_WATCH hook."""

import os
import re
import time

import numpy as np
import torch


def _task_names():
    hdr = os.path.join(os.path.dirname(__file__), "..", "..", "..", "include",
                       "mirage", "persistent_kernel", "runtime_header.h")
    names = {}
    try:
        for m in re.finditer(r"^\s*(TASK_[A-Z0-9_]+)\s*=\s*(\d+)",
                             open(hdr).read(), re.M):
            names[int(m.group(2))] = m.group(1)[5:]
    except OSError:
        pass
    return names


class LaunchWatch:
    """Create before the launch (a CUDA stream cannot be created while the
    persistent kernel runs); call wait() after it."""

    def __init__(self, step: torch.Tensor, profiler: torch.Tensor = None,
                 num_workers: int = 128):
        self.step, self.profiler, self.num_workers = step, profiler, num_workers
        self.side = torch.cuda.Stream()
        self.names = _task_names()

    def _last_entries(self):
        buf = self.profiler.to("cpu").numpy()[1:]
        n = (len(buf) // self.num_workers) * self.num_workers
        buf = buf[:n].reshape(-1, self.num_workers)
        cnt = (buf != 0).sum(0)
        last = buf[np.maximum(cnt - 1, 0), np.arange(self.num_workers)]
        tag = (last & np.uint64(0xffffffff)).astype(np.int64)
        return buf, cnt, tag & 3, (tag >> 2) & 0x1ff

    def wait(self, done: torch.cuda.Event, stall_s: float, gap_s: float = 2.0):
        """Poll until `done`; False if the positions stop moving for stall_s."""
        t0, prev, t_prev = time.perf_counter(), None, time.perf_counter()
        while not done.query():
            time.sleep(gap_s)
            with torch.cuda.stream(self.side):
                st = self.step.to("cpu")
            cur = (int(st.min()), int(st.max()), int(st.sum()))
            now = time.perf_counter()
            if cur != prev:
                prev, t_prev = cur, now
                print(f"WATCH t={now - t0:7.1f}s positions {cur[0]} .. {cur[1]}",
                      flush=True)
            elif now - t_prev > stall_s:
                print(f"WATCH t={now - t0:7.1f}s no progress for "
                      f"{now - t_prev:.0f} s at positions {cur[0]} .. {cur[1]}",
                      flush=True)
                if self.profiler is not None:
                    with torch.cuda.stream(self.side):
                        buf, cnt, etype, eidx = self._last_entries()
                    hist = {}
                    for w in range(self.num_workers):
                        key = ("no entry" if cnt[w] == 0 else "%s %s" % (
                            ("in  ", "done", "inst")[int(etype[w])],
                            self.names.get(int(eidx[w]), int(eidx[w]))))
                        hist.setdefault(key, []).append(w)
                    for key, ws in sorted(hist.items(), key=lambda kv: -len(kv[1])):
                        print(f"WATCH   {len(ws):3d} workers last: {key:44s} "
                              f"e.g. {ws[:6]}", flush=True)
                    tags = (buf[buf != 0] & np.uint64(0xffffffff)).astype(np.int64)
                    et, ei = tags & 3, (tags >> 2) & 0x1ff
                    for t in np.unique(ei):
                        nb = int(((ei == t) & (et == 0)).sum())
                        ne = int(((ei == t) & (et == 1)).sum())
                        print(f"WATCH   {str(self.names.get(int(t), int(t))):36s} "
                              f"begun {nb:8d} ended {ne:8d} open {nb - ne}",
                              flush=True)
                    dump = os.environ.get("MPK_BENCH_PROFILE")
                    if dump:
                        np.save(dump, buf[:int(cnt.max()) + 1])
                return False
        return True
