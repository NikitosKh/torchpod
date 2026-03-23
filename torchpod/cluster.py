import os, sys, socket, threading, subprocess, time
import cloudpickle as pickle
from .controller import Controller

def _free_port():
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("", 0)); return s.getsockname()[1]

class Job:
  def __init__(self, ctrl, agents=None, hosts=None):
    self._ctrl, self._agents, self._hosts = ctrl, agents or [], hosts or []
    self._done, self._thread = threading.Event(), None

  def _loop(self, on_log, on_failure):
    self._ctrl._on_log = on_log
    while not self._done.is_set():
      try: failures = self._ctrl.poll()
      except Exception: break
      for f in failures:
        if on_failure: on_failure(f)
        elif on_log: on_log(f"!! rank {f['rank']} died (code={f['exit_code']} sig={f['signal']})")
      if self._ctrl.workers and all(not w["alive"] for w in self._ctrl.workers.values()): self._done.set(); break
      time.sleep(0.05)

  def wait(self, on_log=print, on_failure=None): self._loop(on_log, on_failure); return self
  def wait_async(self, on_log=print, on_failure=None):
    self._thread = threading.Thread(target=self._loop, args=(on_log, on_failure), daemon=True); self._thread.start(); return self
  def logs(self, rank=None, tail=20): self._ctrl.poll(); return self._ctrl.get_logs(rank=rank, tail=tail)
  def status(self): self._ctrl.poll(); return {r: {"alive": w["alive"], "host": w["host"], "pid": w["pid"]} for r, w in self._ctrl.workers.items()}
  def kill(self): self._ctrl.kill_all(); self._done.set()
  def stop(self):
    self.kill()
    for p in self._agents:
      p.terminate()
      try: p.wait(timeout=5)
      except Exception: pass
    self._ctrl.close()
  @property
  def done(self): return self._done.is_set()
  def log_dir(self, host=None): return f"/tmp/torchpod/{host}/" if host else {h: f"/tmp/torchpod/{h}/" for h in self._hosts}

def _start_agent(host, cmd_addr, log_addr):
  if host in ("localhost", "127.0.0.1"):
    return subprocess.Popen([sys.executable, "-m", "torchpod.agent", cmd_addr, log_addr, host],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", host,
                   f"python3 -m torchpod.agent {cmd_addr} {log_addr} {host} &"],
                  capture_output=True, timeout=10)
  return None

class Cluster:
  def __init__(self, hosts=None, port=0): self.hosts, self.port = hosts or ["localhost"], port
  def _addr(self, master=None): return master or ("127.0.0.1" if self.hosts == ["localhost"] else self.hosts[0])

  def _ensure_agents(self, ctrl, addr):
    cmd_addr, log_addr = f"tcp://{addr}:{ctrl.port}", f"tcp://{addr}:{ctrl.log_port}"
    procs = [p for h in self.hosts if (p := _start_agent(h, cmd_addr, log_addr)) is not None]
    ctrl.wait_for_agents(self.hosts, timeout=15)
    return procs

  def launch(self, fn, gpus_per_host=1, master_addr=None, master_port=0, on_log=print):
    code_bytes = pickle.dumps(fn)
    addr, ctrl = self._addr(master_addr), Controller(port=self.port)
    procs = self._ensure_agents(ctrl, addr)
    ctrl.spawn(code_bytes, gpus_per_host, addr, master_port or _free_port())
    if on_log: on_log(f"[torchpod] {len(ctrl.workers)} workers launched across {len(self.hosts)} hosts")
    return Job(ctrl, procs, self.hosts)

  def attach(self, on_log=print):
    ctrl = Controller(port=self.port)
    ctrl.request_status(self.hosts, timeout=10)
    alive, total = sum(1 for w in ctrl.workers.values() if w["alive"]), len(ctrl.workers)
    if on_log: on_log(f"[torchpod] reattached: {alive}/{total} workers alive" if total else "[torchpod] no workers running")
    return Job(ctrl, [], self.hosts)

  def shutdown(self, on_log=print):
    for host in self.hosts:
      if host in ("localhost", "127.0.0.1"): subprocess.run(["pkill", "-f", "torchpod.agent"], capture_output=True)
      else: subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", host, "pkill -f torchpod.agent"], capture_output=True, timeout=10)
    if on_log: on_log(f"[torchpod] agents stopped on {len(self.hosts)} hosts")

  def exec(self, command):
    results = {}
    for host in self.hosts:
      r = subprocess.run(command, shell=True, capture_output=True, text=True) if host in ("localhost", "127.0.0.1") \
          else subprocess.run(["ssh", host, command], capture_output=True, text=True)
      results[host] = r.stdout + r.stderr
    return results
