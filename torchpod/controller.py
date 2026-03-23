import zmq, time

class Controller:
  def __init__(self, port=0):
    self.ctx = zmq.Context()
    self.cmd = self.ctx.socket(zmq.ROUTER)
    self.cmd.setsockopt(zmq.LINGER, 0)
    self.logs = self.ctx.socket(zmq.PULL)
    self.logs.setsockopt(zmq.LINGER, 0)
    if port == 0:
      self.port = self.cmd.bind_to_random_port("tcp://*")
      self.log_port = self.logs.bind_to_random_port("tcp://*")
    else:
      self.port, self.log_port = port, port + 1
      self.cmd.bind(f"tcp://*:{port}")
      self.logs.bind(f"tcp://*:{port + 1}")
    self.agents, self.workers, self._log_lines, self._on_log = {}, {}, [], None

  def _send(self, host, msg):
    self.cmd.send(host.encode(), zmq.SNDMORE); self.cmd.send(b"", zmq.SNDMORE); self.cmd.send_string(msg)

  def _send_with_payload(self, host, msg, payload):
    self.cmd.send(host.encode(), zmq.SNDMORE); self.cmd.send(b"", zmq.SNDMORE)
    self.cmd.send_string(msg, zmq.SNDMORE); self.cmd.send(payload)

  def _recv(self):
    try: identity = self.cmd.recv(zmq.NOBLOCK); self.cmd.recv(); return identity.decode(), self.cmd.recv_string()
    except zmq.Again: return None

  def _recv_log(self):
    try: return self.logs.recv_string(zmq.NOBLOCK)
    except zmq.Again: return None

  def _process(self, host, data):
    p = data.split(); kind = p[0]
    if kind == "ready": self.agents[host] = True
    elif kind == "up": self.workers[int(p[1])] = {"alive": True, "host": host, "pid": int(p[2]), "code": -1, "sig": 0}
    elif kind == "down":
      rank, pid, code, sig = int(p[1]), int(p[2]), int(p[3]), int(p[4])
      self.workers[rank] = {"alive": False, "host": host, "pid": pid, "code": code, "sig": sig}
      return {"rank": rank, "pid": pid, "exit_code": code, "signal": sig, "host": host}
    return None

  def wait_for_agents(self, hostnames, timeout=30):
    expected, deadline = set(hostnames), time.time() + timeout
    while expected and time.time() < deadline:
      msg = self._recv()
      if msg: self._process(msg[0], msg[1]); expected.discard(msg[0])
      time.sleep(0.01)
    if expected: raise TimeoutError(f"agents not ready: {expected}")

  def request_status(self, hostnames, timeout=10):
    self.workers.clear()
    expected, deadline = set(hostnames), time.time() + timeout
    while expected and time.time() < deadline:
      msg = self._recv()
      if msg: self._process(msg[0], msg[1]); expected.discard(msg[0])
      else: time.sleep(0.05)
    for host in self.agents: self._send(host, "status")
    end, got = time.time() + 1.5, False
    while time.time() < end:
      msg = self._recv()
      if msg: self._process(msg[0], msg[1]); got = got or bool(self.workers)
      elif got: break
      else: time.sleep(0.05)

  def spawn(self, code_bytes, gpus_per_host, master_addr="localhost", master_port=0):
    hosts = list(self.agents.keys())
    world_size = len(hosts) * gpus_per_host
    for i, host in enumerate(hosts):
      self._send_with_payload(host, f"spawn {gpus_per_host} {i * gpus_per_host} {world_size} {master_addr} {master_port}", code_bytes)
    deadline = time.time() + 30
    while len(self.workers) < world_size and time.time() < deadline:
      msg = self._recv()
      if msg: self._process(msg[0], msg[1])
      time.sleep(0.01)
    if len(self.workers) < world_size: raise RuntimeError(f"only {len(self.workers)}/{world_size} workers started")

  def kill_all(self):
    for host in self.agents: self._send(host, "kill")
    self.workers.clear()

  def poll(self):
    failures = []
    while (msg := self._recv()) is not None:
      f = self._process(msg[0], msg[1])
      if f: failures.append(f)
    while (line := self._recv_log()) is not None:
      self._log_lines.append(line)
      if self._on_log: self._on_log(line)
    return failures

  def get_logs(self, rank=None, tail=None):
    lines = [l for l in self._log_lines if l.startswith(f"[{rank}:")] if rank is not None else self._log_lines
    return lines[-tail:] if tail else lines

  def close(self): self.cmd.close(); self.logs.close(); self.ctx.term()
