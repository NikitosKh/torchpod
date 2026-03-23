import zmq, os, sys, signal, time, fcntl

running = True
def _stop(sig, frame): global running; running = False

def _nonblock(fd): fcntl.fcntl(fd, fcntl.F_SETFL, fcntl.fcntl(fd, fcntl.F_GETFL) | os.O_NONBLOCK)

class Worker:
  __slots__ = ["pid", "rank", "outfd", "errfd", "log_out", "log_err", "alive", "code", "sig"]
  def __init__(self, pid, rank, outfd, errfd, log_out, log_err):
    self.pid, self.rank, self.outfd, self.errfd, self.log_out, self.log_err = pid, rank, outfd, errfd, log_out, log_err
    self.alive, self.code, self.sig = True, -1, 0
  def close_fds(self):
    for fd in [self.outfd, self.errfd, self.log_out, self.log_err]:
      if fd >= 0:
        try: os.close(fd)
        except OSError: pass
    self.outfd = self.errfd = self.log_out = self.log_err = -1

def spawn_one(rank, local_rank, world_size, code_path, master_addr, master_port, logdir):
  pout_r, pout_w = os.pipe()
  perr_r, perr_w = os.pipe()
  pid = os.fork()
  if pid == 0:
    os.dup2(pout_w, 1); os.close(pout_r); os.close(pout_w)
    os.dup2(perr_w, 2); os.close(perr_r); os.close(perr_w)
    os.environ.update({"TORCHPOD_RANK": str(rank), "TORCHPOD_LOCAL_RANK": str(local_rank), "TORCHPOD_WORLD_SIZE": str(world_size),
                        "TORCHPOD_CODE": code_path, "TORCHPOD_MASTER_ADDR": master_addr, "TORCHPOD_MASTER_PORT": str(master_port)})
    os.execlp("python3", "python3", "-m", "torchpod.worker")
  os.close(pout_w); os.close(perr_w)
  _nonblock(pout_r); _nonblock(perr_r)
  log_out = os.open(f"{logdir}/rank_{rank}.out", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
  log_err = os.open(f"{logdir}/rank_{rank}.err", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
  return Worker(pid, rank, pout_r, perr_r, log_out, log_err)

class Agent:
  def __init__(self, cmd_addr, log_addr, hostname):
    self.host, self.logdir, self.workers = hostname, f"/tmp/torchpod/{hostname}", []
    os.makedirs(self.logdir, exist_ok=True)
    self.ctx = zmq.Context()
    self.cmd = self.ctx.socket(zmq.DEALER)
    self.cmd.setsockopt_string(zmq.IDENTITY, hostname)
    self.cmd.setsockopt(zmq.RECONNECT_IVL, 200)
    self.cmd.setsockopt(zmq.RECONNECT_IVL_MAX, 2000)
    self.cmd.connect(cmd_addr)
    self.logs = self.ctx.socket(zmq.PUSH)
    self.logs.setsockopt(zmq.SNDHWM, 1000)
    self.logs.setsockopt(zmq.RECONNECT_IVL, 200)
    self.logs.connect(log_addr)

  def _send(self, msg): self.cmd.send(b"", zmq.SNDMORE); self.cmd.send_string(msg)

  def _recv(self):
    try: self.cmd.recv(zmq.NOBLOCK)
    except zmq.Again: return None, None
    cmd = self.cmd.recv_string()
    payload = self.cmd.recv() if self.cmd.getsockopt(zmq.RCVMORE) else None
    return cmd, payload

  def _log(self, line):
    try: self.logs.send_string(line, zmq.NOBLOCK)
    except zmq.ZMQError: pass

  def do_spawn(self, p, code_bytes):
    n, base, ws, addr, port = int(p[1]), int(p[2]), int(p[3]), p[4], p[5]
    code_path = os.path.join(self.logdir, "code.pkl")
    with open(code_path, "wb") as f: f.write(code_bytes)
    for i in range(n):
      w = spawn_one(base + i, i, ws, code_path, addr, port, self.logdir)
      self.workers.append(w)
      self._send(f"up {w.rank} {w.pid}")

  def do_kill(self):
    for w in self.workers:
      if not w.alive: w.close_fds(); continue
      os.kill(w.pid, signal.SIGTERM); time.sleep(0.1)
      try:
        if os.waitpid(w.pid, os.WNOHANG)[0] == 0: os.kill(w.pid, signal.SIGKILL)
      except OSError: pass
      try: os.waitpid(w.pid, 0)
      except OSError: pass
      w.alive = False; w.close_fds()
    self.workers.clear()

  def do_status(self):
    for w in self.workers:
      self._send(f"up {w.rank} {w.pid}" if w.alive else f"down {w.rank} {w.pid} {w.code} {w.sig}")

  def check_workers(self):
    for w in self.workers:
      if not w.alive: continue
      try: pid, st = os.waitpid(w.pid, os.WNOHANG)
      except ChildProcessError: continue
      if pid == 0: continue
      w.alive, w.code, w.sig = False, (os.WEXITSTATUS(st) if os.WIFEXITED(st) else -1), (os.WTERMSIG(st) if os.WIFSIGNALED(st) else 0)
      self._send(f"down {w.rank} {w.pid} {w.code} {w.sig}")
      w.close_fds()

  def drain_logs(self):
    for w in self.workers:
      for fd, logfd, s in [(w.outfd, w.log_out, "out"), (w.errfd, w.log_err, "err")]:
        if fd < 0: continue
        try: data = os.read(fd, 4096)
        except OSError: continue
        if not data: continue
        if logfd >= 0:
          try: os.write(logfd, data)
          except OSError: pass
        self._log(f"[{w.rank}:{s}] " + data.decode(errors="replace"))

  def run(self):
    self._send(f"ready {self.host}")
    last_ready = time.monotonic()
    while running:
      poller = zmq.Poller(); poller.register(self.cmd, zmq.POLLIN); socks = dict(poller.poll(50))
      if self.cmd in socks:
        while True:
          cmd, payload = self._recv()
          if cmd is None: break
          p = cmd.split()
          if   p[0] == "spawn":  self.do_spawn(p, payload)
          elif p[0] == "kill":   self.do_kill()
          elif p[0] == "status": self.do_status()
          elif p[0] == "ping":   self._send("pong")
      if time.monotonic() - last_ready >= 0.5: self._send(f"ready {self.host}"); last_ready = time.monotonic()
      self.check_workers(); self.drain_logs()
    self.do_kill()

if __name__ == "__main__":
  signal.signal(signal.SIGINT, _stop); signal.signal(signal.SIGTERM, _stop); signal.signal(signal.SIGPIPE, signal.SIG_IGN)
  Agent(sys.argv[1], sys.argv[2], sys.argv[3]).run()
