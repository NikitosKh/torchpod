# torchpod

Launch distributed PyTorch training straight from a Jupyter notebook. One decorator, no config files, no SLURM, no bash scripts. Built for researchers and startups running bare-metal GPU clusters who just want to train. Somewhat inspired by [Monarch](https://github.com/pytorch-labs/monarch)

```
pip install torchpod
uv add torchpod
```

## Quickstart

```python
from torchpod import distributed

@distributed(gpus_per_host=8)
def train(ctx):
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, DistributedSampler

    ctx.init_distributed()
    device = torch.device(f'cuda:{ctx.local_rank}')

    model = nn.parallel.DistributedDataParallel(
        MyModel().to(device), device_ids=[ctx.local_rank]
    )
    # ... normal PyTorch training loop ...

job = train()
job.stop()
```

That's it. Your function gets serialized, shipped to every host, forked across GPUs, and wired up with NCCL. Logs stream back to your notebook in real time.

## How it actually works

The whole thing is ~350 lines across 6 files. No magic, just ZMQ and `fork()`.

### The pieces

**Controller** (`controller.py`) — lives in your notebook process. Opens two ZMQ sockets:
- A `ROUTER` socket for commands (spawn, kill, status)
- A `PULL` socket for log streaming

Both bind to ephemeral ports by default so you can run multiple jobs without collisions.

**Agent** (`agent.py`) — a lightweight daemon, one per host. Runs in the background, stays alive between jobs. Connects back to the controller over ZMQ using `DEALER` sockets and re-announces itself with `ready` every 500ms so the controller can discover it at any time. When it gets a `spawn` command, it `fork()`s one worker per GPU, pipes their stdout/stderr back through the log socket, and monitors them with `waitpid()`. When a worker dies, the agent immediately reports the exit code and signal back to the controller. Handles `SIGTERM`/`SIGINT` gracefully — kills any running workers before shutting down.

**Worker** (`worker.py`) — one per GPU. Unpickles your training function, builds a `Ctx` with rank/world_size/local_rank, calls `ctx.init_distributed()` to set up NCCL, and runs your code. That's the whole file — 22 lines.

**Cluster** (`cluster.py`) — the user-facing API. Handles launching agents (locally via `subprocess` or remotely via SSH), serializing your function with `cloudpickle`, and returning a `Job` you can wait on, query, or kill.

**Decorator** (`decorator.py`) — syntactic sugar. `@distributed(gpus_per_host=8)` just creates a `Cluster` and calls `.launch()` under the hood.

### ZMQ topology

```
Your notebook                 Host 1                    Host 2
┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│  Controller  │         │    Agent     │         │    Agent     │
│              │         │              │         │              │
│  ROUTER ◄────┼── ZMQ ──┤► DEALER      │         │  DEALER ◄────┤── ZMQ ──┐
│  (commands)  │         │              │         │              │         │
│              │         │  fork():     │         │  fork():     │         │
│  PULL ◄──────┼── ZMQ ──┤► PUSH        │         │  PUSH ◄──────┤── ZMQ ──┘
│  (logs)      │         │  (log lines) │         │  (log lines) │
│              │         │              │         │              │
│              │         │ ├─ worker 0  │         │  ├─ worker 4 │
│              │         │ ├─ worker 1  │         │  ├─ worker 5 │
│              │         │ ├─ worker 2  │         │  ├─ worker 6 │
│              │         │ └─ worker 3  │         │  └─ worker 7 │
└──────────────┘         └──────────────┘         └──────────────┘
```

The controller uses `ROUTER`/`DEALER` (not `REQ`/`REP`) because it needs to address agents by hostname and handle messages asynchronously. The log socket is a separate `PUSH`/`PULL` pair so stdout streaming doesn't block command processing.

### What happens when you call `train()`

1. `cloudpickle.dumps(fn)` serializes your function — closures, lambdas, notebook-defined classes all work
2. Controller binds two ZMQ sockets on ephemeral ports
3. Agent daemons are started on each host (locally via `subprocess.Popen`, remotely via `ssh`)
4. Agents connect to the controller and send `ready`
5. Controller sends `spawn` with the pickled code as a binary ZMQ frame
6. Each agent writes the code to disk, then `fork()`s N workers (one per GPU)
7. Workers read the pickle, build a `Ctx`, and call your function
8. Agent pipes stdout/stderr back through the `PUSH` socket — this is why `print()` in your training code shows up in the notebook
9. `waitpid()` detects worker exits, agent reports `down <rank> <pid> <exit_code> <signal>` so the controller knows immediately when something crashes

### Why fork instead of subprocess

`os.fork()` is the fastest way to spawn a process that shares the parent's memory. The agent sets up env vars (`TORCHPOD_RANK`, `TORCHPOD_LOCAL_RANK`, etc.) and then `execlp`s into `python3 -m torchpod.worker`, so each worker gets a clean Python interpreter with the right environment. `dup2` redirects stdout/stderr to pipes that the agent reads from.

### Why the code goes over ZMQ

The pickled function is sent as a binary ZMQ frame alongside the spawn command. This means no shared filesystem is needed between your notebook machine and the training hosts. The agent writes it to `/tmp/torchpod/<hostname>/code.pkl` locally.

## Job control

```python
from torchpod import Cluster

cluster = Cluster(hosts=['localhost'])
job = cluster.launch(train_fn, gpus_per_host=4)

job.wait_async()                   # non-blocking, streams logs in background
job.status()                       # {0: {'alive': True, 'host': ...}, ...}
job.logs(rank=0, tail=10)          # last 10 lines from rank 0
job.kill()                         # SIGTERM then SIGKILL
job.stop()                         # kill + cleanup agents + close ZMQ
```

## Multi-host

```python
@distributed(hosts=['node01', 'node02', 'node03', 'node04'], gpus_per_host=8)
def train(ctx):
    # ctx.world_size = 32
    # ctx.rank = 0..31
    # ctx.local_rank = 0..7
    ctx.init_distributed()
    ...
```

Requirements: SSH key auth to all hosts and `pip install torchpod` on each one. No shared filesystem.

## Failure detection

```python
def on_failure(f):
    print(f"rank {f['rank']} died with exit code {f['exit_code']}")

job = cluster.launch(train_fn, gpus_per_host=8)
job.wait(on_failure=on_failure)
```

## Dependencies

Just two: `pyzmq` and `cloudpickle`. PyTorch is needed on the training hosts but isn't a hard dependency of the package itself.


## P.S
I wrote it when playing with how zmq works. If you have any suggestions on how to imporve, consider opening a pr.
