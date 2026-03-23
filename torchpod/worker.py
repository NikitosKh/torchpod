import os, sys, pickle, inspect, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="runpy")

class Ctx:
  def __init__(self):
    self.rank, self.local_rank = int(os.environ.get("TORCHPOD_RANK", 0)), int(os.environ.get("TORCHPOD_LOCAL_RANK", 0))
    self.world_size = int(os.environ.get("TORCHPOD_WORLD_SIZE", 1))
    self.master_addr, self.master_port = os.environ.get("TORCHPOD_MASTER_ADDR", "localhost"), int(os.environ.get("TORCHPOD_MASTER_PORT", 0))

  def init_distributed(self, backend="nccl"):
    import torch, torch.distributed as dist
    torch.cuda.set_device(self.local_rank)
    os.environ.update({"MASTER_ADDR": self.master_addr, "MASTER_PORT": str(self.master_port),
                        "RANK": str(self.rank), "WORLD_SIZE": str(self.world_size), "LOCAL_RANK": str(self.local_rank)})
    dist.init_process_group(backend)

def main():
  sys.stdout.reconfigure(line_buffering=True); sys.stderr.reconfigure(line_buffering=True)
  with open(os.environ["TORCHPOD_CODE"], "rb") as f: fn = pickle.load(f)
  fn(Ctx()) if len(inspect.signature(fn).parameters) > 0 else fn()

if __name__ == "__main__": main()
