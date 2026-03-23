import functools
from .cluster import Cluster

def distributed(hosts=None, gpus_per_host=1, port=0, master_addr=None, master_port=0):
  def decorator(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
      job = Cluster(hosts=hosts or ["localhost"], port=port).launch(fn, gpus_per_host=gpus_per_host, master_addr=master_addr, master_port=master_port)
      job.wait()
      return job
    return wrapper
  return decorator
