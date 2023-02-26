import torch
import time
from functools import wraps

def profile(device='cuda', profile_memory=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = Profiler(device=device, profile_memory=profile_memory)
            with profiler:
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

class Profiler:
    def __init__(self, device='cuda', profile_memory=True):
        self.device = device
        self.profile_memory = profile_memory
        self.memory_usage = []
        
    def __enter__(self):
        if self.profile_memory:
            torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(self.device)
        self.start = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize(self.device)
        self.end = time.time()
        print(f"Total execution Time: {self.end - self.start}ms")
        
        if self.profile_memory:
            memory_stats = torch.cuda.memory_stats(self.device)
            peak_memory_usage = memory_stats['allocated_bytes.all.peak'] / 1024 ** 2
            total_memory_allocated = memory_stats['allocated_bytes.all.current'] / 1024 ** 2
            memory_cached = memory_stats['reserved_bytes.all.peak'] / 1024 ** 2
            print(f"Peak Memory Usage: {peak_memory_usage:.2f}MB")
            print(f"Total Memory Allocated: {total_memory_allocated:.2f}MB")
            print(f"Memory Cached: {memory_cached:.2f}MB")
            self.print_memory_timeline()
        
    def print_memory_timeline(self):
        if self.profile_memory:
            current_memory = torch.cuda.memory_stats(self.device)['allocated_bytes.all.current'] / 1024 ** 2
            print(f"Memory Usage: {current_memory:.2f}MB")
