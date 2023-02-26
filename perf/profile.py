import torch
import time
from functools import wraps

def profile(device='cuda', profile_memory=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = CustomProfiler(device=device, profile_memory=profile_memory)
            with profiler:
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

class CustomProfiler:
    def __init__(self, device='cuda', profile_memory=True):
        self.device = device
        self.profile_memory = profile_memory
        self.events = []
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
        
        # Iterate over all CUDAevents
        for i, event in enumerate(self.events):
            start_time = event[0].elapsed_time(event[1])
            print(f"Event {i+1}: {start_time}ms")
        
        # Print total time
        print(f"Total Time: {self.end - self.start}ms")
        
        # Optionally print memory usage
        if self.profile_memory:
            memory_stats = torch.cuda.memory_stats(self.device)
            peak_memory_usage = memory_stats['allocated_bytes.all.peak'] / 1024 ** 2
            total_memory_allocated = memory_stats['allocated_bytes.all.current'] / 1024 ** 2
            memory_cached = memory_stats['reserved_bytes.all.peak'] / 1024 ** 2
            print(f"Peak Memory Usage: {peak_memory_usage:.2f}MB")
            print(f"Total Memory Allocated: {total_memory_allocated:.2f}MB")
            print(f"Memory Cached: {memory_cached:.2f}MB")
            self.print_memory_timeline()
        
    def add_event(self, name):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.events.append((event, name))
        
    def record(self, name):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.events[-1] = (self.events[-1][0], event, self.events[-1][1])
        self.add_event(name)
        
    def exit(self):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.events[-1] = (self.events[-1][0], event, self.events[-1][1])
        
    def print_memory_timeline(self):
        memory_stats = torch.cuda.memory_stats(self.device)
        if self.profile_memory:
            self.memory_usage.append((memory_stats['allocated_bytes.all.current'] / 1024 ** 2, time.time()))
            if 'event' in memory_stats:
                memory_events = memory_stats['event'].items()
                for i, (event_name, event_dict) in enumerate(memory_events):
                    event_size = event_dict['allocated_bytes.all.current'] / 1024 ** 2
                    event_time = event_dict['timestamp']
                    self.memory_usage.append((event_size, event_time))
                    print(f"Memory Event {i+1}: {event_size:.2f}MB, {event_name}, Time (s): {event_time:.4f}")
            else:
                current_memory = memory_stats['allocated_bytes.all.current'] / 1024 ** 2
                self.memory_usage.append((current_memory, time.time()))
                print(f"Memory Usage: {current_memory:.2f}MB")
