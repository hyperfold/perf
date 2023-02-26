# perf
hyperfold profiler

#### Usage 
```python3
@profile()
def f():
    a = torch.randn(1000, 1000)

# Run the fuction with profiler
f()
```
