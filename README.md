# perf
hyperfold profiler

#### setup
```
git clone https://github.com/hyperfold/perf
cd perf
pip install -r requirements.txt
pip install .
```

#### Usage 
```python3
@profile()
def f():
    a = torch.randn(1000, 1000)

# Run the fuction with profiler
f()
```
