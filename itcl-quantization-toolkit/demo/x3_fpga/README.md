# X3 regression demo

### How to run

1 - Install the `demo` venv with poetry:

```bash
poetry install
```

2 - Run the demo with the venv

```bash
poetry run python demo/x3_fpga/quantize_x3.py cpu small
poetry run python demo/x3_fpga/quantize_x3.py fpga small
poetry run python demo/x3_fpga/quantize_x3.py cpu large
```
