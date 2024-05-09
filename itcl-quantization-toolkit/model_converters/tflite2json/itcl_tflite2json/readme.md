# TFLITE2JSON

## Usage

Add to a Poetry venv

```bash
poetry add <path_to_this_package>
```

Note: This does not install the extension as editable.

### As a Python Script
This library should only expose a single function, ``convert``
```python
from itcl_tflite2json import convert

convert("model.tflite", "model.json")
```

### As a CLI Tool



## Development

Install the developer dependencies with

```bash
poetry install
```