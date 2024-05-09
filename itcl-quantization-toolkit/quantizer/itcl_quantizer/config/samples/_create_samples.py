from itcl_quantizer.config.models.keras import QuantizerCfg
from pathlib import Path
import json
import yaml


def main():

    absolute_path = Path(__file__).parent.resolve()
    absolute_path.mkdir(parents=True, exist_ok=True)

    cfg = QuantizerCfg()

    base_cfg = cfg.dict()

    cfg.tflite()  # Updates the default values to tflite ones

    tflite_cfg = cfg.dict()

    for cfg, name in [(base_cfg, "base"), (tflite_cfg, "tflite")]:

        with open(absolute_path / f"json/{name}.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        with open(absolute_path / f"yaml/{name}.yaml", "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)


if __name__ == "__main__":
    main()
