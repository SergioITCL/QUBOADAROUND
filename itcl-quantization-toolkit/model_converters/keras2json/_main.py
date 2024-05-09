import json
from tensorflow import keras
from keras2json import convert
from pathlib import Path


def _main():

    for model_path in Path("models").glob("*.h5"):

        out_path = str(model_path).replace(".h5", ".json")

        model = keras.models.load_model(model_path, compile=False)

        network = convert(model)

        with open(out_path, "w") as f:
            json.dump(
                network,
                f,
            )

        print(f"Generated model at {out_path}")


if __name__ == "__main__":
    _main()
