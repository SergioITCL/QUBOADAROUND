from itcl_quantizer.config.models.keras import QuantizerCfg, RoundingMinimaCfg
import json
import tempfile
import pytest


class TestConfigKeras:
    @pytest.fixture
    def default_cfg(self):
        return QuantizerCfg()

    def test_dump_load(self, default_cfg: QuantizerCfg):

        default_cfg_dict = default_cfg.dict()
        with tempfile.NamedTemporaryFile(suffix=".json") as f:
            print(f.name)
            print(
                "written bytes",
                f.write(bytearray(json.dumps(default_cfg_dict), "utf-8")),
            )

            f.seek(0)  # reset the file pointer

            read_cfg = json.load(f)
            assert read_cfg == default_cfg_dict

            # Try to read it with pydantic
            cfg = QuantizerCfg(**read_cfg)

            assert hasattr(cfg.ada_round_net, "t_max")

    def test_auto_determine_cfg(self, default_cfg: QuantizerCfg):
        MAX_RETRIES = 10
        default_cfg.ada_round_net = RoundingMinimaCfg(max_retries=MAX_RETRIES)

        default_cfg_dict = default_cfg.dict()
        with tempfile.NamedTemporaryFile(suffix=".json") as f:
            f.write(bytearray(json.dumps(default_cfg_dict), "utf-8"))

            f.seek(0)

            read_cfg = json.load(f)

            assert read_cfg == default_cfg
            print(read_cfg)
            cfg = QuantizerCfg(**read_cfg)

            assert cfg.ada_round_net is not None
            assert not hasattr(cfg.ada_round_net, "t_max")
            assert not hasattr(cfg.ada_round_net, "t_min")
            assert cfg.ada_round_net.max_retries == MAX_RETRIES
