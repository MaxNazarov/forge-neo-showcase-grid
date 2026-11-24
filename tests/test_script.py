"""Runs the script against stand-ins for Forge and Gradio."""
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


class ScriptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        loras = Path(self.tmp.name)
        (loras / "size.safetensors").write_bytes(b"")
        (loras / "size.json").write_text('{"activation text": "trig", "preferred weight": 0.8}')
        self.state = types.SimpleNamespace(interrupted=False, skipped=False, stopping_generation=False)
        self.calls, self.interrupt_at = [], None

        def process(p):
            self.calls.append(p)
            if len(self.calls) == self.interrupt_at:
                self.state.interrupted = True
            n = p.batch_size * p.n_iter
            return types.SimpleNamespace(images=[Image.new("RGB", (90, 30))] * n, infotexts=["i"] * n, info="i",
                                         all_prompts=[p.prompt] * n, all_seeds=[p.seed] * n, seed=p.seed, index_of_first_image=0)

        self.images = MagicMock()
        base = type("Base", (), {"elem_id": lambda self, x: x})
        self.gr = MagicMock()
        m = lambda name, **k: types.SimpleNamespace(__name__=name, **k)
        mods = {
            "gradio": self.gr,
            "modules": m("modules", images=self.images, scripts=types.SimpleNamespace(Script=base), sd_models=MagicMock()),
            "modules.paths_internal": m("p", roboto_ttf_file=None),
            "modules.processing": m("pr", Processed=lambda p, imgs, seed=None, info="", **k: types.SimpleNamespace(images=imgs, info=info, **k),
                                    fix_seed=lambda p: setattr(p, "seed", 42), process_images=process),
            "modules.shared": m("s", cmd_opts=types.SimpleNamespace(lora_dir=str(loras)), state=self.state,
                                opts=types.SimpleNamespace(lora_preferred_name="Filename", grid_format="png")),
        }
        patcher = patch.dict(sys.modules, mods)
        patcher.start()
        self.addCleanup(patcher.stop)
        spec = importlib.util.spec_from_file_location("showcase_script", ROOT / "scripts" / "lora_showcase_grid.py")
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        self.p = types.SimpleNamespace(prompt="cat <lora:size:1>", hr_prompt="", seed=-1, batch_size=1, n_iter=1,
                                       extra_generation_params={}, outpath_grids="g")

    def run_script(self, strengths="5, -3", mode="By value", lora="size", activation=True, **style_kwargs):
        style = self.mod.core.Style(**style_kwargs)
        values = [getattr(style, k) for k in self.mod.core.STYLE_KEYS]
        return self.mod.Script().run(self.p, False, "", ["/"], lora, strengths, mode, activation, True, *values)

    def test_prompts_seed_and_grid(self):
        out = self.run_script()
        self.assertEqual([c.prompt for c in self.calls], ["cat, <lora:size:5>, trig", "cat", "cat, <lora:size:-3>, trig"])
        self.assertEqual({c.seed for c in self.calls}, {42})
        self.assertEqual(len(out.images), 4)
        self.assertEqual(out.images[0].size, (106, 122))
        self.images.save_image.assert_called_once()

    def test_empty_strengths_use_preferred(self):
        self.run_script(strengths="", mode="Off")
        self.assertIn("<lora:size:0.8>", self.calls[0].prompt)

    def test_hires_prompt(self):
        self.p.hr_prompt = "hr"
        self.run_script(activation=False)
        self.assertEqual([c.hr_prompt for c in self.calls], ["hr, <lora:size:5>", "hr", "hr, <lora:size:-3>"])

    def test_batches_make_one_grid_each(self):
        self.p.batch_size = 2
        out = self.run_script()
        self.assertEqual((out.index_of_first_image, len(out.images), len(out.infotexts)), (2, 8, 8))

    def test_interrupt_drops_unfinished(self):
        self.interrupt_at = 3
        out = self.run_script()
        self.assertEqual(len(out.images), 3)
        self.assertEqual(out.images[0].height, 30 * 2 + 8 * 3)

    def test_output_size_keeps_single_images(self):
        out = self.run_script(out_width=61)
        self.assertEqual(out.images[0].size, (61, 8 * 4 + 15 * 3))
        self.assertEqual({i.size for i in out.images[1:]}, {(90, 30)})

    def test_too_small_output_fails_before_generating(self):
        with self.assertRaises(ValueError):
            self.run_script(out_height=20)
        self.assertEqual(self.calls, [])

    def test_no_lora_generates_normally(self):
        self.run_script(lora=None)
        self.assertEqual(len(self.calls), 1)

    def test_ui_returns_run_inputs(self):
        self.assertEqual(len(self.mod.Script().ui(False)), 8 + len(self.mod.core.STYLE_KEYS))


if __name__ == "__main__":
    unittest.main()
