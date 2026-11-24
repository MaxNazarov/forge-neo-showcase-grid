import sys
import unittest
import tempfile
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import showcase_core as core


class CoreTests(unittest.TestCase):
    def test_panel_order(self):
        for s, mode, want in [
            ([5, -3], "By value", [5, None, -3]), ([-3, 5], "By value", [-3, None, 5]),
            ([5], "By value", [5, None]), ([-3], "By value", [None, -3]),
            ([1, 2], "By value", [None, 1, 2]), ([2, 1], "By value", [2, 1, None]),
            ([1, 1], "Off", [1]), ([1], "First", [None, 1]), ([1], "Last", [1, None])]:
            self.assertEqual(core.build_panels(s, mode), want)

    def test_parse(self):
        self.assertEqual(core.parse_strengths("5, -3"), [5, -3])
        self.assertEqual(core.parse_strengths("-2-2 (+2)"), [-2, 0, 2])
        with self.assertRaises(ValueError):
            core.parse_strengths("x")

    def test_colors(self):
        self.assertEqual(core.parse_color("rgba(255.0, 10.4, 0, 1)"), (255, 10, 0))
        self.assertEqual(core.parse_color("#fff"), (255, 255, 255))
        self.assertEqual(core.parse_color("junk"), (0, 0, 0))

    def test_stitch_size_and_gap(self):
        imgs = [Image.new("RGB", (100, 40), "red")] * 3
        s = core.Style(gap=5, border=7, grid_color="#00ff00")
        g = core.stitch(imgs, s)
        self.assertEqual(g.size, (114, 144))
        self.assertEqual(g.getpixel((50, 48)), (0, 255, 0))
        s.layout = core.HORIZONTAL
        self.assertEqual(core.stitch(imgs, s).size, (324, 54))

    def test_output_size(self):
        panels = [(s, Image.new("RGB", (1536, 512), "red")) for s in (5, None, -3)]
        for layout in core.LAYOUTS:
            self.assertEqual(core.make_showcase(panels, core.Style(layout=layout)).size,
                             (1552, 1568) if layout == core.VERTICAL else (4640, 528))
            for w, h in ((800, 1000), (800, 0), (0, 1000), (333, 777), (5000, 5000), (1552, 1568)):
                size = core.make_showcase(panels, core.Style(layout=layout, out_width=w, out_height=h)).size
                if w:
                    self.assertLessEqual(size[0], w)
                if h:
                    self.assertLessEqual(size[1], h)
                # one side comes within rounding (under 1 px per panel) of its limit
                self.assertTrue((w and w - size[0] <= 3) or (h and h - size[1] <= 3), (layout, w, h, size))
        self.assertEqual(panels[0][1].size, (1536, 512))
        self.assertEqual(core.make_showcase(panels, core.Style(out_width=1552, out_height=1568)).size, (1552, 1568))

    def test_output_too_small(self):
        with self.assertRaises(ValueError):
            core.check_output_size(3, core.Style(out_height=30))

    def test_labels(self):
        s = core.Style()
        self.assertEqual(core.label_text(5, s), "5.0")
        self.assertEqual(core.label_text(None, s), "No LoRA")
        s.decimals, s.label_template = -1, "{lora} x{strength}"
        self.assertEqual(core.label_text(0.75, s, "a"), "a x0.75")

    def test_every_position_draws_inside_and_keeps_source(self):
        src = Image.new("RGB", (300, 100), "white")
        for pos in core.POSITIONS:
            out = core.draw_label(src, "5.0", core.Style(position=pos, box_color="#000000"), {})
            self.assertIn((0, 0, 0), {c for _, c in out.getcolors(100000)}, pos)
        self.assertEqual(src.getcolors(), [(30000, (255, 255, 255))])

    def test_long_label_shrinks_to_fit(self):
        src = Image.new("RGB", (300, 100), "white")
        out = core.draw_label(src, "a very long label " * 4, core.Style(position="Middle center", margin=10), {})
        self.assertEqual({out.getpixel((x, y)) for x in range(10) for y in range(100)}, {(255, 255, 255)})

    def test_style_bad_values(self):
        s = core.make_style({"gap": "x", "border": -4, "font_size": 12.5, "position": "nowhere", "extra": 1, "box": False})
        self.assertEqual((s.gap, s.border, s.font_size, s.position, s.box), (8, 0, 12.5, "Bottom right", False))

    def test_presets(self):
        with tempfile.TemporaryDirectory() as d:
            name = core.save_preset(d, "../evil", core.Style(gap=33))
            self.assertEqual(core.list_presets(d), [name])
            self.assertEqual(core.load_preset(d, name).gap, 33)
            with self.assertRaises(ValueError):
                core.save_preset(d, " ", core.Style())


if __name__ == "__main__":
    unittest.main()
