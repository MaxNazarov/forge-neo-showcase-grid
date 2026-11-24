"""Pure-Python core of the LoRA Showcase Grid extension (no Forge imports)."""

import json
import math
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFont

MAX_STRENGTHS = 50

BASE_OFF, BASE_VALUE, BASE_FIRST, BASE_LAST = "Off", "By value", "First", "Last"
BASELINE_MODES = [BASE_OFF, BASE_VALUE, BASE_FIRST, BASE_LAST]

VERTICAL, HORIZONTAL = "Vertical", "Horizontal"
LAYOUTS = [VERTICAL, HORIZONTAL]

UNIT_PERCENT, UNIT_PX = "% of image short side", "Pixels"
SIZE_UNITS = [UNIT_PERCENT, UNIT_PX]

POSITIONS = [f"{v} {h}" for v in ("Top", "Middle", "Bottom") for h in ("left", "center", "right")]

DEFAULT_FONT = "Default (Roboto)"
FONT_EXTENSIONS = (".ttf", ".otf", ".ttc")


# --------------------------------------------------------------------------
# Strengths, panels and prompts
# --------------------------------------------------------------------------

_NUM = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))"
RE_RANGE_STEP = re.compile(rf"{_NUM}\s*-\s*{_NUM}\s*\(\s*{_NUM}\s*\)")
RE_RANGE_COUNT = re.compile(rf"{_NUM}\s*-\s*{_NUM}\s*\[\s*(\d+)\s*\]")


def parse_strengths(text) -> list:
    """
    Parses a comma separated list of strengths. Each item is one of:
      0.7              a single value
      0.2-1.0 (+0.2)   a range with a step (end included)
      0-1 [5]          a range split into N evenly spaced values
    Raises ValueError with a readable message on bad input.
    """
    values = []
    for token in (text or "").split(","):
        token = token.strip()
        if not token:
            continue

        if m := RE_RANGE_STEP.fullmatch(token):
            start, end, step = (float(x) for x in m.groups())
            if step == 0 or (end - start) * step < 0:
                raise ValueError(f"'{token}': the step has to move from {start:g} towards {end:g}")
            count = int(math.floor((end - start) / step + 1e-9))
            values.extend(start + i * step for i in range(count + 1))
        elif m := RE_RANGE_COUNT.fullmatch(token):
            start, end, count = float(m.group(1)), float(m.group(2)), int(m.group(3))
            if count < 1:
                raise ValueError(f"'{token}': the count must be at least 1")
            if count == 1:
                values.append(start)
            else:
                values.extend(start + (end - start) * i / (count - 1) for i in range(count))
        else:
            try:
                values.append(float(token))
            except ValueError:
                raise ValueError(f"Can't read '{token}'. Use numbers like 0.5, a range with a step like 0.2-1.0 (+0.2), or a range with a count like 0-1 [5]") from None

        if len(values) > MAX_STRENGTHS:
            raise ValueError(f"Too many strengths (more than {MAX_STRENGTHS})")

    return [round(v, 4) + 0.0 for v in values]  # + 0.0 turns -0.0 into 0.0


def format_strength(value: float) -> str:
    return f"{value:g}"


def strip_lora_tags(prompt: str, names) -> str:
    """Removes <lora:name...> and <lyco:name...> tags for the given names, so the tested LoRA is not applied twice."""
    names = sorted({n for n in names if n}, key=len, reverse=True)
    if not prompt or not names:
        return prompt
    pattern = re.compile(r"<(?:lora|lyco):(?:" + "|".join(re.escape(n) for n in names) + r")(?::[^>]*)?>", re.IGNORECASE)
    stripped = pattern.sub("", prompt)
    if stripped == prompt:
        return prompt
    stripped = re.sub(r"(?<=[ \t])[ \t]+", "", stripped)  # double space left by a tag between spaces
    stripped = re.sub(r"[ \t]*,(?:[ \t]*,)+", ",", stripped)  # empty slots left by removed tags
    return stripped.strip().strip(",").strip()


def add_to_prompt(prompt: str, addition: str) -> str:
    prompt = (prompt or "").strip()
    addition = (addition or "").strip().strip(",").strip()
    if not addition:
        return prompt
    if not prompt.strip(", "):
        return addition
    return f"{prompt.rstrip(', ')}, {addition}"


def build_panels(strengths, baseline_mode) -> list:
    """Panel order without duplicates. None is the "No LoRA" panel."""
    panels = list(dict.fromkeys(strengths))
    if baseline_mode == BASE_FIRST:
        return [None] + panels
    if baseline_mode == BASE_LAST:
        return panels + [None]
    if baseline_mode == BASE_VALUE:
        # Where strength 0 belongs, in the direction the list runs.
        descending = not panels or panels[0] >= panels[-1]
        for i, s in enumerate(panels):
            if (s < 0) if descending else (s > 0):
                return panels[:i] + [None] + panels[i:]
        return panels + [None]
    return panels


def describe_count(strengths_text, baseline_mode) -> str:
    try:
        strengths = parse_strengths(strengths_text)
    except ValueError as e:
        return f"⚠️ {e}"
    if strengths:
        panels = build_panels(strengths, baseline_mode)
        order = " / ".join("No LoRA" if s is None else format_strength(s) for s in panels)
        n = len(panels)
    else:
        n = 1 if baseline_mode == BASE_OFF else 2
        order = "preferred weight" + ("" if n == 1 else " and No LoRA")
    return f"**{n} generation{'' if n == 1 else 's'}** ({order}), each × batch count × batch size."


# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------

@dataclass
class Style:
    layout: str = VERTICAL
    out_width: int = 0  # whole grid, 0 = from generation
    out_height: int = 0
    gap: int = 8
    border: int = 8
    grid_color: str = "#F2C6C6"
    show_labels: bool = True
    label_template: str = "{strength}"
    baseline_text: str = "No LoRA"
    decimals: int = 1
    position: str = "Bottom right"
    font: str = DEFAULT_FONT
    font_path: str = ""
    font_size: float = 12.0
    font_unit: str = UNIT_PERCENT
    text_color: str = "#FFD2D2"
    stroke_color: str = "#000000"
    stroke_width: int = 0
    box: bool = True
    box_color: str = "#000000"
    box_opacity: float = 100.0
    box_radius: int = 0
    padding: float = 25.0  # % of the font size
    margin: int = 0


STYLE_KEYS = [f.name for f in fields(Style)]


def make_style(data: dict) -> Style:
    """A Style from raw UI or preset values. Unknown keys and unusable values fall back to defaults."""
    style = Style()
    for f in fields(Style):
        value = (data or {}).get(f.name)
        if value is None:
            continue
        default = f.default
        try:
            if isinstance(default, bool):
                value = bool(value)
            elif isinstance(default, int):
                value = int(round(float(value)))
            elif isinstance(default, float):
                value = float(value)
            else:
                value = str(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if f.name in NUMERIC_LIMITS:
            low, high = NUMERIC_LIMITS[f.name]
            if value != value:  # NaN
                continue
            value = min(max(value, low), high)
        if f.name in CHOICES and value not in CHOICES[f.name]:
            continue
        setattr(style, f.name, value)
    return style


NUMERIC_LIMITS = {
    "out_width": (0, 16384), "out_height": (0, 16384),
    "gap": (0, 500), "border": (0, 500), "decimals": (-1, 4), "font_size": (1, 1000),
    "stroke_width": (0, 50), "box_opacity": (0, 100), "box_radius": (0, 500),
    "padding": (0, 200), "margin": (0, 1000),
}
CHOICES = {"layout": LAYOUTS, "position": POSITIONS, "font_unit": SIZE_UNITS}


def parse_color(text, default="#000000") -> tuple:
    """RGB from "#hex", a colour name, rgb(...) or the rgba(...) that Gradio's colour picker may return."""
    text = str(text or "").strip()
    m = re.fullmatch(r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*(?:,\s*[\d.]+%?\s*)?\)", text)
    try:
        if m:
            return tuple(min(255, int(round(float(v)))) for v in m.groups())
        return ImageColor.getrgb(text)[:3]
    except ValueError:
        return ImageColor.getrgb(default)[:3]


def list_fonts(*folders) -> dict:
    """File name -> path. Earlier folders win."""
    found = {}
    for folder in folders:
        folder = Path(folder)
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir(), key=lambda f: f.name.lower()):
            if path.suffix.lower() in FONT_EXTENSIONS:
                found.setdefault(path.name, str(path))
    return found


def load_font(style: Style, size: int, fonts: dict, default_path=None):
    """Custom path first, then the picked font, then the default one."""
    for path in (style.font_path.strip().strip('"'), fonts.get(style.font), default_path):
        if not path:
            continue
        try:
            return ImageFont.truetype(str(path), size)
        except OSError as e:
            print(f"[LoRA Showcase Grid] Can't load font {path}: {e}")
    return ImageFont.load_default(size)


def label_text(strength, style: Style, lora_name="") -> str:
    if strength is None:
        return style.baseline_text
    value = format_strength(strength) if style.decimals < 0 else f"{strength:.{style.decimals}f}"
    return style.label_template.replace("{strength}", value).replace("{lora}", lora_name)


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------

def draw_label(img, text, style: Style, fonts: dict, default_font=None):
    img = img.convert("RGB")
    if not text.strip():
        return img
    size = style.font_size
    if style.font_unit == UNIT_PERCENT:
        size = min(img.size) * size / 100
    size = max(1, round(size))

    measure = ImageDraw.Draw(img)

    def layout(size):
        font = load_font(style, size, fonts, default_font)
        x0, y0, x1, y1 = measure.multiline_textbbox((0, 0), text, font=font, stroke_width=style.stroke_width)
        pad = round(size * style.padding / 100)
        return font, x0, y0, math.ceil(x1 - x0) + 2 * pad, math.ceil(y1 - y0) + 2 * pad, pad

    font, x0, y0, w, h, pad = layout(size)
    available = (max(1, img.width - 2 * style.margin), max(1, img.height - 2 * style.margin))
    for _ in range(8):  # shrink labels that don't fit
        if (w <= available[0] and h <= available[1]) or size <= 1:
            break
        ratio = min(available[0] / w, available[1] / h)
        size = max(1, min(size - 1, int(size * ratio)))
        font, x0, y0, w, h, pad = layout(size)

    vertical, horizontal = style.position.lower().split()
    m = style.margin
    x = {"left": m, "center": (img.width - w) // 2, "right": img.width - w - m}[horizontal]
    y = {"top": m, "middle": (img.height - h) // 2, "bottom": img.height - h - m}[vertical]

    if style.box and style.box_opacity > 0:
        mask = Image.new("L", img.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            (x, y, x + w - 1, y + h - 1),
            radius=min(style.box_radius, w // 2, h // 2),
            fill=round(255 * style.box_opacity / 100),
        )
        img.paste(Image.new("RGB", img.size, parse_color(style.box_color)), (0, 0), mask)

    ImageDraw.Draw(img).multiline_text(
        (x + pad - x0, y + pad - y0), text, font=font, fill=parse_color(style.text_color, "#FFFFFF"),
        stroke_width=style.stroke_width, stroke_fill=parse_color(style.stroke_color), align="center",
    )
    return img


def check_output_size(count, style: Style):
    """Raises ValueError when gaps and borders leave no room for the panels."""
    gaps = style.gap * (count - 1)
    vertical = style.layout == VERTICAL
    need_w = 2 * style.border + (1 if vertical else count + gaps)
    need_h = 2 * style.border + (count + gaps if vertical else 1)
    for name, limit, need in (("width", style.out_width, need_w), ("height", style.out_height, need_h)):
        if limit and limit < need:
            raise ValueError(f"Output {name} {limit} is too small: borders and gaps for {count} panels need at least {need} px")


def fit_images(imgs, style: Style):
    """Scales all images by one factor so the whole grid, with gaps and borders, fits the output size."""
    if not imgs or not (style.out_width or style.out_height):
        return list(imgs)
    check_output_size(len(imgs), style)
    gaps = style.gap * (len(imgs) - 1)
    vertical = style.layout == VERTICAL
    if vertical:
        total_w, total_h = max(i.width for i in imgs), sum(i.height for i in imgs)
    else:
        total_w, total_h = sum(i.width for i in imgs), max(i.height for i in imgs)
    scales = []
    if style.out_width:
        scales.append((style.out_width - 2 * style.border - (0 if vertical else gaps)) / total_w)
    if style.out_height:
        scales.append((style.out_height - 2 * style.border - (gaps if vertical else 0)) / total_h)
    scale = min(scales)

    def resize(img):
        # Rounded down (1e-9 absorbs float error) so the sum never exceeds the limit.
        size = tuple(max(1, int(v * scale + 1e-9)) for v in img.size)
        return img if size == img.size else img.convert("RGB").resize(size, Image.Resampling.LANCZOS)

    return [resize(img) for img in imgs]


def stitch(imgs, style: Style):
    """Joins images in one column or row at their own size; smaller ones are centred."""
    if not imgs:
        raise ValueError("No images to join")
    gaps = style.gap * (len(imgs) - 1)
    b = style.border
    max_w, max_h = max(i.width for i in imgs), max(i.height for i in imgs)
    vertical = style.layout == VERTICAL
    if vertical:
        size = (max_w + 2 * b, sum(i.height for i in imgs) + gaps + 2 * b)
    else:
        size = (sum(i.width for i in imgs) + gaps + 2 * b, max_h + 2 * b)
    grid = Image.new("RGB", size, parse_color(style.grid_color))
    offset = b
    for img in imgs:
        if vertical:
            grid.paste(img, (b + (max_w - img.width) // 2, offset))
            offset += img.height + style.gap
        else:
            grid.paste(img, (offset, b + (max_h - img.height) // 2))
            offset += img.width + style.gap
    return grid


def make_showcase(panels, style: Style, fonts=None, default_font=None, lora_name=""):
    """panels: list of (strength or None, image). The given images are not changed."""
    imgs = []
    fitted = fit_images([img for _, img in panels], style)  # before labels, so text stays sharp
    for (strength, _), img in zip(panels, fitted):
        if style.show_labels:
            img = draw_label(img, label_text(strength, style, lora_name), style, fonts or {}, default_font)
        imgs.append(img)
    return stitch(imgs, style)


def make_preview(strengths, baseline_mode, style: Style, size, fonts=None, default_font=None, lora_name="LoRA"):
    """The showcase with plain gradients instead of generated images."""
    panels = build_panels(strengths or [1.0], baseline_mode)
    w, h = (min(4096, max(16, int(v or 0))) for v in size)
    gray = Image.linear_gradient("L").resize((w, h))
    fill = Image.merge("RGB", [gray.point(lambda v, k=k: int(50 + v * k)) for k in (0.35, 0.4, 0.5)])
    return make_showcase([(s, fill) for s in panels], style, fonts, default_font, lora_name)


# --------------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------------

def preset_path(folder, name) -> Path:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(name or "")).strip(" .")
    if not name:
        raise ValueError("Enter a preset name")
    return Path(folder) / f"{name}.json"


def list_presets(folder) -> list:
    folder = Path(folder)
    return sorted((f.stem for f in folder.glob("*.json")), key=str.lower) if folder.is_dir() else []


def save_preset(folder, name, style: Style) -> str:
    path = preset_path(folder, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(style), indent=2, ensure_ascii=False), encoding="utf-8")
    return path.stem


def load_preset(folder, name) -> Style:
    data = json.loads(preset_path(folder, name).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("The preset file is not valid")
    return make_style(data)
