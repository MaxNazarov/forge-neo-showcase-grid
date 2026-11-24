"""
LoRA Showcase Grid for Forge Neo.

Renders the same prompt and seed at several strengths of one LoRA (and optionally
without it) and joins the results into one labelled image.
LoRA helpers are adapted from the LoRA Strength Grid and LoRA Slider Dataset scripts.
"""
import copy
import json
import os
import sys
from pathlib import Path

import gradio as gr

from modules import images, scripts, sd_models
from modules.paths_internal import roboto_ttf_file
from modules.processing import Processed, fix_seed, process_images
from modules.shared import cmd_opts, opts, state

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import showcase_core as core  # noqa: E402

LOG_PREFIX = "[LoRA Showcase Grid]"
LORA_EXTENSIONS = (".safetensors", ".pt", ".ckpt")
STRENGTHS_HELP = "Panel order = typed order. Empty = preferred weight. Examples: 5, -3  |  -2-2 (+1)  |  0-1 [5]"

PRESET_DIR = ROOT / "presets"
FONT_DIRS = [
    ROOT / "fonts",
    Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows" / "Fonts",
    Path("/usr/share/fonts/truetype"),
    Path("/Library/Fonts"),
]

lora_dir = Path(cmd_opts.lora_dir).resolve()


# --------------------------------------------------------------------------
# LoRA folders and files
# --------------------------------------------------------------------------

def allowed_path(path) -> bool:
    try:
        return Path(path).resolve().is_relative_to(lora_dir)
    except (OSError, ValueError):
        return False


def get_base_path(is_use_custom_path, custom_path) -> Path:
    return lora_dir.joinpath(custom_path) if is_use_custom_path and custom_path else lora_dir


def is_lora_file(filename: str) -> bool:
    return filename.lower().endswith(LORA_EXTENSIONS)


def is_directory_contain_lora(path) -> bool:
    try:
        if allowed_path(path):
            return any(is_lora_file(f) for f in os.listdir(path))
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"{LOG_PREFIX} {e}")
    return False


def get_directories(base_path, include_root=True) -> list:
    directories = ["/"] if include_root else []
    try:
        if allowed_path(base_path):
            for entry in sorted(os.listdir(base_path), key=str.lower):
                full_path = os.path.join(base_path, entry)
                if os.path.isdir(full_path):
                    if is_directory_contain_lora(full_path):
                        directories.append(entry)
                    nested = get_directories(full_path, include_root=False)
                    directories.extend(os.path.join(entry, d) for d in nested)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"{LOG_PREFIX} {e}")
    return directories


def list_lora_files(base_path, directories) -> list:
    """All LoRA files in the chosen directories, in a stable order, without duplicates."""
    files, seen = [], set()
    base_path = Path(base_path)
    for directory in directories or []:
        directory = base_path if directory == "/" else base_path.joinpath(directory)
        if not allowed_path(directory) or not directory.is_dir():
            continue
        for filename in sorted(os.listdir(directory), key=str.lower):
            full = directory.joinpath(filename)
            if is_lora_file(filename) and full.is_file() and full not in seen:
                seen.add(full)
                files.append(full)
    return files


def list_lora_names(base_path, directories) -> list:
    names = []
    for path in list_lora_files(base_path, directories):
        if path.stem not in names:
            names.append(path.stem)
    return names


def get_lora_name(path: Path) -> str:
    """The name to put inside <lora:...>, resolved the same way Forge's LoRA extension does."""
    try:
        import networks  # Forge's built-in LoRA extension

        entry = networks.available_networks.get(path.stem)
        if entry is not None and Path(entry.filename).resolve() == path.resolve():
            return entry.get_alias()
    except Exception:
        pass

    if opts.lora_preferred_name == "Filename" or path.suffix.lower() != ".safetensors":
        return path.stem
    try:
        metadata = sd_models.read_metadata_from_safetensors(str(path))
    except Exception:
        return path.stem
    name = metadata.get("ss_output_name")
    # Some training tools write Python's None as the string "None".
    if not name or name == "None":
        return path.stem
    return name


def load_sidecar(path: Path) -> tuple:
    """(preferred weight, activation text) from the LoRA's .json settings file."""
    weight, activation = 1.0, ""
    json_path = path.with_suffix(".json")
    if json_path.is_file():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            activation = str(data.get("activation text", "") or "").strip().strip(",").strip()
            weight = float(data.get("preferred weight", 1) or 1)  # 0 means "not set"
        except Exception as e:
            print(f"{LOG_PREFIX} Could not read {json_path.name}: {e}")
    return weight, activation


def get_fonts() -> dict:
    return core.list_fonts(*FONT_DIRS)


# --------------------------------------------------------------------------
# Script
# --------------------------------------------------------------------------

class Script(scripts.Script):
    def title(self):
        return "LoRA Showcase Grid"

    def ui(self, is_img2img):
        def show_dir_textbox(enabled, custom_path):
            dirs = get_directories(get_base_path(enabled, custom_path))
            return gr.update(visible=enabled), gr.update(choices=dirs, value=[])

        def update_dirs(is_use_custom_path, custom_path):
            return gr.update(choices=get_directories(get_base_path(is_use_custom_path, custom_path)), value=[])

        def select_all_dirs(is_use_custom_path, custom_path):
            return gr.update(value=get_directories(get_base_path(is_use_custom_path, custom_path)))

        def update_loras(current, is_use_custom_path, custom_path, directories):
            all_loras = list_lora_names(get_base_path(is_use_custom_path, custom_path), directories)
            return gr.update(choices=all_loras, value=current if current in all_loras else None)

        def eid(name):
            return self.elem_id(name)

        d = core.Style()
        with gr.Column():
            base_dir_checkbox = gr.Checkbox(label="Use custom LoRA path", value=False, elem_id=eid("base_dir_checkbox"))
            base_dir_textbox = gr.Textbox(label="LoRA directory", placeholder="Path relative to the LoRA directory (set with --lora-dir at startup)", visible=False, elem_id=eid("base_dir_textbox"))
            directory_checkboxes = gr.CheckboxGroup(label="Select directory", choices=get_directories(lora_dir), value=["/"], elem_id=eid("directory_checkboxes"))
            with gr.Row():
                select_all_dirs_button = gr.Button("All")
                deselect_all_dirs_button = gr.Button("Clear")
            lora_radio = gr.Radio(label="LoRA", choices=list_lora_names(lora_dir, ["/"]), value=None, elem_id=eid("lora_radio"))

            strengths_box = gr.Textbox(label="Strengths", value="", placeholder=STRENGTHS_HELP, info=STRENGTHS_HELP, elem_id=eid("strengths"))
            baseline_mode = gr.Radio(label="No LoRA image", choices=core.BASELINE_MODES, value=core.BASE_VALUE, elem_id=eid("baseline_mode"))
            job_info = gr.Markdown(core.describe_count("", core.BASE_VALUE))
            with gr.Row():
                include_activation = gr.Checkbox(label="Add activation text", value=True, elem_id=eid("include_activation"))
                save_grid = gr.Checkbox(label="Save grid image", value=True, elem_id=eid("save_grid"))

            c = {}
            with gr.Accordion("Style", open=True):
                with gr.Row():
                    preset_select = gr.Dropdown(label="Preset", choices=core.list_presets(PRESET_DIR), value=None, elem_id=eid("preset_select"))
                    preset_load = gr.Button("Load")
                    preset_name = gr.Textbox(label="Save as", elem_id=eid("preset_name"))
                    preset_save = gr.Button("Save")
                preset_info = gr.Markdown("")

                gr.Markdown("**Grid**")
                with gr.Row():
                    c["layout"] = gr.Radio(label="Layout", choices=core.LAYOUTS, value=d.layout, elem_id=eid("layout"))
                    c["gap"] = gr.Number(label="Gap (px)", value=d.gap, precision=0, elem_id=eid("gap"))
                    c["border"] = gr.Number(label="Border (px)", value=d.border, precision=0, elem_id=eid("border"))
                    c["grid_color"] = gr.ColorPicker(label="Gap and border color", value=d.grid_color, elem_id=eid("grid_color"))

                with gr.Row():
                    c["out_width"] = gr.Number(label="Output grid width (0 = from generation)", value=d.out_width, precision=0, minimum=0, elem_id=eid("out_width"))
                    c["out_height"] = gr.Number(label="Output grid height (0 = from generation)", value=d.out_height, precision=0, minimum=0, elem_id=eid("out_height"))

                gr.Markdown("**Label text**")
                with gr.Row():
                    c["show_labels"] = gr.Checkbox(label="Show labels", value=d.show_labels, elem_id=eid("show_labels"))
                    c["label_template"] = gr.Textbox(label="Label ({strength}, {lora})", value=d.label_template, elem_id=eid("label_template"))
                    c["baseline_text"] = gr.Textbox(label="No LoRA label", value=d.baseline_text, elem_id=eid("baseline_text"))
                    c["decimals"] = gr.Number(label="Decimals (-1 = as needed)", value=d.decimals, precision=0, elem_id=eid("decimals"))
                c["position"] = gr.Dropdown(label="Position", choices=core.POSITIONS, value=d.position, elem_id=eid("position"))
                with gr.Row():
                    c["font"] = gr.Dropdown(label="Font", choices=[core.DEFAULT_FONT, *get_fonts()], value=d.font, allow_custom_value=True, elem_id=eid("font"))
                    c["font_path"] = gr.Textbox(label="Custom font file (overrides Font)", value=d.font_path, placeholder="Full path to a .ttf / .otf file", elem_id=eid("font_path"))
                with gr.Row():
                    c["font_size"] = gr.Number(label="Font size", value=d.font_size, elem_id=eid("font_size"))
                    c["font_unit"] = gr.Radio(label="Font size unit", choices=core.SIZE_UNITS, value=d.font_unit, elem_id=eid("font_unit"))
                with gr.Row():
                    c["text_color"] = gr.ColorPicker(label="Text color", value=d.text_color, elem_id=eid("text_color"))
                    c["stroke_color"] = gr.ColorPicker(label="Outline color", value=d.stroke_color, elem_id=eid("stroke_color"))
                    c["stroke_width"] = gr.Number(label="Outline width (px)", value=d.stroke_width, precision=0, elem_id=eid("stroke_width"))

                gr.Markdown("**Label box**")
                with gr.Row():
                    c["box"] = gr.Checkbox(label="Show box", value=d.box, elem_id=eid("box"))
                    c["box_color"] = gr.ColorPicker(label="Box color", value=d.box_color, elem_id=eid("box_color"))
                    c["box_opacity"] = gr.Slider(label="Box opacity (%)", minimum=0, maximum=100, step=1, value=d.box_opacity, elem_id=eid("box_opacity"))
                with gr.Row():
                    c["box_radius"] = gr.Number(label="Corner radius (px)", value=d.box_radius, precision=0, elem_id=eid("box_radius"))
                    c["padding"] = gr.Number(label="Padding (% of font size)", value=d.padding, elem_id=eid("padding"))
                    c["margin"] = gr.Number(label="Distance from edge (px)", value=d.margin, precision=0, elem_id=eid("margin"))

                with gr.Row():
                    preview_width = gr.Number(label="Preview panel width", value=1536, precision=0, elem_id=eid("preview_width"))
                    preview_height = gr.Number(label="Preview panel height", value=512, precision=0, elem_id=eid("preview_height"))
                    preview_button = gr.Button("Preview style")
                preview_image = gr.Image(label="Style preview", type="pil", interactive=False)

            style_inputs = [c[key] for key in core.STYLE_KEYS]

            def style_from(values):
                return core.make_style(dict(zip(core.STYLE_KEYS, values)))

            def on_preview(strengths_text, mode, lora, width, height, *values):
                try:
                    strengths = core.parse_strengths(strengths_text)
                    return core.make_preview(strengths, mode, style_from(values), (width, height), get_fonts(), roboto_ttf_file, lora or "LoRA")
                except ValueError as e:
                    raise gr.Error(str(e))

            def on_save(name, *values):
                try:
                    saved = core.save_preset(PRESET_DIR, name, style_from(values))
                except (OSError, ValueError) as e:
                    return gr.update(), f"⚠️ {e}"
                return gr.update(choices=core.list_presets(PRESET_DIR), value=saved), f"Saved preset **{saved}**."

            def on_load(name):
                try:
                    style = core.load_preset(PRESET_DIR, name)
                except (OSError, ValueError) as e:
                    return [f"⚠️ {e}"] + [gr.update() for _ in core.STYLE_KEYS]
                return [f"Loaded preset **{name}**."] + [gr.update(value=getattr(style, key)) for key in core.STYLE_KEYS]

            base_dir_checkbox.change(fn=show_dir_textbox, inputs=[base_dir_checkbox, base_dir_textbox], outputs=[base_dir_textbox, directory_checkboxes])
            base_dir_textbox.change(fn=update_dirs, inputs=[base_dir_checkbox, base_dir_textbox], outputs=[directory_checkboxes])
            directory_checkboxes.change(fn=update_loras, inputs=[lora_radio, base_dir_checkbox, base_dir_textbox, directory_checkboxes], outputs=[lora_radio])
            select_all_dirs_button.click(fn=select_all_dirs, inputs=[base_dir_checkbox, base_dir_textbox], outputs=directory_checkboxes)
            deselect_all_dirs_button.click(fn=lambda: gr.update(value=[]), inputs=None, outputs=directory_checkboxes)
            for component in (strengths_box, baseline_mode):
                component.change(fn=core.describe_count, inputs=[strengths_box, baseline_mode], outputs=job_info, show_progress=False)
            preview_button.click(fn=on_preview, inputs=[strengths_box, baseline_mode, lora_radio, preview_width, preview_height, *style_inputs], outputs=preview_image)
            preset_save.click(fn=on_save, inputs=[preset_name, *style_inputs], outputs=[preset_select, preset_info])
            preset_load.click(fn=on_load, inputs=[preset_select], outputs=[preset_info, *style_inputs])

        return [
            base_dir_checkbox, base_dir_textbox, directory_checkboxes, lora_radio,
            strengths_box, baseline_mode, include_activation, save_grid, *style_inputs,
        ]

    def run(self, p, is_use_custom_path, custom_path, directories, selected_lora,
            strengths_text, baseline_mode, include_activation, save_grid, *style_values):
        if not selected_lora:
            print(f"{LOG_PREFIX} No LoRA selected; generating normally.")
            return process_images(p)

        base_path = get_base_path(is_use_custom_path, custom_path)
        path = next((f for f in list_lora_files(base_path, directories) if f.stem == selected_lora), None)
        if path is None:
            print(f"{LOG_PREFIX} LoRA '{selected_lora}' was not found; generating normally.")
            return process_images(p)

        style = core.make_style(dict(zip(core.STYLE_KEYS, style_values)))
        tag_name = get_lora_name(path)
        preferred_weight, activation = load_sidecar(path)
        strengths = core.parse_strengths(strengths_text) or [preferred_weight]
        panels = core.build_panels(strengths, baseline_mode)
        if len(panels) > 1:
            core.check_output_size(len(panels), style)  # fail before generating, not after

        # The tested LoRA must not also be applied by a tag already in the prompt.
        names = {path.stem, tag_name}
        base_prompt = core.strip_lora_tags(p.prompt, names)
        base_hr_prompt = core.strip_lora_tags(getattr(p, "hr_prompt", "") or "", names)

        fix_seed(p)  # every panel uses the same seed
        p.do_not_save_grid = True
        state.job_count = len(panels) * p.n_iter

        results = []  # (strength, [(image, infotext, prompt, seed), ...])
        for number, strength in enumerate(panels):
            if state.interrupted or getattr(state, "stopping_generation", False):
                break
            name = "No LoRA" if strength is None else f"{path.stem} @ {core.format_strength(strength)}"
            state.job = f"{number + 1} out of {len(panels)}: {name}"

            cp = copy.copy(p)
            cp.extra_generation_params = dict(p.extra_generation_params)
            cp.extra_generation_params["Showcase grid"] = name
            cp.prompt, hr_prompt = base_prompt, base_hr_prompt
            if strength is not None:
                addition = f"<lora:{tag_name}:{core.format_strength(strength)}>"
                if include_activation and activation:
                    addition += f", {activation}"
                cp.prompt = core.add_to_prompt(base_prompt, addition)
                if base_hr_prompt:
                    hr_prompt = core.add_to_prompt(base_hr_prompt, addition)
            if base_hr_prompt:
                cp.hr_prompt = hr_prompt

            proc = process_images(cp)
            if state.interrupted or state.skipped:
                continue  # an unfinished panel doesn't belong in the grid

            start = proc.index_of_first_image
            cell = []
            for i, img in enumerate(proc.images[start:start + p.batch_size * p.n_iter]):
                infotext = proc.infotexts[start + i] if start + i < len(proc.infotexts) else proc.info
                prompt = proc.all_prompts[i] if i < len(proc.all_prompts) else cp.prompt
                seed = proc.all_seeds[i] if i < len(proc.all_seeds) else proc.seed
                cell.append((img, infotext, prompt, seed))
            if cell:
                results.append((strength, cell))

        if not results:
            return Processed(p, [], p.seed, "")
        if len(results) < len(panels):
            print(f"{LOG_PREFIX} {len(results)} of {len(panels)} panels finished.")

        fonts = get_fonts()
        grid_info = "\n".join([
            f"Prompt: {base_prompt or 'Empty'}",
            f"LoRA: {path.stem}",
            f"Strengths: {', '.join('No LoRA' if s is None else core.format_strength(s) for s, _ in results)}",
            f"Seed: {p.seed}",
        ])
        grids = []
        if len(results) > 1:
            for k in range(max(len(cell) for _, cell in results)):
                column = [(s, cell[k][0]) for s, cell in results if k < len(cell)]
                if len(column) < 2:
                    continue
                grid = core.make_showcase(column, style, fonts, roboto_ttf_file, path.stem)
                grids.append(grid)
                if save_grid:
                    images.save_image(grid, p.outpath_grids, "lora_showcase", seed=p.seed, prompt=base_prompt, extension=opts.grid_format, info=grid_info, grid=True, p=p)

        flat = [item for _, cell in results for item in cell]
        n = len(grids)
        return Processed(
            p, grids + [f[0] for f in flat], seed=p.seed, info=flat[0][1],
            all_prompts=[base_prompt] * n + [f[2] for f in flat],
            all_seeds=[p.seed] * n + [f[3] for f in flat],
            infotexts=[grid_info] * n + [f[1] for f in flat],
            index_of_first_image=n,
        )
