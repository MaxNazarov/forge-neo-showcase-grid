# LoRA Showcase Grid for Forge Neo

Renders one prompt and seed at several strengths of one LoRA, optionally without it, and joins the images into one labelled strip. No header or logo is added.

## Install
**Extensions** → **Install from URL** → paste `https://github.com/MaxNazarov/forge-neo-showcase-grid` → **Install**, then restart the WebUI.

## Use
1. Choose **LoRA Showcase Grid** in the **Script** menu.
2. Pick a LoRA and type **Strengths**, e.g. `5, -3`. Ranges work too: `-2-2 (+1)`, `0-1 [5]`. Empty = preferred weight.
3. Choose where the **No LoRA** image goes: off, by value (where 0 belongs), first or last.
4. Adjust **Style**, check it with **Preview style**, then **Generate**.

Images are not cropped or resized. For wide panels set a wide size, e.g. 1536×512.

## Style
- **Grid:** vertical / horizontal, gap, border, color.
- **Text:** label template (`{strength}`, `{lora}`), No LoRA text, decimals, 9 positions, font, size (pixels or % of the image's short side), color, outline.
- **Box:** on/off, color, opacity, corner radius, padding, distance from the edge.
- **Presets:** saved in `presets/`. To delete one, delete its file.

Fonts come from `fonts/`, the system fonts folder, or a custom file path. Labels too big for the image are shrunk to fit.

## Notes
- Existing tags of the tested LoRA are removed from the prompt and hires prompt. Tags inside saved styles are not.
- Batch count / size above 1 makes one grid per image.
- After an interrupt, the grid uses finished panels only.

Parts adapted from LoRA Strength Grid, based on [Lora Queue Helper](https://github.com/Yinzo/sd-webui-Lora-queue-helper) by Yinzo (MIT).
