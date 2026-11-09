"""Package the browser game and trained policy into one offline HTML file."""

from __future__ import annotations

import json
from pathlib import Path
import re

from .export import export_model

ROOT = Path(__file__).resolve().parents[1]


def build_arena(checkpoint: str | Path, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = export_model(checkpoint, output.with_suffix(".model.json"))
    assets = ROOT / "arena" / "dist"
    # Keep the buildless static entry usable as well as the single-file artifact.
    (assets / "model.json").write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    (assets / "model.mjs").write_text((ROOT / "portable" / "model.mjs").read_text(encoding="utf-8"), encoding="utf-8")
    scripts = []
    for path in [ROOT / "portable" / "model.mjs", assets / "engine.mjs", assets / "app.mjs"]:
        source = path.read_text(encoding="utf-8")
        source = re.sub(r"^import .+?;\s*$", "", source, flags=re.MULTILINE)
        source = re.sub(r"\bexport (?=(?:const|function|class)\b)", "", source)
        scripts.append(source)
    script = "globalThis.ARENA_MODEL = " + json.dumps(payload, separators=(",", ":")) + ";\n" + "\n".join(scripts)
    script = script.replace("</script", "<\\/script")
    html = (assets / "index.html").read_text(encoding="utf-8")
    css = (assets / "style.css").read_text(encoding="utf-8")
    html = html.replace('<link rel="stylesheet" href="style.css">', "<style>" + css + "</style>")
    html = html.replace('<script type="module" src="app.mjs"></script>', '<script type="module">' + script + '</script>')
    output.write_text(html, encoding="utf-8")
    return output.resolve()


def run(args) -> None:
    output = build_arena(args.checkpoint, args.output)
    print(f"Ready: {output}\nOpen this HTML file in your browser, then choose Start match.")
