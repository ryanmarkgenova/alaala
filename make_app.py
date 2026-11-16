"""Build Alaala.app: run once from Terminal (and again after indexing new photos).

    python make_app.py

It does two things:
1. Caches small thumbnails in data/thumbs. The app reads these instead of the
   Photos library, so it doesn't need special macOS permissions.
2. Creates Alaala.app in this folder. Double-click it, or drag it to your Dock.
"""
import plistlib
import shutil
import stat
import subprocess
import sys

import pandas as pd
from tqdm import tqdm

from common import DATA, INDEX, ROOT

APP = ROOT / "Alaala.app"


def cache_thumbs(size=480):
    from PIL import Image

    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except ImportError:
        pass
    out = DATA / "thumbs"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(INDEX)
    df = df[df.image_path.notna()]
    todo = [(u, p) for u, p in zip(df.uuid, df.image_path) if not (out / f"{u}.jpg").exists()]
    failed = 0
    for uuid, path in tqdm(todo, desc="Thumbnails"):
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((size, size))
                im.save(out / f"{uuid}.jpg", "JPEG", quality=82)
        except Exception:
            failed += 1
    print(f"Thumbnails ready ({len(df) - failed:,})" + (f", {failed} unreadable" if failed else ""))


def draw_icon(png_path):
    """A warm rounded square with a simple photo glyph."""
    from PIL import Image, ImageDraw

    n = 1024
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    grad = Image.new("RGBA", (n, n))
    top, bottom = (234, 108, 52), (176, 52, 28)
    gd = ImageDraw.Draw(grad)
    for y in range(n):
        t = y / n
        gd.line([(0, y), (n, y)], fill=tuple(int(a + (b - a) * t) for a, b in zip(top, bottom)) + (255,))
    mask = Image.new("L", (n, n), 0)
    ImageDraw.Draw(mask).rounded_rectangle([90, 90, n - 90, n - 90], radius=190, fill=255)
    img.paste(grad, (0, 0), mask)
    d = ImageDraw.Draw(img)
    w = (255, 248, 240, 255)
    d.rounded_rectangle([250, 300, 774, 724], radius=56, outline=w, width=44)          # photo frame
    d.ellipse([600, 370, 690, 460], fill=(255, 214, 120, 255))                          # sun
    d.polygon([(292, 682), (452, 500), (560, 620), (620, 560), (732, 682)], fill=w)     # hills
    img.save(png_path)


def make_icns(dest):
    """Build AppIcon.icns with macOS's iconutil. Skipped quietly if unavailable."""
    if not shutil.which("iconutil"):
        return False
    from PIL import Image

    work = DATA / "AppIcon.iconset"
    work.mkdir(parents=True, exist_ok=True)
    base = DATA / "icon_1024.png"
    draw_icon(base)
    src = Image.open(base)
    for size in (16, 32, 128, 256, 512):
        src.resize((size, size), Image.LANCZOS).save(work / f"icon_{size}x{size}.png")
        src.resize((size * 2, size * 2), Image.LANCZOS).save(work / f"icon_{size}x{size}@2x.png")
    ok = subprocess.run(["iconutil", "-c", "icns", str(work), "-o", str(dest)]).returncode == 0
    shutil.rmtree(work, ignore_errors=True)
    return ok


def build_app():
    python = ROOT / ".venv" / "bin" / "python"
    if not python.exists():
        python = sys.executable  # fall back to whatever ran this script
    if APP.exists():
        shutil.rmtree(APP)
    macos = APP / "Contents" / "MacOS"
    res = APP / "Contents" / "Resources"
    macos.mkdir(parents=True)
    res.mkdir(parents=True)

    launcher = macos / "Alaala"
    launcher.write_text(f"""#!/bin/bash
cd "{ROOT}"
exec "{python}" "{ROOT / 'app.py'}" >> "{DATA / 'app.log'}" 2>&1
""")
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    info = {
        "CFBundleName": "Alaala",
        "CFBundleDisplayName": "Alaala",
        "CFBundleIdentifier": "local.alaala.app",
        "CFBundleExecutable": "Alaala",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": "0.1",
        "LSMinimumSystemVersion": "12.0",
        "NSHighResolutionCapable": True,
        "NSAppleEventsUsageDescription":
            "Alaala adds keywords to photos you choose, so the Photos search can find them.",
    }
    if make_icns(res / "AppIcon.icns"):
        info["CFBundleIconFile"] = "AppIcon"
    with open(APP / "Contents" / "Info.plist", "wb") as f:
        plistlib.dump(info, f)
    print(f"Built {APP}")


def main():
    if not INDEX.exists():
        sys.exit("Run ingest.py and embed.py first.")
    try:
        import webview  # noqa: F401
    except ImportError:
        print("Tip: `uv pip install pywebview` for a real app window (otherwise it opens in your browser).")
    cache_thumbs()
    build_app()
    print("\nDouble-click Alaala.app in Finder, or drag it to your Dock.")
    if sys.platform == "darwin":
        subprocess.run(["open", "-R", str(APP)])  # reveal it in Finder


if __name__ == "__main__":
    main()
