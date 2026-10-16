"""qwen3-tts_studio logo v2: bold, high-contrast, crisp at small sizes.
Outputs: assets/logo.png (512) + assets/icon.ico (hand-sharpened multi-size)."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUT = Path(__file__).resolve().parent
SIZE = 512
VIOLET = (139, 92, 246)
CYAN = (34, 211, 238)
AMBER = (251, 191, 36)
BG = (10, 14, 28, 255)


def render(size: int) -> Image.Image:
    s = size / 512  # scale factor for stroke widths
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # solid rounded base + crisp bright rim
    d.rounded_rectangle([4, 4, size - 4, size - 4], radius=int(112 * s) + 4, fill=BG)
    d.rounded_rectangle([4, 4, size - 4, size - 4], radius=int(112 * s) + 4,
                        outline=(196, 181, 253, 255), width=max(2, int(4 * s)))
    # ambient glows (kept inside the tile so edges stay sharp)
    glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse([int(40 * s), int(30 * s), int(300 * s), int(290 * s)], fill=VIOLET + (55,))
    gd.ellipse([int(230 * s), int(250 * s), int(490 * s), int(500 * s)], fill=CYAN + (45,))
    img = Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(int(40 * s) + 2)))
    d = ImageDraw.Draw(img)
    # bold gradient Q with soft drop shadow for separation
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                              int(330 * s))
    cx, cy = size // 2, int(225 * s)
    sh = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(sh).text((cx + int(6 * s), cy + int(8 * s)), "Q", font=font,
                            anchor="mm", fill=(0, 0, 0, 160))
    img = Image.alpha_composite(img, sh.filter(ImageFilter.GaussianBlur(max(1, int(4 * s)))))
    d = ImageDraw.Draw(img)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).text((cx, cy), "Q", font=font, anchor="mm", fill=255)
    grad = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    gd2 = ImageDraw.Draw(grad)
    for y in range(size):
        t = y / size
        gd2.line([(0, y), (size, y)], fill=(
            int(VIOLET[0] + (CYAN[0] - VIOLET[0]) * t),
            int(VIOLET[1] + (CYAN[1] - VIOLET[1]) * t),
            int(VIOLET[2] + (CYAN[2] - VIOLET[2]) * t), 255))
    img.paste(grad, (0, 0), mask)
    d = ImageDraw.Draw(img)
    # chunky waveform bars (few, thick — stays legible tiny)
    bars = [0.30, 0.48, 0.68, 0.92, 1.0, 0.92, 0.68, 0.48, 0.34]
    maxh = int(120 * s)
    bw, gap = int(26 * s), int(12 * s)
    total = len(bars) * bw + (len(bars) - 1) * gap
    x0, base = (size - total) // 2, int(462 * s)
    for i, f in enumerate(bars):
        h = max(int(10 * s), int(maxh * f))
        x = x0 + i * (bw + gap)
        t = i / (len(bars) - 1)
        col = tuple(int(VIOLET[c] + (CYAN[c] - VIOLET[c]) * t) for c in range(3)) + (255,)
        d.rounded_rectangle([x, base - h, x + bw, base], radius=max(2, int(8 * s)), fill=col)
    # amber status dot, generous size
    r = int(22 * s)
    d.ellipse([int(408 * s) - r, int(96 * s) - r, int(408 * s) + r, int(96 * s) + r],
              fill=AMBER + (255,))
    return img


master = render(512)
master.save(OUT / "logo.png")

# ICO: Pillow's plugin builds multi-size files from one master via `sizes`
# (it ignores append_images). Pre-sharpen the master a touch so the tiny
# downscales stay crisp, then verify every frame reads back.
from PIL import ImageOps
sharp = master.filter(ImageFilter.UnsharpMask(radius=2, percent=45, threshold=1))
sharp.save(OUT / "icon.ico",
           sizes=[(16, 16), (24, 24), (32, 32), (48, 48),
                  (64, 64), (128, 128), (256, 256)])
chk = open(OUT / "icon.ico", "rb").read(6)
count = int.from_bytes(chk[4:6], "little")
assert count == 7, f"ICO directory has {count} images, want 7"
print("icon.ico images ok:", count)
for z in (64, 32):
    render(z).save(OUT / f"logo-{z}.png")
print("wrote logo.png, logo-64.png, logo-32.png, icon.ico")
