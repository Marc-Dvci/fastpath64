#!/usr/bin/env python3
"""Generate the writeup figures from the committed result CSVs.

No dependencies: emits SVG directly, so figures regenerate anywhere the repo is checked out
and stay in lockstep with the numbers rather than being redrawn by hand.

    python3 bench/make_figures.py            # writes figures/*.svg

Palette is the validated categorical pair (blue #2a78d6 / orange #eb6834), which clears the
colourblind-separation and contrast gates on both surfaces. Both light and dark steps are
embedded via prefers-color-scheme so the figures stay legible in either GitHub theme.
"""

import argparse
import csv
import os
from collections import OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figures")

# --- palette (validated) -------------------------------------------------------------------
LIGHT = {
    "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7", "s1": "#2a78d6", "s2": "#eb6834",
}
DARK = {
    "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
    "grid": "#2c2c2a", "axis": "#383835", "s1": "#3987e5", "s2": "#d95926",
}

NL = chr(10)
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def style_block():
    """Literal hex on every element, dark mode as a CSS override.

    CSS custom properties are not resolved by every SVG rasteriser (librsvg renders them as
    black), and these figures have to survive GitHub, Devpost and any PDF export. Presentation
    attributes carry the light values; CSS wins over them where it is supported, so the dark
    block is a pure override.
    """
    def dk(sel, prop, key):
        return f"{sel}{{{prop}:{DARK[key]}}}"
    dark = "".join([
        dk(".bg", "fill", "surface"), dk(".ttl", "fill", "ink"), dk(".sub", "fill", "ink2"),
        dk(".lbl", "fill", "ink2"), dk(".tick", "fill", "muted"), dk(".val", "fill", "ink"),
        dk(".note", "fill", "muted"), dk(".grid", "stroke", "grid"), dk(".axis", "stroke", "axis"),
        dk(".s1", "fill", "s1"), dk(".s2", "fill", "s2"),
    ])
    return (
        "<style>"
        f".ttl{{font:600 17px {FONT}}}"
        f".sub{{font:400 12.5px {FONT}}}"
        f".lbl{{font:400 12px {FONT}}}"
        f".tick{{font:400 11px {FONT};font-variant-numeric:tabular-nums}}"
        f".val{{font:600 12px {FONT};font-variant-numeric:tabular-nums}}"
        f".note{{font:400 11.5px {FONT}}}"
        f"@media (prefers-color-scheme: dark){{{dark}}}"
        "</style>"
    )


def bar(x, y, w, h, fill, cls, r=4):
    """Bar with rounded data-end only: the baseline end stays square."""
    if h <= 0:
        return ""
    r = min(r, h, w / 2)
    return (
        f'<path d="M{x:.1f},{y + h:.1f} L{x:.1f},{y + r:.1f} '
        f'Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} L{x + w - r:.1f},{y:.1f} '
        f'Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} L{x + w:.1f},{y + h:.1f} Z" '
        f'fill="{fill}" class="{cls}"/>'
    )


def grouped_bars(path, title, subtitle, groups, series, note=None, unit="t/s", ratio_on=1):
    """groups: OrderedDict[label] -> [v_series0, v_series1]; series: [(name, palette_key), ...]

    Layout is vertically banded so nothing can collide: title/subtitle, plot, group labels,
    ratio row, legend, note. The legend sits bottom-left rather than top-right so a long title
    can never run into it.
    """
    W, H = 880, 480
    ml, mr, mt, mb = 62, 24, 78, 152
    pw, ph = W - ml - mr, H - mt - mb
    base = mt + ph

    y_glabel = base + 20       # group labels (up to 2 lines, 15px apart)
    y_ratio = base + 62        # the comparison the figure exists to make
    y_legend = base + 92
    y_note = H - 26

    vmax = (max(max(v for v in vals if v is not None) for vals in groups.values()) or 1) * 1.20

    def ypos(v):
        return base - (v / vmax) * ph

    n_g, n_s = len(groups), len(series)
    gw = pw / n_g
    bw = min(56, (gw - 28) / n_s)
    gap = 2  # 2px surface gap between adjacent bars

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'role="img" aria-label="{esc(title)}">', style_block(),
        f'<g><rect class="bg" width="{W}" height="{H}" fill="{LIGHT["surface"]}"/>',
        f'<text class="ttl" x="{ml}" y="32" fill="{LIGHT["ink"]}">{esc(title)}</text>',
        f'<text class="sub" x="{ml}" y="53" fill="{LIGHT["ink2"]}">{esc(subtitle)}</text>',
        f'<text class="tick" x="{ml}" y="{mt - 12}" fill="{LIGHT["muted"]}">{esc(unit)}</text>',
    ]

    for i in range(5):
        v = vmax * i / 4
        y = ypos(v)
        out.append(f'<line class="grid" x1="{ml}" y1="{y:.1f}" x2="{ml + pw}" y2="{y:.1f}" '
                   f'stroke="{LIGHT["grid"]}" stroke-width="1"/>')
        out.append(f'<text class="tick" x="{ml - 9}" y="{y + 4:.1f}" text-anchor="end" '
                   f'fill="{LIGHT["muted"]}">{v:.0f}</text>')
    out.append(f'<line class="axis" x1="{ml}" y1="{base}" x2="{ml + pw}" y2="{base}" '
               f'stroke="{LIGHT["axis"]}" stroke-width="1"/>')

    for gi, (glabel, vals) in enumerate(groups.items()):
        gx = ml + gi * gw
        x0 = gx + (gw - (n_s * bw + (n_s - 1) * gap)) / 2
        for si, v in enumerate(vals):
            if v is None:
                continue
            x, y = x0 + si * (bw + gap), ypos(v)
            key = series[si][1]
            out.append(bar(x, y, bw, base - y, LIGHT[key], key))
            out.append(f'<text class="val" x="{x + bw / 2:.1f}" y="{y - 8:.1f}" '
                       f'text-anchor="middle" fill="{LIGHT["ink"]}">{v:,.1f}</text>')
        for li, line in enumerate(glabel.split(NL)):
            out.append(f'<text class="lbl" x="{gx + gw / 2:.1f}" y="{y_glabel + li * 15:.1f}" '
                       f'text-anchor="middle" fill="{LIGHT["ink2"]}">{esc(line)}</text>')
        if vals[0] and vals[ratio_on]:
            out.append(f'<text class="val" x="{gx + gw / 2:.1f}" y="{y_ratio}" '
                       f'text-anchor="middle" fill="{LIGHT["ink"]}">'
                       f'{vals[ratio_on] / vals[0]:.2f}x</text>')

    lx = ml
    for name, key in series:
        out.append(f'<rect class="{key}" x="{lx}" y="{y_legend - 9}" width="10" height="10" '
                   f'rx="2" fill="{LIGHT[key]}"/>')
        out.append(f'<text class="lbl" x="{lx + 15}" y="{y_legend}" '
                   f'fill="{LIGHT["ink2"]}">{esc(name)}</text>')
        lx += 7.0 * len(name) + 40

    if note:
        for i, line in enumerate(note.split(NL)):
            out.append(f'<text class="note" x="{ml}" y="{y_note + i * 14}" '
                       f'fill="{LIGHT["muted"]}">{esc(line)}</text>')
    out.append("</g></svg>")

    os.makedirs(FIGDIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(out))
    print("wrote", os.path.relpath(path, ROOT))


def scatter(path, title, subtitle, points, xlabel, ylabel, arrows=(), refs=(), note=None):
    """Size-versus-speed plot: the two axes a format is actually chosen on.

    points: [(x, y, label, sublabel, palette_key, filled)]
    arrows: [(i_from, i_to, label)] indices into points
    refs:   [("h"|"v", value, label)] dashed reference lines
    """
    W, H = 880, 480
    ml, mr, mt, mb = 74, 200, 78, 106
    pw, ph = W - ml - mr, H - mt - mb

    xs = [p[0] for p in points] + [v for k, v, _ in refs if k == "v"]
    ys = [p[1] for p in points] + [v for k, v, _ in refs if k == "h"]
    x0, x1 = min(xs) * 0.94, max(xs) * 1.06
    y0, y1 = 0.0, max(ys) * 1.22

    def px(v):
        return ml + (v - x0) / (x1 - x0) * pw

    def py(v):
        return mt + ph - (v - y0) / (y1 - y0) * ph

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'role="img" aria-label="{esc(title)}">', style_block(),
        f'<g><rect class="bg" width="{W}" height="{H}" fill="{LIGHT["surface"]}"/>',
        f'<text class="ttl" x="{ml}" y="32" fill="{LIGHT["ink"]}">{esc(title)}</text>',
        f'<text class="sub" x="{ml}" y="53" fill="{LIGHT["ink2"]}">{esc(subtitle)}</text>',
        f'<text class="tick" x="{ml}" y="{mt - 12}" fill="{LIGHT["muted"]}">{esc(ylabel)}</text>',
    ]

    for i in range(5):
        v = y1 * i / 4
        y = py(v)
        out.append(f'<line class="grid" x1="{ml}" y1="{y:.1f}" x2="{ml + pw}" y2="{y:.1f}" '
                   f'stroke="{LIGHT["grid"]}" stroke-width="1"/>')
        out.append(f'<text class="tick" x="{ml - 9}" y="{y + 4:.1f}" text-anchor="end" '
                   f'fill="{LIGHT["muted"]}">{v:.0f}</text>')
    out.append(f'<line class="axis" x1="{ml}" y1="{py(0):.1f}" x2="{ml + pw}" y2="{py(0):.1f}" '
               f'stroke="{LIGHT["axis"]}" stroke-width="1"/>')

    # x ticks at the actual file sizes: the reader compares two files, not a continuum
    for v in sorted({round(p[0], 3) for p in points}):
        out.append(f'<text class="tick" x="{px(v):.1f}" y="{py(0) + 20:.1f}" text-anchor="middle" '
                   f'fill="{LIGHT["muted"]}">{v:.2f}</text>')
    out.append(f'<text class="lbl" x="{ml + pw / 2:.1f}" y="{py(0) + 44:.1f}" text-anchor="middle" '
               f'fill="{LIGHT["ink2"]}">{esc(xlabel)}</text>')

    # the reference lines turn the plot into quadrants: up and to the left is strictly better
    for kind, v, label in refs:
        if kind == "h":
            y = py(v)
            out.append(f'<line class="axis" x1="{ml}" y1="{y:.1f}" x2="{ml + pw}" y2="{y:.1f}" '
                       f'stroke="{LIGHT["axis"]}" stroke-width="1" stroke-dasharray="5 5"/>')
        else:
            x = px(v)
            out.append(f'<line class="axis" x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{py(0):.1f}" '
                       f'stroke="{LIGHT["axis"]}" stroke-width="1" stroke-dasharray="5 5"/>')

    for i_from, i_to, label in arrows:
        xa, ya = px(points[i_from][0]), py(points[i_from][1])
        xb, yb = px(points[i_to][0]), py(points[i_to][1])
        sign = 1 if yb > ya else -1
        out.append(f'<line class="axis" x1="{xa:.1f}" y1="{ya + sign * 15:.1f}" x2="{xb:.1f}" '
                   f'y2="{yb - sign * 15:.1f}" stroke="{LIGHT["s2"]}" stroke-width="2.5"/>')
        out.append(f'<path d="M{xb:.1f},{yb - sign * 13:.1f} l-6,{sign * 11} l12,0 Z" '
                   f'fill="{LIGHT["s2"]}" class="s2"/>')
        if label:
            out.append(f'<text class="val" x="{xa + 14:.1f}" y="{(ya + yb) / 2 + 4:.1f}" '
                       f'fill="{LIGHT["ink"]}">{esc(label)}</text>')

    for x, y, label, sub, key, filled in points:
        cx, cy = px(x), py(y)
        fill = LIGHT[key] if filled else LIGHT["surface"]
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="9" fill="{fill}" '
                   f'stroke="{LIGHT[key]}" stroke-width="3" class="{key if filled else ""}"/>')
        out.append(f'<text class="val" x="{cx + 17:.1f}" y="{cy - 2:.1f}" '
                   f'fill="{LIGHT["ink"]}">{esc(label)}</text>')
        if sub:
            out.append(f'<text class="lbl" x="{cx + 17:.1f}" y="{cy + 16:.1f}" '
                       f'fill="{LIGHT["ink2"]}">{esc(sub)}</text>')

    if note:
        for i, line in enumerate(note.split(NL)):
            out.append(f'<text class="note" x="{ml}" y="{H - 30 + i * 14}" '
                       f'fill="{LIGHT["muted"]}">{esc(line)}</text>')
    out.append("</g></svg>")

    os.makedirs(FIGDIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(out))
    print("wrote", os.path.relpath(path, ROOT))


# --- data loading --------------------------------------------------------------------------

def load(path):
    """(model, case) -> t/s, from a llama-bench CSV."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                npr, ngen = int(r.get("n_prompt") or 0), int(r.get("n_gen") or 0)
                case = f"pp{npr}" if npr and not ngen else f"tg{ngen}"
                name = (r.get("model_filename") or "").rsplit("/", 1)[-1].removesuffix(".gguf")
                out[(name, case)] = float(r["avg_ts"])
            except (KeyError, ValueError):
                continue
    return out


def load_sizes(path):
    """model -> file size in GiB, from llama-bench's own metadata columns."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                name = (r.get("model_filename") or "").rsplit("/", 1)[-1].removesuffix(".gguf")
                out[name] = int(r["model_size"]) / 2 ** 30
            except (KeyError, ValueError):
                continue
    return out


def pick(d, needle, case):
    for (name, c), v in d.items():
        if needle in name and c == case:
            return v
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab-dir", default=os.path.join(ROOT, "results", "raw-ab", "fastpath-ab"))
    ap.add_argument("--phase0-dir", default=os.path.join(ROOT, "results", "raw"))
    args = ap.parse_args()

    # ---- figure 1: the A/B result, with the untouched format as a control
    stock = load(os.path.join(args.ab_dir, "bench-stock.csv"))
    patched = load(os.path.join(args.ab_dir, "bench-patched.csv"))
    if stock and patched:
        g = OrderedDict()
        for case in ("pp512", "pp2048"):
            g[f"IQ4_XS\n{case}"] = [pick(stock, "iq4_xs", case), pick(patched, "iq4_xs", case)]
        for case in ("pp512", "pp2048"):
            g[f"Q4_K (control)\n{case}"] = [pick(stock, "q4_k", case), pick(patched, "q4_k", case)]
        g = OrderedDict((k, v) for k, v in g.items() if v[0] and v[1])
        if g:
            grouped_bars(
                os.path.join(FIGDIR, "fig1_speedup.svg"),
                "Prefill throughput, Llama-3.2-3B on Neoverse N2",
                "Both builds from one pinned commit, same CI job, same physical runner",
                g, [("stock llama.cpp", "s1"), ("FastPath64", "s2")],
                note="Q4_K is untouched by this work and holds at 1.00x, so the IQ4_XS gain is\n"
                     "attributable to the kernel rather than to run conditions.",
            )

    # ---- figure 2: the mechanism, isolated before any kernel was written
    on = load(os.path.join(args.phase0_dir, "phase0-n2-stock", "bench-n2-stock.csv"))
    off = load(os.path.join(args.phase0_dir, "phase0-n2-no-repack", "bench-n2-no-repack.csv"))
    if on and off:
        g = OrderedDict()
        for label, needle in (("Q4_K", "q4_k"), ("IQ4_XS", "iq4_xs")):
            g[f"{label}\npp512"] = [pick(off, needle, "pp512"), pick(on, needle, "pp512")]
        g = OrderedDict((k, v) for k, v in g.items() if v[0] and v[1])
        if g:
            grouped_bars(
                os.path.join(FIGDIR, "fig2_toggle.svg"),
                "What Arm's fast path was worth, before this work",
                "Same commit and runner; only GGML_CPU_REPACK differs",
                g, [("repack OFF", "s1"), ("repack ON (default)", "s2")],
                note="Disabling the fast path costs Q4_K 35% of its prefill and costs IQ4_XS 0.6%,\n"
                     "inside the run-to-run spread: IQ4_XS was never on that path to begin with.",
            )

    # ---- figure 3: the trade-off the format choice used to force, and no longer does
    sizes = load_sizes(os.path.join(args.ab_dir, "bench-stock.csv"))

    def size_of(needle):
        return next((v for k, v in sizes.items() if needle in k), None)

    sz_i, sz_q = size_of("iq4_xs"), size_of("q4_k")
    st_i, ft_i = pick(stock, "iq4_xs", "pp512"), pick(patched, "iq4_xs", "pp512")
    st_q = pick(stock, "q4_k", "pp512")
    if all(v is not None for v in (sz_i, sz_q, st_i, ft_i, st_q)):
        pts = [
            (sz_i, st_i, "IQ4_XS, stock", f"{sz_i:.2f} GiB · {st_i:.1f} t/s", "s1", False),
            (sz_q, st_q, "Q4_K_M", f"{sz_q:.2f} GiB · {st_q:.1f} t/s · untouched", "s1", True),
            (sz_i, ft_i, "IQ4_XS, FastPath64", f"{sz_i:.2f} GiB · {ft_i:.1f} t/s", "s2", True),
        ]
        scatter(
            os.path.join(FIGDIR, "fig3_tradeoff.svg"),
            "Smaller and faster: the trade-off that disappeared",
            "Llama-3.2-3B on Neoverse N2, prefill pp512, same runner",
            pts,
            xlabel="file size on disk and in RAM (GiB) — smaller is better",
            ylabel="t/s", arrows=[(0, 2, "  2.12x")],
            refs=[("h", st_q, ""), ("v", sz_q, "")],
            note=(
                f"IQ4_XS was {100 * (1 - sz_i / sz_q):.0f}% smaller and ran at "
                f"{st_i / st_q:.2f}x the prefill of Q4_K_M: fitting the instance cost throughput.\n"
                f"It is now {ft_i / st_q:.2f}x — above and to the left of the format it "
                "replaces, so the choice costs nothing."
            ),
        )


if __name__ == "__main__":
    main()
