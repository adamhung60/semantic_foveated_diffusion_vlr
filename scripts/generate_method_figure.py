"""Generate a minimal, clean method figure for Semantic Foveated Diffusion."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_rounded_box(ax, x, y, w, h, facecolor="#F8FAFC", edgecolor="#CBD5E1",
                     lw=1.5, radius=0.12, zorder=1):
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(box)
    return box


def draw_arrow(ax, x1, y1, x2, y2, color="#475569", lw=2.2, zorder=5):
    arrow = patches.FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>,head_length=6.5,head_width=4.2",
        color=color,
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(arrow)


def make_scene_pixels(res=64, high_detail_center=False):
    """Stylized image of a dog on a green lawn under a sky."""
    yy, xx = np.linspace(0, 1, res), np.linspace(0, 1, res)
    X, Y = np.meshgrid(xx, yy)
    img = np.zeros((res, res, 3), dtype=np.float32)

    sky_color = np.array([0.74, 0.88, 0.98])
    lawn_light = np.array([0.46, 0.78, 0.42])

    is_sky = Y > 0.52
    img[is_sky] = sky_color
    img[~is_sky] = lawn_light

    body = ((X - 0.52) / 0.17) ** 2 + ((Y - 0.42) / 0.14) ** 2 <= 1.0
    head = ((X - 0.42) / 0.12) ** 2 + ((Y - 0.58) / 0.11) ** 2 <= 1.0
    ear_l = ((X - 0.34) / 0.04) ** 2 + ((Y - 0.69) / 0.06) ** 2 <= 1.0
    ear_r = ((X - 0.49) / 0.04) ** 2 + ((Y - 0.69) / 0.06) ** 2 <= 1.0
    leg1 = (np.abs(X - 0.40) < 0.03) & (Y >= 0.26) & (Y <= 0.38)
    leg2 = (np.abs(X - 0.61) < 0.03) & (Y >= 0.26) & (Y <= 0.38)
    tail = ((X - 0.69) / 0.06) ** 2 + ((Y - 0.50) / 0.03) ** 2 <= 1.0

    fur_main = np.array([0.85, 0.60, 0.26])
    fur_dark = np.array([0.65, 0.42, 0.14])
    fur_chest = np.array([0.95, 0.84, 0.65])

    dog_mask = body | head | leg1 | leg2 | tail
    img[dog_mask] = fur_main
    img[ear_l | ear_r] = fur_dark

    if high_detail_center:
        chest = ((X - 0.43) / 0.07) ** 2 + ((Y - 0.44) / 0.08) ** 2 <= 1.0
        img[chest & dog_mask] = fur_chest
        eye_l = ((X - 0.38) ** 2 + (Y - 0.60) ** 2) <= 0.018 ** 2
        eye_r = ((X - 0.45) ** 2 + (Y - 0.60) ** 2) <= 0.018 ** 2
        nose = ((X - 0.35) ** 2 + (Y - 0.55) ** 2) <= 0.022 ** 2
        collar = (np.abs(Y - 0.50) < 0.018) & (X >= 0.35) & (X <= 0.50) & dog_mask
        img[collar] = np.array([0.85, 0.18, 0.20])
        img[eye_l | eye_r | nose] = np.array([0.12, 0.12, 0.12])

    return img


def draw_grid_image(ax, x0, y0, size, n_blocks=4, mode="draft", zorder=3):
    cell = size / n_blocks
    focus_rc = {(1, 1), (1, 2), (2, 1), (2, 2)}

    if mode == "draft":
        coarse_img = make_scene_pixels(res=12, high_detail_center=False)
        ax.imshow(
            coarse_img,
            origin="lower",
            extent=[x0, x0 + size, y0, y0 + size],
            interpolation="nearest",
            zorder=zorder,
        )
        for i in range(n_blocks + 1):
            ax.plot([x0 + i * cell, x0 + i * cell], [y0, y0 + size],
                    color="#FFFFFF", lw=1.8, zorder=zorder + 1)
            ax.plot([x0, x0 + size], [y0 + i * cell, y0 + i * cell],
                    color="#FFFFFF", lw=1.8, zorder=zorder + 1)

    elif mode == "mask":
        for r in range(n_blocks):
            for c in range(n_blocks):
                bx = x0 + c * cell
                by = y0 + r * cell
                is_hr = (r, c) in focus_rc
                fc = "#EF4444" if is_hr else "#E2E8F0"
                rect = patches.Rectangle(
                    (bx, by), cell, cell,
                    facecolor=fc, edgecolor="#FFFFFF", linewidth=1.8, zorder=zorder
                )
                ax.add_patch(rect)

    elif mode == "mixed":
        coarse_img = make_scene_pixels(res=12, high_detail_center=False)
        ax.imshow(
            coarse_img,
            origin="lower",
            extent=[x0, x0 + size, y0, y0 + size],
            interpolation="nearest",
            zorder=zorder,
        )
        for i in range(n_blocks + 1):
            ax.plot([x0 + i * cell, x0 + i * cell], [y0, y0 + size],
                    color="#FFFFFF", lw=1.6, zorder=zorder + 1)
            ax.plot([x0, x0 + size], [y0 + i * cell, y0 + i * cell],
                    color="#FFFFFF", lw=1.6, zorder=zorder + 1)
        sub = cell / 2.0
        for (r, c) in focus_rc:
            bx = x0 + c * cell
            by = y0 + r * cell
            ax.plot([bx + sub, bx + sub], [by, by + cell],
                    color="#FCA5A5", lw=1.3, zorder=zorder + 1.5)
            ax.plot([bx, bx + cell], [by + sub, by + sub],
                    color="#FCA5A5", lw=1.3, zorder=zorder + 1.5)
        hr_box = patches.Rectangle(
            (x0 + cell, y0 + cell), 2 * cell, 2 * cell,
            facecolor="none", edgecolor="#DC2626", linewidth=2.5, zorder=zorder + 2
        )
        ax.add_patch(hr_box)

    elif mode == "output":
        coarse_img = make_scene_pixels(res=12, high_detail_center=False)
        fine_img = make_scene_pixels(res=96, high_detail_center=True)
        ax.imshow(
            coarse_img,
            origin="lower",
            extent=[x0, x0 + size, y0, y0 + size],
            interpolation="bilinear",
            zorder=zorder,
        )
        ax.imshow(
            fine_img[24:72, 24:72],
            origin="lower",
            extent=[x0 + cell, x0 + 3 * cell, y0 + cell, y0 + 3 * cell],
            interpolation="bilinear",
            zorder=zorder + 0.5,
        )
        hr_box = patches.Rectangle(
            (x0 + cell, y0 + cell), 2 * cell, 2 * cell,
            facecolor="none", edgecolor="#DC2626", linewidth=2.0,
            linestyle="--", zorder=zorder + 2
        )
        ax.add_patch(hr_box)

    border = patches.Rectangle(
        (x0, y0), size, size,
        facecolor="none", edgecolor="#334155", linewidth=1.6, zorder=zorder + 3
    )
    ax.add_patch(border)


def main():
    fig, ax = plt.subplots(figsize=(13.2, 3.6), dpi=300)
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 3.6)
    ax.axis("off")

    # Stage containers
    draw_rounded_box(ax, 0.15, 0.20, 2.45, 3.25, facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=1.3)
    ax.text(1.375, 3.18, "Inputs", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#1E293B")

    draw_rounded_box(ax, 2.95, 0.20, 4.85, 3.25, facecolor="#F0F9FF", edgecolor="#93C5FD", lw=1.4)
    ax.text(5.375, 3.18, "Stage 1: Low-Res Draft & Mask",
            ha="center", va="center", fontsize=11, fontweight="bold", color="#0369A1")

    draw_rounded_box(ax, 8.15, 0.20, 4.85, 3.25, facecolor="#FFF7ED", edgecolor="#FDBA74", lw=1.4)
    ax.text(10.575, 3.18, "Stage 2: Mixed-Res Denoising",
            ha="center", va="center", fontsize=11, fontweight="bold", color="#C2410C")

    # Input cards
    draw_rounded_box(ax, 0.32, 1.78, 2.11, 1.08, facecolor="#FFFFFF", edgecolor="#64748B", lw=1.3, radius=0.08, zorder=2)
    ax.text(1.375, 2.58, "Prompt", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#0F172A", zorder=3)
    ax.text(1.375, 2.13, "“A little dog on\na green lawn”", ha="center", va="center",
            fontsize=9, style="italic", color="#334155", zorder=3)

    draw_rounded_box(ax, 0.32, 0.58, 2.11, 0.95, facecolor="#FEF2F2", edgecolor="#DC2626", lw=1.3, radius=0.08, zorder=2)
    ax.text(1.375, 1.28, "Focus", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#991B1B", zorder=3)
    ax.text(1.375, 0.88, "“the dog”", ha="center", va="center",
            fontsize=9.5, style="italic", fontweight="bold", color="#B91C1C", zorder=3)

    # 4 visual squares
    s_size = 1.80
    y_grid = 1.02
    y_mid = y_grid + s_size / 2  # 1.92
    y_label = 0.72

    x_s1 = 3.25
    draw_grid_image(ax, x_s1, y_grid, s_size, mode="draft")
    ax.text(x_s1 + s_size / 2, y_label, "1. Low-Res Draft", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#0F172A")

    x_s2 = 5.70
    draw_grid_image(ax, x_s2, y_grid, s_size, mode="mask")
    ax.text(x_s2 + s_size / 2, y_label, "2. Inferred Mask", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#0F172A")

    x_s3 = 8.45
    draw_grid_image(ax, x_s3, y_grid, s_size, mode="mixed")
    ax.text(x_s3 + s_size / 2, y_label, "3. Mixed-Res Tokens", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#0F172A")

    x_s4 = 10.90
    draw_grid_image(ax, x_s4, y_grid, s_size, mode="output")
    ax.text(x_s4 + s_size / 2, y_label, "4. Foveated Output", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#0F172A")

    # Arrows
    draw_arrow(ax, 2.45, 2.15, x_s1 - 0.05, 2.15, color="#334155")
    draw_arrow(ax, x_s1 + s_size + 0.05, y_mid, x_s2 - 0.05, y_mid, color="#0284C7")
    draw_arrow(ax, x_s2 + s_size + 0.05, y_mid, x_s3 - 0.05, y_mid, color="#475569")
    draw_arrow(ax, x_s3 + s_size + 0.05, y_mid, x_s4 - 0.05, y_mid, color="#EA580C")

    # Focus -> Inferred Mask (clean bottom path)
    x_focus = 1.375
    x_mask = x_s2 + s_size / 2
    ax.plot([x_focus, x_focus, x_mask], [0.58, 0.36, 0.36], color="#DC2626", lw=2.0, zorder=5)
    draw_arrow(ax, x_mask, 0.36, x_mask, 0.56, color="#DC2626", lw=2.0, zorder=5)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "method_pipeline.png")
    pdf_path = os.path.join(out_dir, "method_pipeline.pdf")
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.06, dpi=300)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved {png_path} and {pdf_path}")


if __name__ == "__main__":
    main()
