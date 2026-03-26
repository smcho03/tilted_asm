"""
sanity_05_height_animation.py  (tilted_asm_sanity)
Gaussian h_amp: 0 → -100 um 동안 I_CMOS 애니메이션
Approach B, C, VPP 세 방식 나란히 저장
"""
import sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from me_asm        import forward_propagate_asm
from me_tilted_asm import forward_propagate_B, forward_propagate_C
from sanity_params import (wavelength, A0, t, n1, n2_complex,
                            x_coords, y_coords, x_cmos,
                            y_prime, z_prime)

N        = len(x_coords)
sigma    = 4e-3
n_frames = 20
amp_max  = -100e-6

X_tmp, Y_tmp = np.meshgrid(x_coords, y_coords)
h_amps = np.linspace(0, amp_max, n_frames)


def run(out_dir: Path):
    print("[05] Height animation  (B, C, VPP) ...", flush=True)

    kwargs = dict(wavelength=wavelength, A0=A0, t=t, n1=n1, n2_complex=n2_complex,
                  x_cmos_location=x_cmos, pad_factor=1)

    frames_vpp, frames_B, frames_C = [], [], []

    for i, amp in enumerate(h_amps):
        print(f"  frame {i+1}/{n_frames}  h_amp={amp*1e6:.1f} um", flush=True)
        h_def = amp * np.exp(-((X_tmp - x_coords[N//2])**2 + Y_tmp**2) / (2*sigma**2))
        frames_vpp.append(forward_propagate_asm(h_def, x_coords, y_coords, y_prime, z_prime, **kwargs)['I_CMOS'])
        frames_B  .append(forward_propagate_B  (h_def, x_coords, y_coords, y_prime, z_prime, **kwargs)['I_CMOS'])
        frames_C  .append(forward_propagate_C  (h_def, x_coords, y_coords, y_prime, z_prime, **kwargs)['I_CMOS'])

    all_I = frames_vpp + frames_B + frames_C
    I_max = max(I.max() for I in all_I)
    ext_cmos = [z_prime[0]*1e3, z_prime[-1]*1e3, y_prime[-1]*1e3, y_prime[0]*1e3]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Tilted ASM Sanity 05 – Height Animation', fontsize=12)

    ims = []
    for ax, lbl, frames in zip(axes, ['VPP', 'Approach B', 'Approach C'],
                                [frames_vpp, frames_B, frames_C]):
        im = ax.imshow(frames[0], extent=ext_cmos, aspect='auto', cmap='hot',
                       origin='upper', vmin=0, vmax=I_max if I_max > 0 else 1)
        ax.set_title(f'I_CMOS  [{lbl}]  h=0 um')
        ax.set_xlabel("z' [mm]"); ax.set_ylabel("y' [mm]")
        ims.append(im)

    plt.tight_layout()

    def update(fi):
        for im, ax, lbl, frames in zip(ims, axes,
                                        ['VPP', 'Approach B', 'Approach C'],
                                        [frames_vpp, frames_B, frames_C]):
            im.set_data(frames[fi])
            ax.set_title(f'I_CMOS  [{lbl}]  h={h_amps[fi]*1e6:.1f} um')
        return ims

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=200, blit=False)

    gif_path = out_dir / 'sanity_05_animation.gif'
    ani.save(str(gif_path), writer='pillow', fps=5)
    print(f"  saved -> {gif_path}", flush=True)

    try:
        mp4_path = out_dir / 'sanity_05_animation.mp4'
        ani.save(str(mp4_path), writer='ffmpeg', fps=5,
                 extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
        print(f"  saved -> {mp4_path}", flush=True)
    except Exception as e:
        print(f"  [mp4 skip] {e}", flush=True)

    plt.close(fig)


if __name__ == '__main__':
    from datetime import datetime
    run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir  = Path(__file__).parent / 'results' / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    run(out_dir)
