"""
sanity_05_height_animation.py  (me_asm)
Gaussian 높이 h_amp 가 0 → -500 um 변하는 동안 I_CMOS 애니메이션 저장.
출력: sanity_results/sanity_05_animation.gif  (및 .mp4 가능 시)
"""
import sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

sys.path.insert(1, str(Path(__file__).parent.parent))

from me_asm       import forward_propagate_asm
from sanity_params import (wavelength, A0, t, n1, n2_complex,
                            x_coords, y_coords, x_cmos,
                            y_prime, z_prime)

N      = len(x_coords)
sigma  = 4e-3          # 4 mm sigma → max_slope = 0.607*500um/4mm = 0.076 < 0.083 (Nyquist safe)
n_frames = 30          # 프레임 수 (0 → 500 um)
amp_max  = -100e-6     # 최대 변형 깊이 [m]

X_tmp, Y_tmp = np.meshgrid(x_coords, y_coords)
h_amps = np.linspace(0, amp_max, n_frames)


def run(out_dir: Path):
    print("[05] Height animation ...", flush=True)

    # 먼저 ref 계산 (h=0)
    h_ref = np.zeros((len(y_coords), N))
    res_ref = forward_propagate_asm(
        h_ref, x_coords, y_coords, y_prime, z_prime,
        wavelength, A0, t, n1, n2_complex, x_cmos, pad_factor=2)
    I_ref = res_ref['I_CMOS']

    # 각 프레임 I_CMOS 계산
    frames_I = []
    for i, amp in enumerate(h_amps):
        print(f"  frame {i+1}/{n_frames}  h_amp={amp*1e6:.1f} um", flush=True)
        h_def = amp * np.exp(
            -((X_tmp - x_coords[N//2])**2 + Y_tmp**2) / (2*sigma**2))
        res = forward_propagate_asm(
            h_def, x_coords, y_coords, y_prime, z_prime,
            wavelength, A0, t, n1, n2_complex, x_cmos, pad_factor=2)
        frames_I.append(res['I_CMOS'])

    # I_max: 모든 프레임 공통 스케일
    I_all_max = max(I.max() for I in frames_I)

    # ── 애니메이션 생성 ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('ASM Sanity 05 – Height Animation', fontsize=12)

    ext_cmos = [z_prime[0]*1e3, z_prime[-1]*1e3,
                y_prime[-1]*1e3, y_prime[0]*1e3]

    # 왼쪽: I_ref (고정)
    axes[0].imshow(I_ref, extent=ext_cmos, aspect='auto',
                   cmap='hot', origin='upper',
                   vmin=0, vmax=I_ref.max() if I_ref.max() > 0 else 1)
    axes[0].set_title('I_CMOS  (ref, h=0)')
    axes[0].set_xlabel('z\' [mm]'); axes[0].set_ylabel('y\' [mm]')

    # 오른쪽: 변하는 프레임
    im = axes[1].imshow(frames_I[0], extent=ext_cmos, aspect='auto',
                        cmap='hot', origin='upper',
                        vmin=0, vmax=I_all_max if I_all_max > 0 else 1)
    axes[1].set_xlabel('z\' [mm]')
    title_def = axes[1].set_title(f'I_CMOS  h={h_amps[0]*1e6:.1f} um')
    plt.colorbar(im, ax=axes[1], label='Intensity')
    plt.tight_layout()

    def update(frame_idx):
        im.set_data(frames_I[frame_idx])
        title_def.set_text(f'I_CMOS  h={h_amps[frame_idx]*1e6:.1f} um')
        return im, title_def

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=150, blit=True)

    # GIF 저장 (Pillow)
    gif_path = out_dir / 'sanity_05_animation.gif'
    ani.save(str(gif_path), writer='pillow', fps=8)
    print(f"  saved → {gif_path}", flush=True)

    # MP4 저장 시도 (ffmpeg 있을 때만)
    try:
        mp4_path = out_dir / 'sanity_05_animation.mp4'
        ani.save(str(mp4_path), writer='ffmpeg', fps=8,
                 extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
        print(f"  saved → {mp4_path}", flush=True)
    except Exception as e:
        print(f"  [mp4 skip] {e}", flush=True)

    plt.close(fig)


if __name__ == '__main__':
    from datetime import datetime
    run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(__file__).parent / 'results' / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    run(out_dir)
