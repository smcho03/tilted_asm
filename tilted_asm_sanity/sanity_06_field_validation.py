"""
sanity_06_field_validation.py  (tilted_asm_sanity)
복소장 검증 — 해석 해 vs 시뮬레이션 비교

Test 1: 에너지 보존 (모든 방식)
  - E_CMOS / E_ref for B, C, VPP

Test 2: 위상 일관성 — 평면 거울, 평면파 입사
  - 반사 후 CMOS 위상은 k0*x_cmos + const 에 가깝게 일정해야 함
  - 실제: Gaussian aperture 때문에 가장자리로 갈수록 위상 변화
  - 중심선 (y'=0) 위상 분포: 세 방식 비교

Test 3: 강도 프로파일 비교
  - 평면 거울: 세 방식의 중심 z'선 (y'=0) 강도 단면 비교
  - 변형 거울 (Gaussian bump): 마찬가지

Test 4: 방식 간 차이 정량화
  - 평면 거울, 변형 거울 모두에서:
    rms(I_B - I_VPP) / mean(I_VPP)
    rms(I_C - I_VPP) / mean(I_VPP)
    rms(I_B - I_C)   / mean(I_VPP)
"""
import sys, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from me_asm        import forward_propagate_asm
from me_tilted_asm import forward_propagate_B, forward_propagate_C
from sanity_params import (wavelength, A0, t, n1, n2_complex,
                            x_coords, y_coords, x_cmos,
                            y_prime, z_prime, pixel_size)

STYLE = {
    'figure.facecolor': '#0d1117', 'axes.facecolor': '#0d1117',
    'text.color': '#e6edf3', 'axes.labelcolor': '#e6edf3',
    'xtick.color': '#8b949e', 'ytick.color': '#8b949e',
    'axes.titlesize': 9, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'font.family': 'monospace',
}

dx   = float(x_coords[1] - x_coords[0])
dy   = float(y_coords[1] - y_coords[0])
dy_c = float(y_prime[1]  - y_prime[0])
dz_c = float(z_prime[1]  - z_prime[0])

N = len(x_coords)
X_tmp, Y_tmp = np.meshgrid(x_coords, y_coords)
h_ref = np.zeros((len(y_coords), N))

# Gaussian deformation (Nyquist safe)
h_amp  = -100e-6
sigma  = 4e-3
h_def  = h_amp * np.exp(-((X_tmp - x_coords[N//2])**2 + Y_tmp**2) / (2*sigma**2))


def _energy(res):
    E_in   = float(np.sum(np.abs(res['U_in'])**2))  * dx * dy
    E_ref  = float(np.sum(np.abs(res['U_ref'])**2)) * dx * dy
    E_cmos = float(np.sum(res['I_CMOS'])) * dy_c * dz_c
    return E_in, E_ref, E_cmos


def run(out_dir: Path):
    print("[06] Field validation ...", flush=True)

    kwargs = dict(wavelength=wavelength, A0=A0, t=t, n1=n1, n2_complex=n2_complex,
                  x_cmos_location=x_cmos, pad_factor=2)

    # ── 계산 ────────────────────────────────────────────────────────────────
    print("  computing flat mirror fields ...", flush=True)
    r_vpp_flat = forward_propagate_asm(h_ref, x_coords, y_coords, y_prime, z_prime, **kwargs)
    r_B_flat   = forward_propagate_B  (h_ref, x_coords, y_coords, y_prime, z_prime, **kwargs)
    r_C_flat   = forward_propagate_C  (h_ref, x_coords, y_coords, y_prime, z_prime, **kwargs)

    print("  computing deformed mirror fields ...", flush=True)
    r_vpp_def  = forward_propagate_asm(h_def, x_coords, y_coords, y_prime, z_prime, **kwargs)
    r_B_def    = forward_propagate_B  (h_def, x_coords, y_coords, y_prime, z_prime, **kwargs)
    r_C_def    = forward_propagate_C  (h_def, x_coords, y_coords, y_prime, z_prime, **kwargs)

    # ── Test 1: 에너지 보존 ─────────────────────────────────────────────────
    print("  [Test 1] Energy conservation:", flush=True)
    for tag, res_f, res_d in [
        ('VPP', r_vpp_flat, r_vpp_def),
        ('B',   r_B_flat,   r_B_def),
        ('C',   r_C_flat,   r_C_def),
    ]:
        Ei, Er, Ec_f = _energy(res_f)
        _, _, Ec_d   = _energy(res_d)
        print(f"    [{tag}] flat: E_CMOS/E_ref = {Ec_f/Er*100:.2f}%  "
              f"deformed: E_CMOS/E_ref = {Ec_d/Er*100:.2f}%")

    # ── Test 4: 방식 간 차이 정량화 ─────────────────────────────────────────
    print("  [Test 4] RMS differences (normalized to mean(I_VPP)):", flush=True)
    for label, r_vpp, r_B, r_C in [
        ('flat',    r_vpp_flat, r_B_flat, r_C_flat),
        ('deformed',r_vpp_def,  r_B_def,  r_C_def),
    ]:
        I_v = r_vpp['I_CMOS']
        I_b = r_B['I_CMOS']
        I_c = r_C['I_CMOS']
        mu  = I_v.mean() + 1e-30
        rms = lambda a, b: float(np.sqrt(np.mean((a-b)**2)))
        print(f"    [{label}]  rms(B-VPP)/mu={rms(I_b,I_v)/mu*100:.2f}%  "
              f"rms(C-VPP)/mu={rms(I_c,I_v)/mu*100:.2f}%  "
              f"rms(B-C)/mu={rms(I_b,I_c)/mu*100:.2f}%")

    # ── 시각화: 중심 단면 비교 ───────────────────────────────────────────────
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Sanity 06 – Field Validation: Cross-section Comparison', fontsize=11,
                 color='#e6edf3')
    fig.patch.set_facecolor('#0d1117')

    # y'=0 인덱스 (중심 행)
    iy_cmos = len(y_prime) // 2
    # z'=z_center 인덱스 (중심 열)
    iz_center = np.argmin(np.abs(z_prime - float((z_prime[0]+z_prime[-1])/2)))

    colors = {'VPP': '#58a6ff', 'B': '#3fb950', 'C': '#f78166'}

    # ax[0,0]: flat mirror — z' 단면 (y'=0)
    ax = axes[0, 0]
    for tag, res in [('VPP', r_vpp_flat), ('B', r_B_flat), ('C', r_C_flat)]:
        ax.plot(z_prime * 1e3, res['I_CMOS'][:, iy_cmos], label=tag, color=colors[tag])
    ax.set_title("Flat mirror: I_CMOS along z'  (y'=0)")
    ax.set_xlabel("z' [mm]"); ax.set_ylabel('Intensity [a.u.]')
    ax.legend(fontsize=8); ax.grid(True, color='#30363d')

    # ax[0,1]: flat mirror — y' 단면 (z'=z_center)
    ax = axes[0, 1]
    for tag, res in [('VPP', r_vpp_flat), ('B', r_B_flat), ('C', r_C_flat)]:
        ax.plot(y_prime * 1e3, res['I_CMOS'][iz_center, :], label=tag, color=colors[tag])
    ax.set_title("Flat mirror: I_CMOS along y'  (z'=z_center)")
    ax.set_xlabel("y' [mm]"); ax.set_ylabel('Intensity [a.u.]')
    ax.legend(fontsize=8); ax.grid(True, color='#30363d')

    # ax[1,0]: deformed — z' 단면 (y'=0)
    ax = axes[1, 0]
    for tag, res in [('VPP', r_vpp_def), ('B', r_B_def), ('C', r_C_def)]:
        ax.plot(z_prime * 1e3, res['I_CMOS'][:, iy_cmos], label=tag, color=colors[tag])
    ax.set_title("Gaussian h=-100um: I_CMOS along z'  (y'=0)")
    ax.set_xlabel("z' [mm]"); ax.set_ylabel('Intensity [a.u.]')
    ax.legend(fontsize=8); ax.grid(True, color='#30363d')

    # ax[1,1]: deformed — y' 단면 (z'=z_center)
    ax = axes[1, 1]
    for tag, res in [('VPP', r_vpp_def), ('B', r_B_def), ('C', r_C_def)]:
        ax.plot(y_prime * 1e3, res['I_CMOS'][iz_center, :], label=tag, color=colors[tag])
    ax.set_title("Gaussian h=-100um: I_CMOS along y'  (z'=z_center)")
    ax.set_xlabel("y' [mm]"); ax.set_ylabel('Intensity [a.u.]')
    ax.legend(fontsize=8); ax.grid(True, color='#30363d')

    plt.tight_layout()
    plt.savefig(out_dir / 'sanity_06_field_validation.png',
                dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  saved -> sanity_06_field_validation.png")

    # ── 위상 단면 비교 (flat mirror) ─────────────────────────────────────────
    from skimage.restoration import unwrap_phase
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    fig2.suptitle('Sanity 06 – Phase Cross-section  (Flat Mirror)', fontsize=11,
                  color='#e6edf3')
    fig2.patch.set_facecolor('#0d1117')

    ax = axes2[0]
    for tag, res in [('VPP', r_vpp_flat), ('B', r_B_flat), ('C', r_C_flat)]:
        phi_1d = unwrap_phase(np.angle(res['U_CMOS'][:, iy_cmos]))
        ax.plot(z_prime * 1e3, phi_1d, label=tag, color=colors[tag])
    ax.set_title("Phase along z'  (flat mirror, y'=0)")
    ax.set_xlabel("z' [mm]"); ax.set_ylabel('Phase [rad]')
    ax.legend(fontsize=8); ax.grid(True, color='#30363d')

    ax = axes2[1]
    for tag, res in [('VPP', r_vpp_def), ('B', r_B_def), ('C', r_C_def)]:
        phi_1d = unwrap_phase(np.angle(res['U_CMOS'][:, iy_cmos]))
        ax.plot(z_prime * 1e3, phi_1d, label=tag, color=colors[tag])
    ax.set_title("Phase along z'  (Gaussian h=-100um, y'=0)")
    ax.set_xlabel("z' [mm]"); ax.set_ylabel('Phase [rad]')
    ax.legend(fontsize=8); ax.grid(True, color='#30363d')

    plt.tight_layout()
    plt.savefig(out_dir / 'sanity_06_phase_validation.png',
                dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  saved -> sanity_06_phase_validation.png")


if __name__ == '__main__':
    from datetime import datetime
    run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir  = Path(__file__).parent / 'results' / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    run(out_dir)
