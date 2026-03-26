"""
sanity_04_deformation_patterns.py  (me_asm)
6가지 변형 패턴 (각각 3x3 저장)
  1. 중앙 단일 픽셀
  2. 중앙 가우시안
  3. 대칭 두 점 가우시안
  4. y = k 직선 (한 행)
  5. x = k 직선 (한 열)
  6. 랜덤 변형 (band-limited, 공간 스무딩)
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(1, str(Path(__file__).parent.parent))

from me_asm       import forward_propagate_asm
from sanity_utils import plot_3x3
from sanity_params import (wavelength, A0, t, n1, n2_complex,
                            x_coords, y_coords, x_cmos,
                            y_prime, z_prime)

N     = len(x_coords)
# ── Nyquist 조건: max|dh/dx| < λ/(2·pixel) = 0.083 ──────────────
# Gaussian max_slope = 0.607·|h_amp|/sigma → sigma >= |h_amp|*0.607/0.083
h_amp        = -100e-6    # 100 µm 깊이
sigma        = 4e-3       # 4 mm  → max_slope = 0.607*100µm/4mm = 0.015 ✓
h_amp_local  = -10e-6     # 10 µm (소형 범프)
sigma_local  = 500e-6     # 500 µm → max_slope = 0.607*10µm/500µm = 0.012 ✓
h_amp_rand   = -1.5e-6    # 1.5 µm (band-limited noise)
f_cutoff_rand = 0.02      # 좁은 대역 → max_slope ≈ 0.063 ✓

h_ref = np.zeros((len(y_coords), N))
X_tmp, Y_tmp = np.meshgrid(x_coords, y_coords)
x_cx = x_coords[N//2]

# 1. 소형 Gaussian 범프 (단일 픽셀 → Nyquist 위반하므로 대체)
h1 = h_amp_local * np.exp(
    -((X_tmp - x_cx)**2 + Y_tmp**2) / (2*sigma_local**2))

# 2. 중앙 Gaussian
h2 = h_amp * np.exp(
    -((X_tmp - x_cx)**2 + Y_tmp**2) / (2*sigma**2))

# 3. 대칭 두 점 Gaussian (겹침으로 기울기 합산 → sigma_two=6mm 으로 Nyquist 확보)
sigma_two = 7e-3   # 7 mm → 중간점 합산 slope ≈ 0.057 ✓
h3 = h_amp * (
    np.exp(-((X_tmp-x_cx)**2+(Y_tmp-y_coords[3*len(y_coords)//4])**2)/(2*sigma_two**2)) +
    np.exp(-((X_tmp-x_cx)**2+(Y_tmp-y_coords[  len(y_coords)//4])**2)/(2*sigma_two**2)))

# 4. y-방향 Gaussian 스트립 (날카로운 행 대신 y방향 Gaussian으로 스무딩)
h4 = h_amp * np.exp(-Y_tmp**2 / (2*sigma**2))

# 5. x-방향 Gaussian 스트립 (날카로운 열 대신 x방향 Gaussian으로 스무딩)
h5 = h_amp * np.exp(-(X_tmp - x_cx)**2 / (2*sigma**2))

# 6. 랜덤 변형: 좁은 대역 band-limited noise, 소진폭
rng      = np.random.default_rng(seed=42)
raw      = rng.standard_normal((len(y_coords), N))
F        = np.fft.fft2(raw)
freq_y   = np.fft.fftfreq(len(y_coords))
freq_x   = np.fft.fftfreq(N)
FY, FX   = np.meshgrid(freq_y, freq_x, indexing='ij')
mask     = (np.sqrt(FY**2 + FX**2) < f_cutoff_rand).astype(float)
h6_raw   = np.real(np.fft.ifft2(F * mask))
h6       = h_amp_rand * h6_raw / (np.abs(h6_raw).max() + 1e-30)

patterns = [
    ('local_bump',    h1, '1. Local Gaussian bump (50 um)'),
    ('gaussian',      h2, '2. Gaussian 500 um (center)'),
    ('two_gaussians', h3, '3. Two Gaussians (y-sym)'),
    ('y_strip',       h4, '4. y-strip (Gaussian in y)'),
    ('x_strip',       h5, '5. x-strip (Gaussian in x)'),
    ('random',        h6, '6. Random band-limited (1.5 um)'),
]


def run(out_dir: Path):
    print("[04] Deformation patterns ...", flush=True)
    res_ref = forward_propagate_asm(
        h_ref, x_coords, y_coords, y_prime, z_prime,
        wavelength, A0, t, n1, n2_complex, x_cmos, pad_factor=2)

    for key, h_def, label in patterns:
        print(f"  [{key}] ...", flush=True)
        res_def = forward_propagate_asm(
            h_def, x_coords, y_coords, y_prime, z_prime,
            wavelength, A0, t, n1, n2_complex, x_cmos, pad_factor=2)
        plot_3x3(
            title      = f'ASM Sanity 04 – {label}',
            h_ref_nm   = h_ref * 1e9,
            h_def_nm   = h_def * 1e9,
            x_coords   = x_coords,
            y_coords   = y_coords,
            I_ref      = res_ref['I_CMOS'],
            I_def      = res_def['I_CMOS'],
            U_ref_cmos = res_ref['U_CMOS'],
            U_def_cmos = res_def['U_CMOS'],
            y_prime    = y_prime,
            z_prime    = z_prime,
            out_path   = out_dir / f'sanity_04_{key}.png',
        )


if __name__ == '__main__':
    from datetime import datetime
    run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(__file__).parent / 'results' / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    run(out_dir)
