"""
sanity_04_deformation_patterns.py  (tilted_asm_sanity)
6가지 변형 패턴 — VPP/B/C 비교
"""
import sys, numpy as np
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from me_asm        import forward_propagate_asm
from me_tilted_asm import forward_propagate_B, forward_propagate_C
from sanity_utils  import plot_3x3, plot_comparison_3methods
from sanity_params import (wavelength, A0, t, n1, n2_complex,
                            x_coords, y_coords, x_cmos,
                            y_prime, z_prime)

N = len(x_coords)
h_amp        = -100e-6
sigma        = 4e-3
h_amp_local  = -10e-6
sigma_local  = 500e-6
h_amp_rand   = -1.5e-6
f_cutoff_rand = 0.02

h_ref = np.zeros((len(y_coords), N))
X_tmp, Y_tmp = np.meshgrid(x_coords, y_coords)
x_cx = x_coords[N // 2]

h1 = h_amp_local * np.exp(-((X_tmp - x_cx)**2 + Y_tmp**2) / (2*sigma_local**2))
h2 = h_amp * np.exp(-((X_tmp - x_cx)**2 + Y_tmp**2) / (2*sigma**2))
sigma_two = 7e-3
h3 = h_amp * (
    np.exp(-((X_tmp-x_cx)**2+(Y_tmp-y_coords[3*len(y_coords)//4])**2)/(2*sigma_two**2)) +
    np.exp(-((X_tmp-x_cx)**2+(Y_tmp-y_coords[  len(y_coords)//4])**2)/(2*sigma_two**2)))
h4 = h_amp * np.exp(-Y_tmp**2 / (2*sigma**2))
h5 = h_amp * np.exp(-(X_tmp - x_cx)**2 / (2*sigma**2))

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
    ('local_bump',    h1, '1. Local Gaussian bump (10 um)'),
    ('gaussian',      h2, '2. Gaussian 100 um (center)'),
    ('two_gaussians', h3, '3. Two Gaussians (y-sym)'),
    ('y_strip',       h4, '4. y-strip (Gaussian in y)'),
    ('x_strip',       h5, '5. x-strip (Gaussian in x)'),
    ('random',        h6, '6. Random band-limited (1.5 um)'),
]


def run(out_dir: Path):
    print("[04] Deformation patterns ...", flush=True)

    kwargs = dict(wavelength=wavelength, A0=A0, t=t, n1=n1, n2_complex=n2_complex,
                  x_cmos_location=x_cmos, pad_factor=2)

    res_ref_vpp = forward_propagate_asm(h_ref, x_coords, y_coords, y_prime, z_prime, **kwargs)
    res_ref_B   = forward_propagate_B  (h_ref, x_coords, y_coords, y_prime, z_prime, **kwargs)
    res_ref_C   = forward_propagate_C  (h_ref, x_coords, y_coords, y_prime, z_prime, **kwargs)

    for key, h_def, label in patterns:
        print(f"  [{key}] ...", flush=True)
        res_vpp = forward_propagate_asm(h_def, x_coords, y_coords, y_prime, z_prime, **kwargs)
        res_B   = forward_propagate_B  (h_def, x_coords, y_coords, y_prime, z_prime, **kwargs)
        res_C   = forward_propagate_C  (h_def, x_coords, y_coords, y_prime, z_prime, **kwargs)

        # 3x3: Approach B  (ref vs def)
        plot_3x3(
            title      = f'Sanity 04 – {label}  [Approach B]',
            h_ref_nm   = h_ref * 1e9, h_def_nm = h_def * 1e9,
            x_coords   = x_coords, y_coords = y_coords,
            I_ref      = res_ref_B['I_CMOS'], I_def = res_B['I_CMOS'],
            U_ref_cmos = res_ref_B['U_CMOS'], U_def_cmos = res_B['U_CMOS'],
            y_prime    = y_prime, z_prime = z_prime,
            out_path   = out_dir / f'sanity_04_{key}_B.png',
        )

        # 3x3: Approach C  (ref vs def)
        plot_3x3(
            title      = f'Sanity 04 – {label}  [Approach C]',
            h_ref_nm   = h_ref * 1e9, h_def_nm = h_def * 1e9,
            x_coords   = x_coords, y_coords = y_coords,
            I_ref      = res_ref_C['I_CMOS'], I_def = res_C['I_CMOS'],
            U_ref_cmos = res_ref_C['U_CMOS'], U_def_cmos = res_C['U_CMOS'],
            y_prime    = y_prime, z_prime = z_prime,
            out_path   = out_dir / f'sanity_04_{key}_C.png',
        )

        # 3-way 비교
        plot_comparison_3methods(
            title    = f'Sanity 04 – {label}: VPP vs B vs C',
            h_nm     = h_def * 1e9,
            x_coords = x_coords, y_coords = y_coords,
            I_vpp    = res_vpp['I_CMOS'],
            I_B      = res_B['I_CMOS'],
            I_C      = res_C['I_CMOS'],
            y_prime  = y_prime, z_prime = z_prime,
            out_path = out_dir / f'sanity_04_{key}_compare.png',
        )


if __name__ == '__main__':
    from datetime import datetime
    run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir  = Path(__file__).parent / 'results' / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    run(out_dir)
