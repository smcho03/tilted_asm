"""
sanity_03_far_field.py
──────────────────────
CMOS가 매우 멀 때 (z=1m) 가우시안 변형에 대해
  - CMOS 세기 / 위상 균일성 확인
출력: 3x3
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from me_0318       import forward_propagate
from sanity_utils  import plot_3x3
from sanity_params import (wavelength, A0, t, n1, n2_complex,
                           x_coords, y_coords, pixel_size)

# far-field 전용 CMOS 설정 (1m)
z_prop   = 1.0
x_cmos   = x_coords[-1] + z_prop
z_center = (x_coords[0] + x_coords[-1]) / 2

# Nyquist 만족하는 CMOS 범위
W        = (len(x_coords)//2) * pixel_size
half     = z_prop * wavelength / pixel_size + W
N_cmos   = 64
y_prime  = np.linspace(-half, half, N_cmos)
z_prime  = np.linspace(z_center - half, z_center + half, N_cmos)

X_tmp, Y_tmp = np.meshgrid(x_coords, y_coords)
h_def = (-wavelength/8) * np.exp(
    -((X_tmp - x_coords[len(x_coords)//2])**2 + Y_tmp**2) / (2*(4*pixel_size)**2))
h_ref = np.zeros_like(h_def)


def run(out_dir: Path):
    print(f"[03] Far-field  z={z_prop:.1f} m ...", flush=True)
    res_ref = forward_propagate(
        h_ref, x_coords, y_coords, y_prime, z_prime,
        wavelength, A0, t, n1, n2_complex, x_cmos)
    res_def = forward_propagate(
        h_def, x_coords, y_coords, y_prime, z_prime,
        wavelength, A0, t, n1, n2_complex, x_cmos)

    I_ref, I_def = res_ref['I_CMOS'], res_def['I_CMOS']
    print(f"  I_ref CoV : {I_ref.std()/I_ref.mean():.4f}", flush=True)
    print(f"  I_def CoV : {I_def.std()/I_def.mean():.4f}", flush=True)

    plot_3x3(
        title    = f'Sanity 03 – Far-Field  z={z_prop:.1f} m',
        h_ref_nm = h_ref * 1e9,
        h_def_nm = h_def * 1e9,
        x_coords = x_coords,
        y_coords = y_coords,
        I_ref    = I_ref,
        I_def    = I_def,
        phi_ref  = np.angle(res_ref['U_CMOS']),
        phi_def  = np.angle(res_def['U_CMOS']),
        y_prime  = y_prime,
        z_prime  = z_prime,
        out_path = out_dir / 'sanity_03_far_field.png',
    )

if __name__ == '__main__':
    out_dir = Path(__file__).parent / 'sanity_results'
    out_dir.mkdir(exist_ok=True)
    run(out_dir)