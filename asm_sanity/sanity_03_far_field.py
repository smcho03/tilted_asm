"""
sanity_03_far_field.py  (me_asm)
원거리장 (z=100mm) — ASM은 Fraunhofer 근사 없이 처리
ref vs deformed 3x3
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(1, str(Path(__file__).parent.parent))

from me_asm       import forward_propagate_asm
from sanity_utils import plot_3x3
from sanity_params import (wavelength, A0, t, n1, n2_complex,
                            x_coords, y_coords, pixel_size)

z_prop   = 100e-3   # 100 mm 원거리
x_cmos   = x_coords[-1] + z_prop
z_center = (x_coords[0] + x_coords[-1]) / 2

# CMOS 범위: z_vpp_grid 는 x_coords 범위와 비슷한 ~500µm 폭
# → z_prime 도 같은 스케일 안에 있어야 보간이 0이 안 됨
# 빔 확산: λ·z_prop / (D/2) ≈ 200µm  →  ±600µm 면 충분
cmos_width = 600e-6
N_cmos = 256
y_prime = np.linspace(-cmos_width / 2, cmos_width / 2, N_cmos)
z_prime = np.linspace(z_center - cmos_width / 2, z_center + cmos_width / 2, N_cmos)

X_tmp, Y_tmp = np.meshgrid(x_coords, y_coords)
h_def = (-wavelength / 8) * np.exp(
    -((X_tmp - x_coords[len(x_coords)//2])**2 + Y_tmp**2)
    / (2 * (4 * pixel_size)**2))
h_ref = np.zeros_like(h_def)


def run(out_dir: Path):
    print(f"[03] Far-field  z={z_prop*1e3:.0f}mm ...", flush=True)
    res_ref = forward_propagate_asm(
        h_ref, x_coords, y_coords, y_prime, z_prime,
        wavelength, A0, t, n1, n2_complex, x_cmos, pad_factor=2)
    res_def = forward_propagate_asm(
        h_def, x_coords, y_coords, y_prime, z_prime,
        wavelength, A0, t, n1, n2_complex, x_cmos, pad_factor=2)

    I_ref, I_def = res_ref['I_CMOS'], res_def['I_CMOS']
    print(f"  I_ref CoV : {I_ref.std()/(I_ref.mean()+1e-30):.4f}")
    print(f"  I_def CoV : {I_def.std()/(I_def.mean()+1e-30):.4f}")

    plot_3x3(
        title      = f'ASM Sanity 03 - Far-Field  z={z_prop*1e3:.0f}mm',
        h_ref_nm   = h_ref * 1e9,
        h_def_nm   = h_def * 1e9,
        x_coords   = x_coords,
        y_coords   = y_coords,
        I_ref      = I_ref,
        I_def      = I_def,
        U_ref_cmos = res_ref['U_CMOS'],
        U_def_cmos = res_def['U_CMOS'],
        y_prime    = y_prime,
        z_prime    = z_prime,
        out_path   = out_dir / 'sanity_03_far_field.png',
    )


if __name__ == '__main__':
    from datetime import datetime
    run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(__file__).parent / 'results' / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    run(out_dir)
