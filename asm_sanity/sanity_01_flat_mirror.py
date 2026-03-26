"""
sanity_01_flat_mirror.py  (me_asm)
h = 0 평면 거울 — 에너지, I_CMOS, 위상 확인
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(1, str(Path(__file__).parent.parent))

from me_asm       import forward_propagate_asm
from sanity_utils import plot_1x3
from sanity_params import (wavelength, A0, t, n1, n2_complex,
                            x_coords, y_coords, x_cmos,
                            y_prime, z_prime, pixel_size)

h_ref  = np.zeros((len(y_coords), len(x_coords)))
dx     = float(x_coords[1] - x_coords[0])
dy     = float(y_coords[1] - y_coords[0])
dy_c   = float(y_prime[1]  - y_prime[0])
dz_c   = float(z_prime[1]  - z_prime[0])


def run(out_dir: Path):
    print("[01] Flat mirror  h=0 ...", flush=True)
    res = forward_propagate_asm(
        h_ref, x_coords, y_coords, y_prime, z_prime,
        wavelength, A0, t, n1, n2_complex, x_cmos, pad_factor=2)

    E_in   = float(np.sum(np.abs(res['U_in'])**2))  * dx * dy
    E_ref  = float(np.sum(np.abs(res['U_ref'])**2)) * dx * dy
    E_cmos = float(np.sum(res['I_CMOS'])) * dy_c * dz_c

    print(f"  |R|^2 mean : {float(np.mean(np.abs(res['R_tilde'])**2)):.4f}")
    print(f"  E_in       : {E_in:.4e}")
    print(f"  E_ref      : {E_ref:.4e}  ({E_ref/E_in*100:.2f}% of E_in)")
    print(f"  E_absorbed : {E_in-E_ref:.4e}  ({(E_in-E_ref)/E_in*100:.2f}%)")
    print(f"  E_CMOS     : {E_cmos:.4e}  ({E_cmos/E_ref*100:.2f}% of E_ref)")
    print(f"  I_CMOS CoV : {res['I_CMOS'].std()/res['I_CMOS'].mean():.4f}")

    plot_1x3(
        title    = 'ASM Sanity 01 - Flat Mirror  h=0',
        h_nm     = h_ref * 1e9,
        x_coords = x_coords,
        y_coords = y_coords,
        I        = res['I_CMOS'],
        phi      = np.angle(res['U_CMOS']),
        y_prime  = y_prime,
        z_prime  = z_prime,
        out_path = out_dir / 'sanity_01_flat_mirror.png',
    )


if __name__ == '__main__':
    from datetime import datetime
    run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(__file__).parent / 'results' / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    run(out_dir)
