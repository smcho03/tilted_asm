"""
sanity_01_flat_mirror.py  (tilted_asm_sanity)
h = 0 평면 거울 — 에너지 보존 + 세 방식 비교 (VPP, B, C)
"""
import sys, numpy as np
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from me_asm          import forward_propagate_asm
from me_tilted_asm   import forward_propagate_B, forward_propagate_C
from sanity_utils    import plot_1x3, plot_comparison_3methods
from sanity_params   import (wavelength, A0, t, n1, n2_complex,
                              x_coords, y_coords, x_cmos,
                              y_prime, z_prime, pixel_size)

h_ref = np.zeros((len(y_coords), len(x_coords)))
dx    = float(x_coords[1] - x_coords[0])
dy    = float(y_coords[1] - y_coords[0])
dy_c  = float(y_prime[1]  - y_prime[0])
dz_c  = float(z_prime[1]  - z_prime[0])


def run(out_dir: Path):
    print("[01] Flat mirror  h=0 ...", flush=True)

    kwargs = dict(wavelength=wavelength, A0=A0, t=t, n1=n1, n2_complex=n2_complex,
                  x_cmos_location=x_cmos, pad_factor=2)

    res_vpp = forward_propagate_asm(h_ref, x_coords, y_coords, y_prime, z_prime, **kwargs)
    res_B   = forward_propagate_B  (h_ref, x_coords, y_coords, y_prime, z_prime, **kwargs)
    res_C   = forward_propagate_C  (h_ref, x_coords, y_coords, y_prime, z_prime, **kwargs)

    for tag, res in [('VPP', res_vpp), ('B', res_B), ('C', res_C)]:
        E_in   = float(np.sum(np.abs(res['U_in'])**2))  * dx * dy
        E_ref  = float(np.sum(np.abs(res['U_ref'])**2)) * dx * dy
        E_cmos = float(np.sum(res['I_CMOS'])) * dy_c * dz_c
        print(f"  [{tag}]  E_in={E_in:.4e}  E_ref={E_ref:.4e}  "
              f"E_CMOS={E_cmos:.4e}  ({E_cmos/E_ref*100:.2f}% of E_ref)  "
              f"I_CoV={res['I_CMOS'].std()/(res['I_CMOS'].mean()+1e-30):.4f}")

    # 개별 저장 (1x3)
    for tag, res in [('VPP', res_vpp), ('B', res_B), ('C', res_C)]:
        plot_1x3(
            title    = f'Tilted ASM Sanity 01 – Flat Mirror  [{tag}]',
            h_nm     = h_ref * 1e9,
            x_coords = x_coords, y_coords = y_coords,
            I        = res['I_CMOS'],
            phi      = np.angle(res['U_CMOS']),
            y_prime  = y_prime, z_prime = z_prime,
            out_path = out_dir / f'sanity_01_flat_mirror_{tag}.png',
        )

    # 3-way 비교
    plot_comparison_3methods(
        title    = 'Sanity 01 – Flat Mirror: VPP vs B vs C',
        h_nm     = h_ref * 1e9,
        x_coords = x_coords, y_coords = y_coords,
        I_vpp    = res_vpp['I_CMOS'],
        I_B      = res_B['I_CMOS'],
        I_C      = res_C['I_CMOS'],
        y_prime  = y_prime, z_prime = z_prime,
        out_path = out_dir / 'sanity_01_flat_mirror_compare.png',
    )


if __name__ == '__main__':
    from datetime import datetime
    run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir  = Path(__file__).parent / 'results' / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    run(out_dir)
