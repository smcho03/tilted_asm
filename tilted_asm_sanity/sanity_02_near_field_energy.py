"""
sanity_02_near_field_energy.py  (tilted_asm_sanity)
근거리 (gap=100um) 에너지 보존 — VPP, B, C 비교
"""
import sys, numpy as np
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from me_asm        import forward_propagate_asm
from me_tilted_asm import forward_propagate_B, forward_propagate_C
from sanity_utils  import plot_1x3, plot_comparison_3methods

wavelength  = 500e-9; A0, t = 1.0, 0.0
n1 = 1.0;  n2_complex = complex(0.97112, 1.8737)

r_width    = 500e-6
N          = 256
pixel_size = r_width / N

x_coords = np.linspace(1, N, N) * pixel_size
y_coords = np.linspace(-(N-1)/2, (N-1)/2, N) * pixel_size
h_ref    = np.zeros((N, N))

dx = float(x_coords[1] - x_coords[0])
dy = float(y_coords[1] - y_coords[0])

x_cmos_near = x_coords[-1] + 100e-6

z_center   = (x_coords[0] + x_coords[-1]) / 2
cmos_width = 600e-6
N_cmos     = 256

y_prime  = np.linspace(-cmos_width/2, cmos_width/2, N_cmos)
z_prime  = np.linspace(z_center - cmos_width/2, z_center + cmos_width/2, N_cmos)
dy_cmos  = float(y_prime[1] - y_prime[0])
dz_cmos  = float(z_prime[1] - z_prime[0])


def run(out_dir: Path):
    print("[02] Near-field (gap=100um) + Energy ...", flush=True)

    kwargs = dict(wavelength=wavelength, A0=A0, t=t, n1=n1, n2_complex=n2_complex,
                  x_cmos_location=x_cmos_near, pad_factor=4)

    res_vpp = forward_propagate_asm(h_ref, x_coords, y_coords, y_prime, z_prime, **kwargs)
    res_B   = forward_propagate_B  (h_ref, x_coords, y_coords, y_prime, z_prime, **kwargs)
    res_C   = forward_propagate_C  (h_ref, x_coords, y_coords, y_prime, z_prime, **kwargs)

    for tag, res in [('VPP', res_vpp), ('B', res_B), ('C', res_C)]:
        E_in   = float(np.sum(np.abs(res['U_in'])**2))  * dx * dy
        E_ref  = float(np.sum(np.abs(res['U_ref'])**2)) * dx * dy
        E_cmos = float(np.sum(res['I_CMOS'])) * dy_cmos * dz_cmos
        print(f"  [{tag}]  E_in={E_in:.4e}  E_ref={E_ref:.4e}  "
              f"E_CMOS={E_cmos:.4e}  ({E_cmos/E_ref*100:.2f}% of E_ref)")

    for tag, res in [('VPP', res_vpp), ('B', res_B), ('C', res_C)]:
        plot_1x3(
            title    = f'Tilted ASM Sanity 02 – Near-Field gap=100um  [{tag}]',
            h_nm     = h_ref * 1e9,
            x_coords = x_coords, y_coords = y_coords,
            I        = res['I_CMOS'],
            phi      = np.angle(res['U_CMOS']),
            y_prime  = y_prime, z_prime = z_prime,
            out_path = out_dir / f'sanity_02_near_field_{tag}.png',
        )

    plot_comparison_3methods(
        title    = 'Sanity 02 – Near-Field gap=100um: VPP vs B vs C',
        h_nm     = h_ref * 1e9,
        x_coords = x_coords, y_coords = y_coords,
        I_vpp    = res_vpp['I_CMOS'],
        I_B      = res_B['I_CMOS'],
        I_C      = res_C['I_CMOS'],
        y_prime  = y_prime, z_prime = z_prime,
        out_path = out_dir / 'sanity_02_near_field_compare.png',
    )


if __name__ == '__main__':
    from datetime import datetime
    run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir  = Path(__file__).parent / 'results' / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    run(out_dir)
