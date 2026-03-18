"""
sanity_02_near_field_energy.py
──────────────────────────────
CMOS ~ 1 um (near-field) + 에너지 보존
  - z' 방향 밝기 분포 확인
  - 에너지 터미널 출력
출력: 1x3
"""
import sys, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from me_0318 import forward_propagate
from sanity_utils import plot_1x3

wavelength = 500e-9;  A0, t = 1.0, 0.0
n1 = 1.0;  n2_complex = complex(0.97112, 1.8737)
r_width    = 500e-6        # 반사면 실제 폭
N          = 64            # 픽셀 개수

pixel_size = r_width / N   # 픽셀 간격 자동 계산

x_coords = np.linspace(1, N, N) * pixel_size       # x > 0
y_coords = np.linspace(-(N-1)//2, (N-1)//2, N) * pixel_size
h_ref    = np.zeros((N, N))

dx = float(x_coords[1]-x_coords[0])
dy = float(y_coords[1]-y_coords[0])

x_cmos_near = x_coords[-1] + 1e-6   # 3 um gap

z_center = (x_coords[0] + x_coords[-1]) / 2

cmos_width    = 600e-6        # CMOS 실제 폭
N_cmos     = 64         # CMOS 픽셀 개수

cmos_pixel = cmos_width / N_cmos   # CMOS 픽셀 간격 자동 계산

y_prime = np.linspace(-cmos_width/2, cmos_width/2, N_cmos)
z_prime = np.linspace(z_center - cmos_width/2, z_center + cmos_width/2, N_cmos)
dy_cmos = float(y_prime[1]-y_prime[0])
dz_cmos = float(z_prime[1]-z_prime[0])



def run(out_dir: Path):
    print("[02] Near-field (~1 um) + Energy ...")
    res = forward_propagate(
        h_ref, x_coords, y_coords, y_prime, z_prime,
        wavelength, A0, t, n1, n2_complex, x_cmos_near)

    E_in  = float(np.sum(np.abs(res['U_in'])**2))  * dx * dy
    E_ref = float(np.sum(np.abs(res['U_ref'])**2)) * dx * dy
    E_cmos = float(np.sum(res['I_CMOS'])) * dy_cmos * dz_cmos
    print(f"  E_in       : {E_in:.4e}")
    print(f"  E_ref      : {E_ref:.4e}  ({E_ref/E_in*100:.2f}%)")
    print(f"  E_absorbed : {E_in-E_ref:.4e}  ({(E_in-E_ref)/E_in*100:.2f}%)")
    print(f"  E_CMOS     : {E_cmos:.4e}  ({E_cmos/E_ref*100:.2f}% of E_ref)")

    plot_1x3(
        title    = 'Sanity 02 – Near-Field  (gap = 3 um)',
        h_nm     = h_ref * 1e9,
        x_coords = x_coords,
        y_coords = y_coords,
        I        = res['I_CMOS'],
        phi      = np.angle(res['U_CMOS']),
        y_prime  = y_prime,
        z_prime  = z_prime,
        out_path = out_dir / 'sanity_02_near_field_energy.png',
    )

if __name__ == '__main__':
    out_dir = Path(__file__).parent / 'sanity_results'
    out_dir.mkdir(exist_ok=True)
    run(out_dir)