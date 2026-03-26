"""
sanity_params.py  (tilted_asm_sanity 공통 파라미터)
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

wavelength  = 500e-9
A0          = 1.0
t           = 0.0
n1          = 1.0
n2_complex  = complex(0.97112, 1.8737)

N           = 4096
pixel_size  = 3e-6            # 3 µm
r_width     = N * pixel_size  # ~12.3 mm 반사면

x_coords = np.linspace(1, N, N) * pixel_size
y_coords = np.linspace(-(N-1)/2, (N-1)/2, N) * pixel_size

cmos_width  = r_width         # CMOS = 거울과 동일 폭
N_cmos      = 512
cmos_pixel  = cmos_width / N_cmos

x_center = (x_coords[0] + x_coords[-1]) / 2
z_center = x_center
x_cmos   = x_center + 10e-3  # 거울 중심에서 10 mm

y_prime = np.linspace(-cmos_width/2,  cmos_width/2,  N_cmos)
z_prime = np.linspace(z_center - cmos_width/2,
                      z_center + cmos_width/2, N_cmos)
