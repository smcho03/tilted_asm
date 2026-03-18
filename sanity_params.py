"""
sanity_params.py
────────────────
모든 sanity 파일이 공유하는 공통 파라미터
여기만 수정하면 모든 sanity에 반영됨
"""
import numpy as np

# ── 광원 ──────────────────────────────────────────────────────
wavelength  = 500e-9        # 파장 [m]
A0          = 1.0
t           = 0.0
n1          = 1.0
n2_complex  = complex(0.97112, 1.8737)   # 금 500nm

# ── 반사 멤브레인 ─────────────────────────────────────────────
r_width     = 500e-6        # 반사면 실제 폭 [m]
N           = 128           # 픽셀 개수
pixel_size  = r_width / N   # 픽셀 간격 자동 계산 = 3.9 μm

x_coords = np.linspace(1, N, N) * pixel_size          # x > 0
y_coords = np.linspace(-(N-1)/2, (N-1)/2, N) * pixel_size  # 대칭

# ── CMOS ─────────────────────────────────────────────────────
cmos_width  = 600e-6        # CMOS 실제 폭 [m]
N_cmos      = 200           # CMOS 픽셀 개수
cmos_pixel  = cmos_width / N_cmos   # = 3.0 μm

x_center = (x_coords[0] + x_coords[-1]) / 2
z_center = x_center
x_cmos   = x_center + 10e-3   # 중심거리 10mm 고정

y_prime  = np.linspace(-cmos_width/2,  cmos_width/2,  N_cmos)
z_prime  = np.linspace(z_center - cmos_width/2,
                       z_center + cmos_width/2, N_cmos)

# ── Nyquist 검증 (import 시 자동 출력) ───────────────────────
_z_min    = x_cmos - x_coords[-1]
_nyq_src  = wavelength * _z_min / (2 * (cmos_width/2 + abs(y_coords[0])))
_nyq_cmos = wavelength * _z_min / (2 * (z_prime[-1] - x_coords[0]))
_nyq_min  = min(_nyq_src, _nyq_cmos)

assert pixel_size  < _nyq_src,  f"소스 Nyquist 위반: {pixel_size*1e6:.2f}um > {_nyq_src*1e6:.2f}um"
assert cmos_pixel  < _nyq_cmos, f"CMOS Nyquist 위반: {cmos_pixel*1e6:.2f}um > {_nyq_cmos*1e6:.2f}um"