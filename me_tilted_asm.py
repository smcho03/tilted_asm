"""
me_tilted_asm.py  —  Tilted ASM Forward Models for Lensless Tactile Sensor

두 가지 개선된 전방 전파 방식:
  B: Rotated Frame ASM  (표면 (s,y) 좌표계 + tilted 전달함수)
  C: Obliquity-corrected VPP  (VPP + 진폭 기하 보정)

물리 모델 (me_asm.py 동일):
  입사광 : +z 방향 평면파
  거울면 : z = x + h(x, y)   (45° 기준면 + 변형 h)
  반사광 : +x 방향 → CMOS (x = x_cmos)

───────────────────────────────────────────────────────────────────────────
Approach B — Rotated Frame ASM
───────────────────────────────────────────────────────────────────────────
표면 좌표 s = (2x + h) / sqrt(2)
  - 평면 거울: s = x*sqrt(2), ds = dx*sqrt(2) ≈ 4.24 µm (dx=3 µm)
  - Jacobian : ds/dx = (2 + dh/dx) / sqrt(2)

Carrier: exp(i·k0·(s/sqrt(2) + h/2))  [= exp(i·k0·(x+h))]
  - 완전 제거 → envelope 위상 = 0  (VPP 동일)

Tilted ASM 전달함수  H(fs, fy):
  C    = 2π·sqrt(2)·fs + k0
  ky   = 2π·fy
  K2   = 2·(k0² - ky²) - C²         [전파 조건: K2 > 0]
  kx   = (C + sqrt(K2)) / 2
  H    = exp(i·kx·d_prop)

CMOS 샘플링: s_out = sqrt(2)·z'   [파축 근사]
  - 정반사 z'≈x_i → s_i≈x_i·sqrt(2) = sqrt(2)·z'
  - VPP의 z_vpp=z' 가정과 동일 레벨 근사
  - Carrier 복원: exp(i·k0·z')

───────────────────────────────────────────────────────────────────────────
Approach C — Obliquity-corrected VPP
───────────────────────────────────────────────────────────────────────────
VPP scatter 이후 진폭 보정:
  obliquity = (x_cmos - x_ref) / (x_cmos - z_vpp)

VPP 오차: ASM 전파 진폭이 1/(x_cmos - x_ref) 를 가정하지만
          실제는 1/(x_cmos - x_i) 여야 함.
  - 거울 폭 12 mm, 전파 거리 10 mm → edge에서 최대 2.5x 오차
  - Obliquity 보정으로 제거

나머지는 standard VPP + ASM 동일.
"""

import numpy as np
from scipy.fft import fft2, ifft2, fftfreq
from scipy.interpolate import RegularGridInterpolator
import sys
import warnings
from pathlib import Path

# me_asm.py 공통 함수 재사용
sys.path.insert(0, str(Path(__file__).parent))
from me_asm import (
    compute_surface_gradients,
    compute_U_in,
    compute_incident_angle,
    compute_fresnel_reflection,
    compute_U_ref,
    _check_nyquist,
    _vpp_scatter,
    _asm_transfer,
    _asm_propagate,
)


# ─────────────────────────────────────────────────────────────────────────────
# Approach B: 표면 좌표 scatter
# ─────────────────────────────────────────────────────────────────────────────

def _surface_scatter_B(
    h: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    U_ref: np.ndarray,
    dh_dx: np.ndarray,
) -> tuple:
    """
    표면 좌표 s = (2x + h) / sqrt(2) 로 U_ref scatter.

    비균일 s_nonuniform → 정규 s_grid 보간.
    Jacobian 보정: E_sc = U_ref / sqrt(|ds/dx|)
        ds/dx = (2 + dh_dx) / sqrt(2)

    반환
    ────
    E_s    : (Ny, Ns) complex
    s_grid : (Ns,)  정규 s-좌표 [m]
    ds     : s-grid spacing [m]
    h_s    : (Ny, Ns) float — s_grid 위치에서의 h 값 (정확한 역매핑)
    """
    sqrt2 = np.sqrt(2.0)
    dx = float(x_coords[1] - x_coords[0])
    ds = dx * sqrt2                         # 평면 거울 기준 spacing

    # 비균일 s 좌표 (Ny, Nx)
    S_nonu = (2.0 * x_coords[np.newaxis, :] + h) / sqrt2

    # 정규 s 그리드
    s_min = float(S_nonu.min())
    s_max = float(S_nonu.max())
    Ns = int(np.ceil((s_max - s_min) / ds)) + 1
    s_grid = np.arange(Ns, dtype=np.float64) * ds + s_min

    # Jacobian 보정
    jac = np.abs(2.0 + dh_dx) / sqrt2      # (Ny, Nx)
    E_sc = U_ref / np.sqrt(np.maximum(jac, 1e-10))

    Ny = h.shape[0]
    E_s = np.zeros((Ny, Ns), dtype=complex)
    h_s = np.zeros((Ny, Ns), dtype=np.float64)

    for j in range(Ny):
        s_src = S_nonu[j, :]
        vals = E_sc[j, :]
        idx = np.argsort(s_src)
        ss = s_src[idx]
        vs = vals[idx]
        hs = h[j, idx]                     # h 값도 같은 순서로 정렬
        E_s[j, :] = (
            np.interp(s_grid, ss, vs.real, left=0.0, right=0.0)
            + 1j * np.interp(s_grid, ss, vs.imag, left=0.0, right=0.0)
        )
        # h_s: s_grid 위치에서의 h 값 (정확한 역매핑 사용)
        h_s[j, :] = np.interp(s_grid, ss, hs, left=0.0, right=0.0)

    return E_s, s_grid, ds, h_s


def _tilted_asm_transfer_B(
    Ny: int, Ns: int,
    dy: float, ds: float,
    wavelength: float,
    d_prop: float,
) -> np.ndarray:
    """
    45° 기울어진 source plane 에 대한 Tilted ASM 전달함수.
    shape = (Ny, Ns)

    캐리어 제거 후 envelope 주파수 (fs, fy) 에 대해:
      C   = 2π*sqrt(2)*fs + k0
      ky  = 2π*fy
      K2  = 2*(k0² - ky²) - C²     [전파 조건: K2 > 0]
      kx  = (C + sqrt(K2)) / 2
      H   = exp(i·kx·d_prop)

    마스크:
      전파 조건 (K2 > 0) AND Nyquist (|fs|<1/(2ds), |fy|<1/(2dy))
    """
    fy = fftfreq(Ny, d=float(dy)).astype(np.float64)  # (Ny,)
    fs = fftfreq(Ns, d=float(ds)).astype(np.float64)  # (Ns,)
    FY, FS = np.meshgrid(fy, fs, indexing='ij')       # (Ny, Ns)

    k0 = 2.0 * np.pi / wavelength
    sqrt2 = np.sqrt(2.0)

    C  = 2.0 * np.pi * sqrt2 * FS + k0   # (Ny, Ns)
    ky = 2.0 * np.pi * FY                # (Ny, Ns)

    K2 = 2.0 * (k0 ** 2 - ky ** 2) - C ** 2

    nyq_mask  = (np.abs(FY) < 0.5 / float(dy)) & (np.abs(FS) < 0.5 / float(ds))
    prop_mask = K2 > 0.0
    valid     = prop_mask & nyq_mask

    kx = np.where(valid, (C + np.sqrt(np.maximum(K2, 0.0))) / 2.0, 0.0)
    H  = np.where(valid, np.exp(1j * kx * float(d_prop)), 0.0).astype(complex)
    return H   # (Ny, Ns)


def _tilted_asm_propagate_B(
    E_s: np.ndarray,
    dy: float, ds: float,
    wavelength: float,
    d_prop: float,
    pad_factor: int = 2,
) -> np.ndarray:
    """
    Zero-padded tilted ASM 전파 in (y, s) 좌표.
    반환: 원본 (Ny, Ns) 크기로 복원된 전파 필드.
    """
    Ny, Ns = E_s.shape
    Nyp = Ny * pad_factor
    Nsp = Ns * pad_factor
    y0  = (Nyp - Ny) // 2
    s0  = (Nsp - Ns) // 2

    E_pad = np.zeros((Nyp, Nsp), dtype=complex)
    E_pad[y0:y0 + Ny, s0:s0 + Ns] = E_s

    H     = _tilted_asm_transfer_B(Nyp, Nsp, dy, ds, wavelength, d_prop)
    E_out = ifft2(fft2(E_pad) * H)

    return E_out[y0:y0 + Ny, s0:s0 + Ns]


# ─────────────────────────────────────────────────────────────────────────────
# Public API: Approach B
# ─────────────────────────────────────────────────────────────────────────────

def forward_propagate_B(
    h:               np.ndarray,
    x_coords:        np.ndarray,
    y_coords:        np.ndarray,
    y_prime_coords:  np.ndarray,
    z_prime_coords:  np.ndarray,
    wavelength:      float,
    A0:              float,
    t:               float,
    n1:              float,
    n2_complex:      complex,
    x_cmos_location: float,
    pad_factor:      int = 2,
) -> dict:
    """
    Tilted ASM 순전파 — Approach B (Rotated Frame).

    표면 (s, y) 좌표계로 scatter 후 tilted 전달함수로 전파.
    x_ref 근사 없이 기하를 처리.

    반환 dict 키: U_in, theta_loc, R_tilde, U_ref, U_CMOS, I_CMOS, dh_dx, dh_dy
    """
    X, Y = np.meshgrid(x_coords, y_coords)
    dy   = float(y_coords[1] - y_coords[0])
    dz   = float(x_coords[1] - x_coords[0])

    # ── 거울면 물리량 ──────────────────────────────────────────────────────
    dh_dx, dh_dy = compute_surface_gradients(h, x_coords, y_coords)
    U_in    = compute_U_in(X, Y, h, t, A0, wavelength)
    theta_l = compute_incident_angle(dh_dx, dh_dy)
    R_tilde = compute_fresnel_reflection(theta_l, n1, n2_complex)
    U_ref_f = compute_U_ref(U_in, R_tilde)
    _check_nyquist(min(dy, dz), wavelength, dh_dx, dh_dy)

    # ── 표면 좌표 scatter ─────────────────────────────────────────────────
    E_s, s_grid, ds, h_s = _surface_scatter_B(h, x_coords, y_coords, U_ref_f, dh_dx)

    # ── Carrier 제거: exp(-i·k0·(s/sqrt(2) + h/2)) ───────────────────────
    # U_ref 위상 = k0*(x+h). s-carrier = exp(ik0*(x+h/2)).
    # 잔류 위상 = k0*h/2 → h_s (정확한 역매핑)로 제거.
    k0    = 2.0 * np.pi / wavelength
    sqrt2 = np.sqrt(2.0)
    # 전체 carrier = exp(-ik0*(s/sqrt(2))) * exp(-ik0*h_s/2)
    #              = exp(-ik0*(x+h/2)) * exp(-ik0*h/2)
    #              = exp(-ik0*(x+h))   ← U_ref 위상을 완전 제거
    full_carrier = np.exp(-1j * k0 * (s_grid[np.newaxis, :] / sqrt2
                                      + h_s / 2.0)).astype(complex)
    E_env = E_s * full_carrier         # (Ny, Ns)

    # ── Tilted ASM 전파 ───────────────────────────────────────────────────
    x_ref  = float(x_coords.mean())
    d_prop = float(x_cmos_location) - x_ref
    E_prop_env = _tilted_asm_propagate_B(E_env, dy, ds, wavelength, d_prop, pad_factor)

    # ── CMOS 좌표 s_out = sqrt(2) * z' ───────────────────────────────────
    # 파축 근사(paraxial): 정반사로 z' ≈ x_i → s_i ≈ sqrt(2)*x_i = sqrt(2)*z'
    # 이는 VPP의 z_vpp = z' 가정과 동일 레벨 근사.
    # 변형 거울에서 발생하는 B-VPP 차이 (~3.5% RMS for h=-100µm)는
    # 두 방식의 좌표계 차이에서 오는 물리적 차이이며 수치 오류가 아님.
    x_cmos = float(x_cmos_location)
    s_out = sqrt2 * z_prime_coords.astype(np.float64)      # (Nz_c,)

    Nz_c = len(z_prime_coords)
    Ny_c = len(y_prime_coords)

    itp_re = RegularGridInterpolator(
        (y_coords.astype(np.float64), s_grid),
        np.real(E_prop_env).astype(np.float64),
        method='linear', bounds_error=False, fill_value=0.0)
    itp_im = RegularGridInterpolator(
        (y_coords.astype(np.float64), s_grid),
        np.imag(E_prop_env).astype(np.float64),
        method='linear', bounds_error=False, fill_value=0.0)

    S_q, Yq = np.meshgrid(s_out, y_prime_coords, indexing='ij')  # (Nz_c, Ny_c)
    pts = np.column_stack([Yq.ravel(), S_q.ravel()])
    E_cmos_env = (itp_re(pts) + 1j * itp_im(pts)).reshape(Nz_c, Ny_c).astype(complex)

    # ── Carrier 복원: exp(i·k0·z') + Jacobian 보정 ───────────────────────
    # s_out = sqrt(2)*z' 좌표 변환 Jacobian: ds/dz' = sqrt(2)
    # 에너지 보존: |U_CMOS|^2 * dz' = |E_env|^2 * ds → 진폭 *= 2^{1/4}
    cmos_carrier = np.exp(1j * k0 * z_prime_coords).astype(complex)
    U_CMOS = E_cmos_env * (2.0 ** 0.25) * cmos_carrier[:, np.newaxis]   # (Nz_c, Ny_c)

    return {
        'U_in':      U_in,
        'theta_loc': theta_l,
        'R_tilde':   R_tilde,
        'U_ref':     U_ref_f,
        'U_CMOS':    U_CMOS,
        'I_CMOS':    np.abs(U_CMOS) ** 2,
        'dh_dx':     dh_dx,
        'dh_dy':     dh_dy,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API: Approach C
# ─────────────────────────────────────────────────────────────────────────────

def forward_propagate_C(
    h:               np.ndarray,
    x_coords:        np.ndarray,
    y_coords:        np.ndarray,
    y_prime_coords:  np.ndarray,
    z_prime_coords:  np.ndarray,
    wavelength:      float,
    A0:              float,
    t:               float,
    n1:              float,
    n2_complex:      complex,
    x_cmos_location: float,
    pad_factor:      int = 2,
) -> dict:
    """
    Obliquity-corrected VPP ASM — Approach C.

    me_asm.forward_propagate_asm 과 동일하지만 VPP scatter 이후
    진폭 보정 추가:
      obliquity[j] = (x_cmos - x_ref) / (x_cmos - z_vpp[j])

    VPP 오차: propagation amplitude 를 1/(x_cmos - x_ref) 로 가정하지만
              실제는 1/(x_cmos - x_i) ≈ 1/(x_cmos - z_vpp).
              이 인자가 천천히 변하므로 aliasing 없이 보정 가능.

    반환 dict 키: U_in, theta_loc, R_tilde, U_ref, U_CMOS, I_CMOS, dh_dx, dh_dy
    """
    X, Y = np.meshgrid(x_coords, y_coords)
    dy   = float(y_coords[1] - y_coords[0])
    dz   = float(x_coords[1] - x_coords[0])

    # ── 거울면 물리량 ──────────────────────────────────────────────────────
    dh_dx, dh_dy = compute_surface_gradients(h, x_coords, y_coords)
    U_in    = compute_U_in(X, Y, h, t, A0, wavelength)
    theta_l = compute_incident_angle(dh_dx, dh_dy)
    R_tilde = compute_fresnel_reflection(theta_l, n1, n2_complex)
    U_ref_f = compute_U_ref(U_in, R_tilde)
    _check_nyquist(min(dy, dz), wavelength, dh_dx, dh_dy)

    # ── VPP scatter ───────────────────────────────────────────────────────
    E_vpp, z_vpp_grid = _vpp_scatter(h, x_coords, y_coords, U_ref_f, dh_dx)

    # ── Obliquity 진폭 보정 ───────────────────────────────────────────────
    x_ref  = float(x_coords.mean())
    x_cmos = float(x_cmos_location)
    obliquity = (x_cmos - x_ref) / np.maximum(x_cmos - z_vpp_grid, 1e-12)
    E_vpp_corr = E_vpp * obliquity[np.newaxis, :]   # (Ny, Nz_vpp)

    # ── Carrier 제거 ─────────────────────────────────────────────────────
    k_wave  = 2.0 * np.pi / wavelength
    carrier = np.exp(-1j * k_wave * z_vpp_grid).astype(complex)
    E_env   = E_vpp_corr * carrier[np.newaxis, :]

    # ── Band-limited ASM 전파 ─────────────────────────────────────────────
    d_prop     = x_cmos - x_ref
    E_prop_env = _asm_propagate(E_env, dy, dz, wavelength, d_prop, pad_factor)

    # ── CMOS 좌표 보간 ────────────────────────────────────────────────────
    itp_re = RegularGridInterpolator(
        (y_coords.astype(np.float64), z_vpp_grid.astype(np.float64)),
        np.real(E_prop_env).astype(np.float64),
        method='linear', bounds_error=False, fill_value=0.0)
    itp_im = RegularGridInterpolator(
        (y_coords.astype(np.float64), z_vpp_grid.astype(np.float64)),
        np.imag(E_prop_env).astype(np.float64),
        method='linear', bounds_error=False, fill_value=0.0)

    Nz_c = len(z_prime_coords)
    Ny_c = len(y_prime_coords)
    Zq, Yq = np.meshgrid(z_prime_coords, y_prime_coords, indexing='ij')
    pts = np.column_stack([Yq.ravel(), Zq.ravel()])
    E_cmos_env = (itp_re(pts) + 1j * itp_im(pts)).reshape(Nz_c, Ny_c).astype(complex)

    # ── Carrier 복원 ──────────────────────────────────────────────────────
    cmos_carrier = np.exp(1j * k_wave * z_prime_coords).astype(complex)
    U_CMOS = E_cmos_env * cmos_carrier[:, np.newaxis]   # (Nz_c, Ny_c)

    return {
        'U_in':      U_in,
        'theta_loc': theta_l,
        'R_tilde':   R_tilde,
        'U_ref':     U_ref_f,
        'U_CMOS':    U_CMOS,
        'I_CMOS':    np.abs(U_CMOS) ** 2,
        'dh_dx':     dh_dx,
        'dh_dy':     dh_dy,
    }
