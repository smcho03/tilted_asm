"""
me_asm.py  ─  Lensless Tactile Sensor Simulator
VPP (Virtual Plane Projection) + Band-limited ASM

물리 모델:
  입사광 : +z 방향 평면파
  거울면 : z = x + h(x, y)   (기준 45° 평면 z=x에서 h만큼 변형)
  반사광 : +x 방향 → CMOS (x = x_cmos)

VPP + ASM 파이프라인  O(N² log N):
  1. 거울면에서 U_in, R_tilde, U_ref 계산  (me_0318.py 동일)
  2. VPP scatter : (x_i, y_j) → z_vpp = x_i + h(x_i, y_j)
                  비균일 → 정규 격자, Jacobian 보정
  3. Band-limited ASM : VPP 평면 → CMOS 평면
  4. CMOS 좌표로 보간 → I_CMOS
"""

import numpy as np
from scipy.fft import fft2, ifft2, fftfreq
from scipy.interpolate import RegularGridInterpolator
import warnings
import time
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# 거울면 공통 함수  (me_0318.py와 동일 API)
# ─────────────────────────────────────────────────────────────────────────────

def compute_surface_gradients(
    h: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple:
    """표면 기울기 (∂h/∂x, ∂h/∂y) 반환.  h shape = (Ny, Nx)."""
    grad = np.gradient(h, y, x)
    return grad[1], grad[0]   # dh_dx, dh_dy


def compute_U_in(
    X: np.ndarray,
    Y: np.ndarray,
    h: np.ndarray,
    t: float,
    A0: float,
    wavelength: float,
) -> np.ndarray:
    """
    입사 복소장 U_in(x, y).
    me_0318.py와 동일: super-Gaussian 개구 포함.
      위상 = k*(X + h - t)
      진폭 = A0 * exp(-(dx/w)^6) * exp(-(dy/w)^6)
    """
    k = 2.0 * np.pi / wavelength
    phase = k * (X + h - t)

    x_center = (float(X.max()) + float(X.min())) / 2.0
    X_length = float(X.max()) - float(X.min())
    w_r = (X_length / np.sqrt(2.0)) / 2.0
    n = 6

    r = np.sqrt((X - x_center) ** 2 + Y ** 2)
    amp = np.exp(-(r / w_r) ** n)

    return (A0 * amp * np.exp(1j * phase)).astype(complex)


def compute_incident_angle(
    dh_dx: np.ndarray,
    dh_dy: np.ndarray,
) -> np.ndarray:
    """표면 기울기로부터 국소 입사각 θ_loc 계산."""
    norm_n = np.sqrt((1.0 + dh_dx)**2 + dh_dy**2 + 1.0)
    return np.arccos(np.clip(1.0 / norm_n, -1.0, 1.0))


def compute_fresnel_reflection(
    theta_loc: np.ndarray,
    n1: float,
    n2_complex: complex,
) -> np.ndarray:
    """Fresnel s-편광 복소 반사계수 R_tilde."""
    n2    = complex(n2_complex)
    cos_i = np.cos(theta_loc).astype(complex)
    sin_i = np.sin(theta_loc).astype(complex)
    sin_t = (n1 / n2) * sin_i
    cos_t = np.sqrt(1.0 - sin_t**2)
    r_s   = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    return (np.abs(r_s) * np.exp(1j * np.angle(r_s))).astype(complex)


def compute_U_ref(U_in: np.ndarray, R_tilde: np.ndarray) -> np.ndarray:
    """반사 직후 복소장 U_ref = U_in · R_tilde."""
    return (U_in * R_tilde).astype(complex)


# ─────────────────────────────────────────────────────────────────────────────
# Nyquist 검사
# ─────────────────────────────────────────────────────────────────────────────

def _check_nyquist(
    pixel_size: float,
    wavelength: float,
    dh_dx: np.ndarray,
    dh_dy: np.ndarray,
) -> None:
    """
    픽셀 크기 Nyquist 조건 검사.
    위상 k·h 의 공간 변화율이 픽셀로 샘플링 가능한지 확인.
      조건: pixel_size < λ / (2 · max_slope)
    """
    max_slope = max(float(np.abs(dh_dx).max()), float(np.abs(dh_dy).max()))
    if max_slope > 1e-12:
        nyq_limit = wavelength / (2.0 * max_slope)
        if pixel_size >= nyq_limit:
            warnings.warn(
                f"[me_asm] Nyquist 조건 위반: "
                f"pixel_size={pixel_size*1e6:.3f} µm  ≥  "
                f"λ/(2·max_slope)={nyq_limit*1e9:.4f} nm  "
                f"(max_slope={max_slope:.4f})",
                UserWarning,
                stacklevel=3,
            )


# ─────────────────────────────────────────────────────────────────────────────
# VPP scatter: 비균일 격자 → 정규 격자
# ─────────────────────────────────────────────────────────────────────────────

def _vpp_scatter(
    h: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    U_ref: np.ndarray,
    dh_dx: np.ndarray,
) -> tuple:
    """
    VPP 매핑: 비균일 (z_vpp = x + h, y) → 정규 격자 E_vpp(y, z_vpp).

    각 반사면 점 (x_i, y_j):
      반사 빔이 +x 방향으로 진행 → z=x 평면과의 교점 z_vpp = x_i + h(x_i, y_j)

    에너지 보존 (Jacobian 보정):
      dz_vpp/dx = 1 + ∂h/∂x  →  진폭 × |1 + ∂h/∂x|^{-1/2}

    y 방향은 이미 정규 격자이므로, 각 행(j)마다 1D np.interp로 scatter.

    반환
    ────
    E_vpp     : (Ny, Nz_vpp) complex
    z_vpp_grid: (Nz_vpp,)   정규 z_vpp 좌표 [m]
    """
    Ny, Nx = h.shape
    dz = float(x_coords[1] - x_coords[0])

    # 전체 z_vpp 범위 결정 (변형에 따라 x_coords 범위보다 넓어질 수 있음)
    Z_vpp = x_coords[np.newaxis, :] + h          # (Ny, Nx) 비균일
    z_min = float(Z_vpp.min())
    z_max = float(Z_vpp.max())
    Nz_vpp = int(np.ceil((z_max - z_min) / dz)) + 1
    z_vpp_grid = np.arange(Nz_vpp) * dz + z_min  # 정규 격자

    # Jacobian 보정: 진폭 × |1 + dh/dx|^{-1/2}
    jac    = np.abs(1.0 + dh_dx)
    E_sc   = U_ref / np.sqrt(np.maximum(jac, 1e-10))   # (Ny, Nx)

    E_vpp = np.zeros((Ny, Nz_vpp), dtype=complex)

    for j in range(Ny):
        z_src = Z_vpp[j, :]          # 비균일 z 좌표
        vals  = E_sc[j, :]
        idx   = np.argsort(z_src)
        zs    = z_src[idx]
        vs    = vals[idx]
        E_vpp[j, :] = (
            np.interp(z_vpp_grid, zs, vs.real, left=0.0, right=0.0) +
            1j * np.interp(z_vpp_grid, zs, vs.imag, left=0.0, right=0.0)
        )

    return E_vpp, z_vpp_grid


# ─────────────────────────────────────────────────────────────────────────────
# Band-limited ASM 전달함수 + 전파
# ─────────────────────────────────────────────────────────────────────────────

def _asm_transfer(
    Ny: int, Nz: int,
    dy: float, dz: float,
    wavelength: float,
    d_prop: float,
) -> np.ndarray:
    """
    Band-limited ASM 전달함수 H(fy, fz), shape (Ny, Nz).

    H = exp(i·kx·d)   단,  kx = sqrt(k² - (2π·fy)² - (2π·fz)²)

    유효 마스크 (두 조건 모두 만족할 때만 H ≠ 0):
      ① 전파 조건 : fy² + fz² < (1/λ)²           (evanescent 제거)
      ② Nyquist   : |fy| < 1/(2·dy), |fz| < 1/(2·dz)  (aliasing 제거)
    """
    fy = fftfreq(Ny, d=float(dy)).astype(np.float64)
    fz = fftfreq(Nz, d=float(dz)).astype(np.float64)
    FY, FZ = np.meshgrid(fy, fz, indexing='ij')

    k      = 2.0 * np.pi / wavelength
    f_lim2 = (1.0 / wavelength) ** 2
    f2     = FY**2 + FZ**2

    prop_mask = f2 < f_lim2
    nyq_mask  = (np.abs(FY) < 0.5 / float(dy)) & (np.abs(FZ) < 0.5 / float(dz))
    valid     = prop_mask & nyq_mask

    kx2 = k**2 - (2.0 * np.pi)**2 * f2
    kx  = np.where(valid, np.sqrt(np.maximum(kx2, 0.0)), 0.0)
    H   = np.where(valid, np.exp(1j * kx * float(d_prop)), 0.0).astype(complex)
    return H


def _asm_propagate(
    E_vpp: np.ndarray,
    dy: float,
    dz: float,
    wavelength: float,
    d_prop: float,
    pad_factor: int = 2,
) -> np.ndarray:
    """
    Zero-padded band-limited ASM 전파.

    패딩: wrap-around 아티팩트 억제.
    반환: 원본 (Ny, Nz) 크기로 복원된 전파 필드.
    """
    Ny, Nz = E_vpp.shape
    Nyp = Ny * pad_factor
    Nzp = Nz * pad_factor
    y0  = (Nyp - Ny) // 2
    z0  = (Nzp - Nz) // 2

    E_pad = np.zeros((Nyp, Nzp), dtype=complex)
    E_pad[y0:y0+Ny, z0:z0+Nz] = E_vpp

    H     = _asm_transfer(Nyp, Nzp, dy, dz, wavelength, d_prop)
    E_out = ifft2(fft2(E_pad) * H)

    return E_out[y0:y0+Ny, z0:z0+Nz]


# ─────────────────────────────────────────────────────────────────────────────
# Public API  ─  me_0318.forward_propagate 와 동일 인터페이스
# ─────────────────────────────────────────────────────────────────────────────

def forward_propagate_asm(
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
    렌즈 없는 촉각 센서 순전파 (VPP + Band-limited ASM).

    me_0318.forward_propagate() 와 동일한 시그니처 / 반환 dict.

    Parameters
    ──────────
    h               : (Ny, Nx)  표면 높이맵 [m]
    x_coords        : (Nx,)     반사면 x좌표 [m]
    y_coords        : (Ny,)     반사면 y좌표 [m]
    y_prime_coords  : (Ny_c,)   CMOS y좌표 [m]
    z_prime_coords  : (Nz_c,)   CMOS z좌표 [m]
    wavelength      : 파장 [m]
    A0, t           : 진폭, 시간 오프셋
    n1              : 입사 매질 굴절률
    n2_complex      : 반사 매질 복소굴절률
    x_cmos_location : CMOS x위치 [m]
    pad_factor      : ASM zero-padding 배율 (기본 2)

    Returns
    ───────
    dict 키: U_in, theta_loc, R_tilde, U_ref, U_CMOS, I_CMOS, dh_dx, dh_dy
    """
    # ── 격자 ──────────────────────────────────────────────────────────────
    X, Y = np.meshgrid(x_coords, y_coords)
    dy   = float(y_coords[1] - y_coords[0])
    dz   = float(x_coords[1] - x_coords[0])

    # ── Step 1~4: 거울면 물리량 ────────────────────────────────────────
    dh_dx, dh_dy = compute_surface_gradients(h, x_coords, y_coords)
    U_in    = compute_U_in(X, Y, h, t, A0, wavelength)
    theta_l = compute_incident_angle(dh_dx, dh_dy)
    R_tilde = compute_fresnel_reflection(theta_l, n1, n2_complex)
    U_ref   = compute_U_ref(U_in, R_tilde)

    # ── Nyquist 검사 ───────────────────────────────────────────────────
    _check_nyquist(min(dy, dz), wavelength, dh_dx, dh_dy)

    # ── Step 5: VPP scatter ────────────────────────────────────────────
    E_vpp, z_vpp_grid = _vpp_scatter(h, x_coords, y_coords, U_ref, dh_dx)

    # ── Carrier removal ────────────────────────────────────────────────
    # U_ref 위상 ≈ k·z_vpp → 공간주파수 1/λ ≫ Nyquist(1/2·dz)
    # → 이를 제거하고 천천히 변하는 포락선(envelope)만 ASM으로 전파
    k_wave   = 2.0 * np.pi / wavelength
    carrier  = np.exp(-1j * k_wave * z_vpp_grid).astype(complex)   # (Nz_vpp,)
    E_env    = E_vpp * carrier[np.newaxis, :]                       # (Ny, Nz_vpp)

    # ── Step 6: Band-limited ASM 전파 (baseband) ──────────────────────
    x_ref  = float(x_coords.mean())
    d_prop = float(x_cmos_location) - x_ref
    E_prop_env = _asm_propagate(E_env, dy, dz, wavelength, d_prop, pad_factor)

    # ── Step 7: CMOS 좌표 보간 ─────────────────────────────────────────
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
    # meshgrid: z가 행(axis-0), y가 열(axis-1) → shape (Nz_c, Ny_c)
    Zq, Yq = np.meshgrid(z_prime_coords, y_prime_coords, indexing='ij')
    pts    = np.column_stack([Yq.ravel(), Zq.ravel()])
    E_cmos_env = (itp_re(pts) + 1j * itp_im(pts)).reshape(Nz_c, Ny_c).astype(complex)

    # carrier 복원: CMOS 좌표에서 exp(i·k·z') 추가
    cmos_carrier = np.exp(1j * k_wave * z_prime_coords).astype(complex)  # (Nz_c,)
    U_CMOS = E_cmos_env * cmos_carrier[:, np.newaxis]                    # (Nz_c, Ny_c)

    return {
        'U_in':      U_in,
        'theta_loc': theta_l,
        'R_tilde':   R_tilde,
        'U_ref':     U_ref,
        'U_CMOS':    U_CMOS,
        'I_CMOS':    np.abs(U_CMOS)**2,
        'dh_dx':     dh_dx,
        'dh_dy':     dh_dy,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch differentiable forward model  (역문제용)
# ─────────────────────────────────────────────────────────────────────────────

def forward_propagate_torch(
    h_tensor,           # torch.Tensor (Ny, Nx), requires_grad=True
    x_coords:  np.ndarray,
    y_coords:  np.ndarray,
    y_prime_coords: np.ndarray,
    z_prime_coords: np.ndarray,
    wavelength: float,
    A0: float,
    t: float,
    n1: float,
    n2_complex: complex,
    x_cmos_location: float,
    pad_factor: int = 2,
):
    """
    VPP + Band-limited ASM 순전파 – PyTorch autograd 완전 지원.

    h_tensor.requires_grad=True 로 설정하면 I_CMOS까지 gradient 흐름.
    역문제: optimizer.zero_grad(); loss.backward(); optimizer.step()

    반환
    ────
    I_CMOS : torch.Tensor (Nz_c, Ny_c)  – 미분 가능 세기맵
    """
    import torch

    device = h_tensor.device
    dtype  = torch.float64

    # numpy 배열 → torch (no grad)
    def t64(arr):
        return torch.tensor(arr, dtype=dtype, device=device)

    xc = t64(x_coords)    # (Nx,)
    yc = t64(y_coords)    # (Ny,)
    yp = t64(y_prime_coords)
    zp = t64(z_prime_coords)

    Ny, Nx = h_tensor.shape
    dy = float(y_coords[1] - y_coords[0])
    dz = float(x_coords[1] - x_coords[0])

    # ── Step 1: meshgrid ──────────────────────────────────────────────
    X, Y = torch.meshgrid(xc, yc, indexing='xy')   # (Ny, Nx)

    # ── Step 2: U_in ──────────────────────────────────────────────────
    k = 2.0 * torch.pi / wavelength
    phase = k * (X + h_tensor - t)
    x_center = float((x_coords[0] + x_coords[-1]) / 2)
    X_len    = float(x_coords[-1] - x_coords[0])
    w_r      = (X_len / (2.0 ** 0.5)) / 2.0
    n_exp    = 6
    r_t   = torch.sqrt((X - x_center) ** 2 + Y ** 2)
    amp   = torch.exp(-(r_t / w_r) ** n_exp)
    U_in_re = A0 * amp * torch.cos(phase)
    U_in_im = A0 * amp * torch.sin(phase)

    # ── Step 3: Fresnel R_tilde ───────────────────────────────────────
    # dh/dx via central diff (autograd friendly)
    # pad edges with replicate
    h_pad = torch.nn.functional.pad(
        h_tensor.unsqueeze(0).unsqueeze(0),
        (1, 1, 0, 0), mode='replicate').squeeze()  # (Ny, Nx+2)
    dh_dx = (h_pad[:, 2:] - h_pad[:, :-2]) / (2.0 * dz)  # (Ny, Nx)

    h_pad_y = torch.nn.functional.pad(
        h_tensor.unsqueeze(0).unsqueeze(0),
        (0, 0, 1, 1), mode='replicate').squeeze()  # (Ny+2, Nx)
    dh_dy = (h_pad_y[2:, :] - h_pad_y[:-2, :]) / (2.0 * dy)  # (Ny, Nx)

    norm_n   = torch.sqrt((1.0 + dh_dx)**2 + dh_dy**2 + 1.0)
    cos_i    = torch.clamp(1.0 / norm_n, -1.0, 1.0)
    sin_i    = torch.sqrt(torch.clamp(1.0 - cos_i**2, min=0.0))

    n2r = float(n2_complex.real)
    n2i = float(n2_complex.imag)
    # n2 = n2r + i*n2i (complex)
    # sin_t = (n1/n2)*sin_i  → complex
    # r_s   = (n1*cos_i - n2*cos_t)/(n1*cos_i + n2*cos_t)
    # For real n2i the formula becomes complex; compute |r_s| and angle(r_s) numerically
    # We use the numpy result as a constant tensor (grad does NOT flow through R_tilde)
    with torch.no_grad():
        dh_dx_np = dh_dx.detach().cpu().numpy()
        dh_dy_np = dh_dy.detach().cpu().numpy()
        th_np    = np.arctan2(
            np.sqrt(dh_dx_np**2 + dh_dy_np**2),
            1.0 / np.sqrt((1.0+dh_dx_np)**2 + dh_dy_np**2 + 1.0 + 1e-30))
        R_np     = compute_fresnel_reflection(th_np, n1, n2_complex)
        R_re_t   = t64(R_np.real)
        R_im_t   = t64(R_np.imag)

    # ── Step 4: U_ref = U_in * R_tilde ───────────────────────────────
    # (a+ib)*(c+id) = (ac-bd) + i(ad+bc)
    Ur_re = U_in_re * R_re_t - U_in_im * R_im_t
    Ur_im = U_in_re * R_im_t + U_in_im * R_re_t

    # ── Step 5: VPP scatter (differentiable linear interp) ───────────
    Z_vpp = X + h_tensor          # (Ny, Nx) - z_vpp per point
    z_min = float(xc[0].item())
    z_max_val = float((xc[-1] + h_tensor.detach().abs().max()).item()) + abs(float(h_tensor.detach().min()))
    # safer: use full range
    z_vpp_min = float((Z_vpp.detach()).min().item())
    z_vpp_max = float((Z_vpp.detach()).max().item())
    Nz_vpp = int(np.ceil((z_vpp_max - z_vpp_min) / dz)) + 1
    z_vpp_grid = torch.arange(Nz_vpp, dtype=dtype, device=device) * dz + z_vpp_min

    jac   = torch.abs(1.0 + dh_dx)
    E_sc_re = Ur_re / torch.sqrt(torch.clamp(jac, min=1e-10))
    E_sc_im = Ur_im / torch.sqrt(torch.clamp(jac, min=1e-10))

    E_vpp_re = torch.zeros((Ny, Nz_vpp), dtype=dtype, device=device)
    E_vpp_im = torch.zeros((Ny, Nz_vpp), dtype=dtype, device=device)

    # Continuous fractional index in z_vpp_grid
    t_idx = (Z_vpp - z_vpp_grid[0]) / dz   # (Ny, Nx) float
    idx_l = torch.clamp(t_idx.long(), 0, Nz_vpp - 2)
    idx_r = idx_l + 1
    w_r_  = t_idx - idx_l.to(dtype)         # fraction
    w_l_  = 1.0 - w_r_

    # Row-by-row index_put_ (autograd through weights w_l_, w_r_)
    for j in range(Ny):
        il = idx_l[j]   # (Nx,)
        ir = idx_r[j]
        wl = w_l_[j]
        wr = w_r_[j]
        E_vpp_re[j].index_put_((il,), wl * E_sc_re[j], accumulate=True)
        E_vpp_re[j].index_put_((ir,), wr * E_sc_re[j], accumulate=True)
        E_vpp_im[j].index_put_((il,), wl * E_sc_im[j], accumulate=True)
        E_vpp_im[j].index_put_((ir,), wr * E_sc_im[j], accumulate=True)

    # ── Carrier removal ───────────────────────────────────────────────
    carrier_re = torch.cos(k * z_vpp_grid)   # (Nz_vpp,)
    carrier_im = -torch.sin(k * z_vpp_grid)
    # E_env = E_vpp * conj(exp(ik·z)) = E_vpp * (cos - i*sin)
    Env_re = E_vpp_re * carrier_re[None, :] - E_vpp_im * carrier_im[None, :]
    Env_im = E_vpp_re * carrier_im[None, :] + E_vpp_im * carrier_re[None, :]

    # ── Step 6: Band-limited ASM (torch.fft) ─────────────────────────
    Nyp = Ny * pad_factor
    Nzp = Nz_vpp * pad_factor
    y0  = (Nyp - Ny) // 2
    z0  = (Nzp - Nz_vpp) // 2

    E_pad = torch.zeros((Nyp, Nzp), dtype=torch.complex128, device=device)
    E_pad[y0:y0+Ny, z0:z0+Nz_vpp] = torch.complex(Env_re, Env_im)

    # Transfer function H (numpy → torch, no grad needed)
    H_np = _asm_transfer(Nyp, Nzp, dy, dz, wavelength,
                         float(x_cmos_location) - float(xc.mean()))
    H_t  = torch.tensor(H_np, dtype=torch.complex128, device=device)

    E_prop = torch.fft.ifft2(torch.fft.fft2(E_pad) * H_t)
    E_prop = E_prop[y0:y0+Ny, z0:z0+Nz_vpp]  # (Ny, Nz_vpp)

    # ── Step 7: Bilinear interpolation at CMOS coords ─────────────────
    Nz_c = len(z_prime_coords)
    Ny_c = len(y_prime_coords)

    # Normalize query coordinates to [0, Nz_vpp-1] and [0, Ny-1]
    # z-axis: z_vpp_grid, y-axis: y_coords
    z_query = zp   # (Nz_c,)
    y_query = yp   # (Ny_c,)

    # Fractional indices
    zi = (z_query - z_vpp_grid[0]) / dz            # (Nz_c,)
    yi = (y_query - t64(y_coords)[0]) / float(dy)  # (Ny_c,)

    zi0 = torch.clamp(zi.long(), 0, Nz_vpp - 2)
    zi1 = zi0 + 1
    yi0 = torch.clamp(yi.long(), 0, Ny - 2)
    yi1 = yi0 + 1

    wz1 = (zi - zi0.to(dtype)).clamp(0, 1)   # (Nz_c,)
    wz0 = 1.0 - wz1
    wy1 = (yi - yi0.to(dtype)).clamp(0, 1)   # (Ny_c,)
    wy0 = 1.0 - wy1

    # E_prop shape: (Ny, Nz_vpp) complex
    # Output shape: (Nz_c, Ny_c)
    # Gather 4 corners for bilinear
    def g(yi_, zi_):
        # clamp
        yi_c = torch.clamp(yi_, 0, Ny - 1)
        zi_c = torch.clamp(zi_, 0, Nz_vpp - 1)
        return E_prop[yi_c[:, None], zi_c[None, :]]   # (Ny_c, Nz_c)

    E00 = g(yi0, zi0)   # (Ny_c, Nz_c)
    E01 = g(yi0, zi1)
    E10 = g(yi1, zi0)
    E11 = g(yi1, zi1)

    # Bilinear: weight y then z
    E_cmos = (wy0[:, None] * (wz0[None, :] * E00 + wz1[None, :] * E01) +
              wy1[:, None] * (wz0[None, :] * E10 + wz1[None, :] * E11))
    # E_cmos: (Ny_c, Nz_c) → transpose → (Nz_c, Ny_c)
    E_cmos = E_cmos.T

    # ── Carrier restoration at CMOS ───────────────────────────────────
    cmos_re = torch.cos(k * zp)   # (Nz_c,)
    cmos_im = torch.sin(k * zp)
    cmos_c  = torch.complex(cmos_re, cmos_im)
    U_CMOS  = E_cmos * cmos_c[:, None]   # (Nz_c, Ny_c)

    I_CMOS = U_CMOS.abs() ** 2   # (Nz_c, Ny_c) – differentiable

    return I_CMOS


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    import matplotlib
    matplotlib.use('Agg')

    from sanity_utils import plot_3x3, STYLE
    import matplotlib.pyplot as plt

    # ── 파라미터 (task spec) ────────────────────────────────────────────────
    wavelength  = 500e-9
    A0          = 1.0
    t           = 0.0
    n1          = 1.0
    n2_complex  = complex(0.97112, 1.8737)

    pixel_size  = 3e-6
    N           = 256

    x_coords = np.linspace(1, N, N) * pixel_size
    y_coords = np.linspace(-N//2, N//2 - 1, N) * pixel_size

    X_tmp, Y_tmp = np.meshgrid(x_coords, y_coords)

    h_amplitude = -5.0 * wavelength    # ~2.5 µm (위상 변화 ~10π)
    h_sigma     = 30.0 * pixel_size   # 90 µm
    h_cx        = x_coords[N // 2]
    h_cy        = 0.0

    h_def = h_amplitude * np.exp(
        -((X_tmp - h_cx)**2 + (Y_tmp - h_cy)**2) / (2.0 * h_sigma**2)
    )
    h_ref = np.zeros_like(h_def)

    x_cmos_location = x_coords[-1] + 5e-3
    N_cmos          = 256
    pixel_size_cmos = 3e-6

    # CMOS z좌표: 거울 x 중심 주위에 정렬 (me_0318.py 관례)
    z_center       = float(x_coords.mean())
    y_prime_coords = np.linspace(-N_cmos // 2, N_cmos // 2 - 1, N_cmos) * pixel_size_cmos
    z_prime_coords = (np.linspace(-N_cmos // 2, N_cmos // 2 - 1, N_cmos)
                      * pixel_size_cmos + z_center)

    out_dir = Path('.')

    # ── N=256 풀 런 ─────────────────────────────────────────────────────────
    print('=' * 60)
    print(f'VPP+ASM  N={N}x{N}  pad=2')
    print('=' * 60)

    results = {}
    for label, hh in [('ref', h_ref), ('def', h_def)]:
        t0  = time.perf_counter()
        res = forward_propagate_asm(
            hh, x_coords, y_coords, y_prime_coords, z_prime_coords,
            wavelength, A0, t, n1, n2_complex, x_cmos_location,
            pad_factor=4,
        )
        dt = time.perf_counter() - t0
        I  = res['I_CMOS']
        print(f'  [{label}]  I_max={I.max():.3e}  CoV={I.std()/(I.mean()+1e-30):.4f}'
              f'  {dt:.2f}s')
        results[label] = res

    plt.rcParams.update(STYLE)
    out_path = out_dir / 'result_asm.png'
    plot_3x3(
        title='VPP + Band-limited ASM',
        h_ref_nm=h_ref * 1e9,
        h_def_nm=h_def * 1e9,
        x_coords=x_coords,
        y_coords=y_coords,
        I_ref=results['ref']['I_CMOS'],
        I_def=results['def']['I_CMOS'],
        U_ref_cmos=results['ref']['U_CMOS'],
        U_def_cmos=results['def']['U_CMOS'],
        y_prime=y_prime_coords,
        z_prime=z_prime_coords,
        out_path=out_path,
    )
    print(f'\n  saved → {out_path}')

    # ── N=64 소규모: RS vs ASM 비교 ─────────────────────────────────────────
    print('\n' + '=' * 60)
    print('RS vs ASM 비교  N=64  (RS는 수 분 소요될 수 있음)')
    print('=' * 60)

    try:
        from me_0318 import forward_propagate as fp_rs
    except ImportError:
        print('  [경고] me_0318.py를 import할 수 없어 RS 비교를 건넜습니다.')
        sys.exit(0)

    N64        = 64
    ps64       = 3e-6
    xc64       = np.linspace(1, N64, N64) * ps64
    yc64       = np.linspace(-N64 // 2, N64 // 2 - 1, N64) * ps64
    X64, Y64   = np.meshgrid(xc64, yc64)
    h_cx64     = xc64[N64 // 2]

    h_def64 = (-wavelength / 8.0) * np.exp(
        -((X64 - h_cx64)**2 + Y64**2) / (2.0 * (5 * ps64)**2)
    )

    x_cmos64   = xc64[-1] + 10e-3
    Nc64       = 64
    zc64       = float(xc64.mean())
    yp64       = np.linspace(-Nc64 // 2, Nc64 // 2 - 1, Nc64) * ps64
    zp64       = np.linspace(-Nc64 // 2, Nc64 // 2 - 1, Nc64) * ps64 + zc64

    # ASM
    t0 = time.perf_counter()
    res_asm64 = forward_propagate_asm(
        h_def64, xc64, yc64, yp64, zp64,
        wavelength, A0, t, n1, n2_complex, x_cmos64,
        pad_factor=2,
    )
    print(f'  ASM  : {time.perf_counter()-t0:.2f}s')

    # RS
    t0 = time.perf_counter()
    res_rs64 = fp_rs(
        h_def64, xc64, yc64, yp64, zp64,
        wavelength, A0, t, n1, n2_complex, x_cmos64,
    )
    print(f'  RS   : {time.perf_counter()-t0:.1f}s')

    I_asm = res_asm64['I_CMOS']
    I_rs  = res_rs64['I_CMOS']

    denom     = np.maximum(I_rs.max(), 1e-30)
    rel_err   = np.abs(I_asm - I_rs) / denom
    print(f'\n  I_CMOS 상대 오차:')
    print(f'    mean  = {rel_err.mean():.4f}')
    print(f'    max   = {rel_err.max():.4f}')
    print(f'    RMSE  = {np.sqrt(np.mean((I_asm - I_rs)**2)) / denom:.4f}')
