"""
sanity_06_gradient_descent.py  (me_asm)
역문제 데모: I_CMOS(측정) → h(높이맵) 복원  via Adam gradient descent

파이프라인
──────────
1. "정답" h_true (Gaussian) 로 I_target 생성  (numpy forward)
2. h_est = zeros 초기화, requires_grad=True
3. Adam으로 loss = MSE(forward_torch(h_est), I_target) 최소화
4. 수렴 곡선 + 복원 결과 저장

메모리 절약: N=256, pad_factor=2 (작은 grid로 개념 검증)
"""
import sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(1, str(Path(__file__).parent.parent))

import torch
from me_asm import forward_propagate_asm, forward_propagate_torch

# ── 파라미터 (작은 그리드로 빠른 검증) ────────────────────────────────────
wavelength  = 500e-9
A0          = 1.0
t_time      = 0.0
n1          = 1.0
n2_complex  = complex(0.97112, 1.8737)

N          = 256
pixel_size = 3e-6
x_coords   = np.linspace(1, N, N) * pixel_size
y_coords   = np.linspace(-(N-1)/2, (N-1)/2, N) * pixel_size

N_cmos     = 64
r_width    = N * pixel_size
x_center   = (x_coords[0] + x_coords[-1]) / 2
z_center   = x_center
x_cmos     = x_center + 10e-3
y_prime    = np.linspace(-r_width/2, r_width/2, N_cmos)
z_prime    = np.linspace(z_center - r_width/2, z_center + r_width/2, N_cmos)

# ── 정답 h_true ──────────────────────────────────────────────────────────
# Nyquist: max_slope < lambda/(2*pixel_size) = 500nm/(2*3um) = 0.083
# Gaussian slope: 0.607 * |h_amp| / sigma < 0.083
# => sigma > 7.3 * |h_amp|
# h_amp = -5*lambda ~ 2.5um, sigma=50*pixel_size=150um → slope=0.607*2.5/150=0.010 (safe)
sigma   = 50 * pixel_size    # 150 um
h_amp   = -5 * wavelength    # ~2.5 um
X_tmp, Y_tmp = np.meshgrid(x_coords, y_coords)
h_true  = h_amp * np.exp(
    -((X_tmp - x_coords[N//2])**2 + Y_tmp**2) / (2*sigma**2))


def run(out_dir: Path):
    print("[06] Gradient descent demo ...", flush=True)

    # ── Step 1: I_target 생성 (numpy) ────────────────────────────────────
    print("  generating I_target (numpy forward) ...", flush=True)
    res_true = forward_propagate_asm(
        h_true, x_coords, y_coords, y_prime, z_prime,
        wavelength, A0, t_time, n1, n2_complex, x_cmos, pad_factor=2)
    I_target_np = res_true['I_CMOS']   # (Nz_c, Ny_c)

    device = torch.device('cpu')
    I_target = torch.tensor(I_target_np, dtype=torch.float64, device=device)

    # ── Step 2: 초기 추정값 ──────────────────────────────────────────────
    h_est = torch.zeros((N, N), dtype=torch.float64,
                        device=device, requires_grad=True)

    # ── Step 3: Adam 최적화 ───────────────────────────────────────────────
    # Normalize I_target to [0,1] for better conditioning
    I_scale   = float(I_target.max()) + 1e-30
    I_tgt_n   = I_target / I_scale

    lam_reg   = 1e-8   # very light L2 regularisation (stability only)
    lr        = 1e-8
    optimizer = torch.optim.Adam([h_est], lr=lr)
    n_iter    = 500
    losses    = []

    print(f"  Adam  lr={lr}  reg={lam_reg}  {n_iter} iters ...", flush=True)
    for i in range(n_iter):
        optimizer.zero_grad()
        I_pred = forward_propagate_torch(
            h_est, x_coords, y_coords, y_prime, z_prime,
            wavelength, A0, t_time, n1, n2_complex, x_cmos, pad_factor=2)
        I_pred_n  = I_pred / I_scale
        data_loss = torch.mean((I_pred_n - I_tgt_n) ** 2)
        reg_loss  = lam_reg * torch.mean(h_est ** 2)
        loss = data_loss + reg_loss
        loss.backward()
        optimizer.step()
        losses.append(float(data_loss.detach()))
        if (i+1) % 50 == 0:
            print(f"    iter {i+1:4d}  data_loss={losses[-1]:.4e}", flush=True)

    h_est_np = h_est.detach().numpy()

    # ── Step 4: 결과 시각화 ───────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('ASM Sanity 06 – Gradient Descent  (Inverse Problem)', fontsize=13)

    ext_mirror = [x_coords[0]*1e3, x_coords[-1]*1e3,
                  y_coords[-1]*1e3, y_coords[0]*1e3]
    ext_cmos   = [z_prime[0]*1e3, z_prime[-1]*1e3,
                  y_prime[-1]*1e3, y_prime[0]*1e3]

    # 행 0: 높이맵 비교
    im0 = axes[0, 0].imshow(h_true*1e6, extent=ext_mirror,
                             aspect='auto', cmap='RdBu')
    axes[0, 0].set_title('h_true [µm]')
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(h_est_np*1e6, extent=ext_mirror,
                             aspect='auto', cmap='RdBu')
    axes[0, 1].set_title('h_est [µm]')
    plt.colorbar(im1, ax=axes[0, 1])

    err = h_est_np - h_true
    im2 = axes[0, 2].imshow(err*1e6, extent=ext_mirror,
                             aspect='auto', cmap='RdBu')
    axes[0, 2].set_title('error [µm]')
    plt.colorbar(im2, ax=axes[0, 2])

    # 행 1: I_CMOS 비교 + loss 곡선
    vmax = I_target_np.max()
    axes[1, 0].imshow(I_target_np, extent=ext_cmos,
                      aspect='auto', cmap='hot', vmin=0, vmax=vmax)
    axes[1, 0].set_title('I_target (measured)')
    axes[1, 0].set_xlabel('z\' [mm]'); axes[1, 0].set_ylabel('y\' [mm]')

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)  # h_est slope 경고 (역문제 비유일성)
        I_pred_np = forward_propagate_asm(
            h_est_np, x_coords, y_coords, y_prime, z_prime,
            wavelength, A0, t_time, n1, n2_complex, x_cmos, pad_factor=2)['I_CMOS']
    axes[1, 1].imshow(I_pred_np, extent=ext_cmos,
                      aspect='auto', cmap='hot', vmin=0, vmax=vmax)
    axes[1, 1].set_title('I_pred (reconstructed)')
    axes[1, 1].set_xlabel('z\' [mm]')

    axes[1, 2].semilogy(losses)
    axes[1, 2].set_title('Loss curve')
    axes[1, 2].set_xlabel('Iteration'); axes[1, 2].set_ylabel('MSE loss')
    axes[1, 2].grid(True, which='both', alpha=0.4)

    plt.tight_layout()
    out_path = out_dir / 'sanity_06_gradient_descent.png'
    plt.savefig(str(out_path), dpi=100)
    plt.close(fig)
    print(f"  saved → {out_path}", flush=True)

    # 수치 요약
    rel_err = float(np.abs(err).max()) / float(np.abs(h_true).max() + 1e-30)
    print(f"  h max abs error : {np.abs(err).max()*1e6:.3f} um")
    print(f"  h relative error: {rel_err*100:.2f}%")
    print(f"  final loss      : {losses[-1]:.4e}", flush=True)


if __name__ == '__main__':
    from datetime import datetime
    run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(__file__).parent / 'results' / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    run(out_dir)
