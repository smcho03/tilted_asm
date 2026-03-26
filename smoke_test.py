"""Quick smoke test for me_tilted_asm.py"""
import numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from me_tilted_asm import forward_propagate_B, forward_propagate_C
from me_asm import forward_propagate_asm

N = 64; ps = 10e-6
x = np.linspace(1, N, N) * ps
y = np.linspace(-(N-1)/2, (N-1)/2, N) * ps
h = np.zeros((N, N))
x_cmos = x.mean() + 5e-3
yp = np.linspace(-3e-4, 3e-4, 32)
zp = np.linspace(x.mean()-3e-4, x.mean()+3e-4, 32)

kw = dict(wavelength=500e-9, A0=1, t=0, n1=1, n2_complex=complex(0.97,1.87),
          x_cmos_location=x_cmos, pad_factor=2)

print('Testing VPP...', end=' ', flush=True)
rv = forward_propagate_asm(h, x, y, yp, zp, **kw)
print(f'I_CMOS max={rv["I_CMOS"].max():.4e}  OK')

print('Testing Approach B...', end=' ', flush=True)
rb = forward_propagate_B(h, x, y, yp, zp, **kw)
print(f'I_CMOS max={rb["I_CMOS"].max():.4e}  OK')

print('Testing Approach C...', end=' ', flush=True)
rc = forward_propagate_C(h, x, y, yp, zp, **kw)
print(f'I_CMOS max={rc["I_CMOS"].max():.4e}  OK')

print('Smoke test PASSED')
