"""run_all.py — run all tilted_asm_sanity checks in sequence"""
import sys
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent))

run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
out_dir  = Path(__file__).parent / 'results' / run_name
out_dir.mkdir(parents=True, exist_ok=True)
print(f"Results -> {out_dir}")

from sanity_01_flat_mirror        import run as r01
from sanity_02_near_field_energy  import run as r02
from sanity_03_far_field          import run as r03
from sanity_04_deformation_patterns import run as r04
from sanity_05_height_animation   import run as r05
from sanity_06_field_validation   import run as r06

r01(out_dir)
r02(out_dir)
r03(out_dir)
r04(out_dir)
r05(out_dir)
r06(out_dir)

print(f"\nAll done -> {out_dir}")
