"""
run_all_sanity.py
─────────────────
모든 sanity check 실행 → sanity_results/{timestamp}/ 에 저장

실행: python run_all_sanity.py
"""
import sys, time, traceback
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent))

timestamp = datetime.now().strftime("%m%d_%H%M%S")
out_dir   = Path(__file__).parent / 'sanity_results' / timestamp
out_dir.mkdir(parents=True, exist_ok=True)

def section(msg):
    print(f"\n{'='*52}\n  {msg}\n{'='*52}", flush=True)

def run_safe(name, func):
    print(f"\n[{name}] 실행 중 ...", flush=True)
    t0 = time.time()
    try:
        func(out_dir)
        print(f"[{name}] 완료  ({time.time()-t0:.1f}s)", flush=True)
    except Exception:
        print(f"[{name}] ERROR:", flush=True)
        traceback.print_exc()

section("Sanity Check – All Tests")
print(f"  output -> {out_dir}", flush=True)

from sanity_01_flat_mirror          import run as run_01
from sanity_03_far_field            import run as run_03
from sanity_04_deformation_patterns import run as run_04

t_total = time.time()
run_safe('01 flat_mirror',          run_01)
run_safe('03 far_field',            run_03)
run_safe('04 deformation_patterns', run_04)

section("완료")
files = sorted(out_dir.glob('*.png'))
print(f"  총 소요: {time.time()-t_total:.1f}s", flush=True)
print(f"  저장 파일 ({len(files)}개):", flush=True)
for f in files:
    print(f"    {f.name}", flush=True)