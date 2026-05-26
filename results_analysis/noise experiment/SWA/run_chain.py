"""串行：SWA irreg → SWA 噪声实验(σ=2,5,10 × rect+irreg)"""
import subprocess, sys, os, time

BASE = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    ('SWA irreg', os.path.join(BASE, '..', '补充实验', 'run_experiment.py'),
     {'EXP_MODE': 'swa', 'EXP_REGION': 'irreg'}),
    ('SWA Noise', os.path.join(BASE, 'run_swa_noise.py'), {}),
]

t0 = time.time()
for label, script, env_extra in STEPS:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    env = os.environ.copy()
    env.update(env_extra)
    ret = subprocess.run([sys.executable, script], env=env, cwd=os.path.dirname(script))
    if ret.returncode != 0:
        print(f"\n  ERROR: {label} failed (exit {ret.returncode})")
        sys.exit(1)
    print(f"\n  {label} done! ({ (time.time()-t0)/60:.0f} min)")

print(f"\n{'='*60}")
print(f"  All done! Total: {(time.time()-t0)/60:.0f} min")
print(f"{'='*60}")
