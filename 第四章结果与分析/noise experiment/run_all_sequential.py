#!/usr/bin/env python
"""噪声鲁棒性实验串行执行器 — CNN → GAN → U-Net+TF
====================================================
每完成一个方法自动推进下一个，记录总进度
"""
import subprocess, sys, os, json, time

ROOT = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ('CNN', os.path.join(ROOT, 'CNN', 'run_cnn_noise.py')),
    ('GAN', os.path.join(ROOT, 'GAN', 'run_gan_noise.py')),
    ('U-Net+Transformer', os.path.join(ROOT, 'U-net_TF', 'run_unet_tf_noise.py')),
]

PROGRESS_FILE = os.path.join(ROOT, 'master_progress.json')
PYTHON = sys.executable

print("=" * 70)
print("  噪声鲁棒性实验 — 串行执行")
print(f"  Kriging: 已完成")
print(f"  CNN → GAN → U-Net+Transformer")
print(f"  开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

master_progress = {
    'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
    'steps': {},
    'status': 'running',
}

for step_name, script_path in STEPS:
    step_start = time.time()
    print(f"\n{'#'*70}")
    print(f"#  开始: {step_name}")
    print(f"#  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")

    result = subprocess.run(
        [PYTHON, script_path],
        cwd=os.path.dirname(script_path),
        capture_output=False,
    )

    elapsed = time.time() - step_start
    if result.returncode == 0:
        print(f"\n  ✓ {step_name} 完成! 耗时: {elapsed/60:.1f} min")
        master_progress['steps'][step_name] = {
            'status': 'completed',
            'time_min': round(elapsed / 60, 1),
            'returncode': 0,
        }
    else:
        print(f"\n  ✗ {step_name} 失败! (exit {result.returncode}), 耗时: {elapsed/60:.1f} min")
        master_progress['steps'][step_name] = {
            'status': 'failed',
            'time_min': round(elapsed / 60, 1),
            'returncode': result.returncode,
        }
        master_progress['status'] = 'failed'
        # Save progress and stop
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(master_progress, f, indent=2, ensure_ascii=False)
        print(f"\n  实验中断于 {step_name}, 进度已保存至 {PROGRESS_FILE}")
        sys.exit(1)

    # Save incremental progress
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(master_progress, f, indent=2, ensure_ascii=False)

master_progress['status'] = 'completed'
master_progress['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
with open(PROGRESS_FILE, 'w') as f:
    json.dump(master_progress, f, indent=2, ensure_ascii=False)

print(f"\n{'='*70}")
print(f"  全部实验完成!")
print(f"  进度文件: {PROGRESS_FILE}")
print(f"{'='*70}")
