#!/bin/bash
# GPU detection and allocation utilities

get_gpu_count() {
    python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0"
}

get_free_gpus() {
    python3 -c "
import subprocess, re
result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True)
free = []
for line in result.stdout.strip().split('\n'):
    if line.strip():
        parts = line.split(',')
        idx = int(parts[0].strip())
        mem = int(parts[1].strip())
        if mem < 1000:
            free.append(str(idx))
print(','.join(free))
" 2>/dev/null || echo ""
}

print_gpu_info() {
    python3 -c "
import torch
n = torch.cuda.device_count()
print(f'Available GPUs: {n}')
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {p.name} ({p.total_memory / 1e9:.1f}GB)')
"
}
