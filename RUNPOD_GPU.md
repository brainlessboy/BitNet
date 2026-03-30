# RunPod GPU Setup for BitNet Distillation

## Prerequisites

- RunPod account (runpod.io) with credits loaded ($25 minimum)
- `runpodctl` CLI installed for file transfers

### Generate SSH Key

```bash
ssh-keygen -t ed25519
```

Copy your public key and paste into RunPod dashboard > **Settings** > **SSH Public Keys**:
```bash
cat ~/.ssh/id_ed25519.pub
```

### Install runpodctl

```bash
brew install runpod/runpodctl/runpodctl
runpodctl config --apiKey <YOUR_API_KEY>
```

## 1. Create a Pod

Via the RunPod dashboard (runpod.io):
1. Click **Deploy** > **GPU Pod**
2. Select GPU: **H100 SXM 80GB** (or A100 80GB)
3. Template: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (torch + CUDA pre-installed)
4. **Container disk: 70-80GB** (default 20GB is too small — models + packages won't fit)
5. Volume: **50GB** persistent disk (mounts at `/workspace`, used for final results only)
6. Click **Deploy**

**Important**: Use a PyTorch template so torch is pre-installed. Never pip install torch on the pod. The large container disk avoids needing the slow `/workspace` network storage for day-to-day operations.

## 2. Transfer Code to Pod

Package locally (on your Mac):
```bash
tar czf /tmp/bitnet-code.tar.gz \
  --exclude='.git' --exclude='models' --exclude='build' \
  --exclude='distill/checkpoints' --exclude='__pycache__' \
  -C ~/Documents/BitNet .
```

Transfer via runpodctl:
```bash
runpodctl send /tmp/bitnet-code.tar.gz
```

On the pod, receive it:
```bash
runpodctl receive <CODE>
```

## 3. SSH into the Pod

Get the SSH command from the pod's page on the RunPod dashboard:
```bash
ssh <pod-id>@ssh.runpod.io -i ~/.ssh/id_ed25519
```

**Note**: RunPod's SSH proxy is interactive only (no rsync, no remote commands).

## 4. Set Up Environment

**Key principle**: Everything on local container disk (`/root`). The `/workspace` volume is network-attached and painfully slow for pip installs, model downloads, and general I/O. Only use `/workspace` to copy final results for persistence.

```bash
# 1. Unpack code to local disk
mkdir -p /root/BitNet && cd /root/BitNet && tar xzf ~/bitnet-code.tar.gz

# 2. Create venv with system site packages (inherits pre-installed torch)
python3 -m venv --system-site-packages venv
source venv/bin/activate

# 3. Install remaining packages (torch already in template)
pip install transformers datasets safetensors

# 4. Verify
python3 -c "import torch; print(torch.cuda.is_available())"
nvidia-smi
```

## 5. Start Distillation

### Recommended Parameters (H100 SXM 80GB)

| Parameter | 0.5B (validation) | 3B Model | 7B Model | Why |
|-----------|-------------------|----------|----------|-----|
| `--batch_size` | 8 | 4 | 2 | Largest that fits in VRAM |
| `--accumulation_steps` | 4 | 8 | 16 | Effective batch 32 |
| `--max_length` | 256 | 1024 | 512 | Longer = better quality (drop if OOM) |
| `--max_steps` | 5000 | 10000 | 10000 | Ternary weights need many steps |
| `--lr` | 5e-4 | 5e-4 | 5e-4 | Higher than default for ternary |
| `--tau` | 2.0 | 2.0 | 2.0 | Sharper teacher signal |
| `--save_every` | 500 | 500 | 500 | Checkpoints are ~12GB each |

**Key insights**:
- `tau=2.0` >> `tau=5.0` — sharper distillation signal works much better through ternary bottleneck
- `lr=5e-4` >> `lr=1e-4` — ternary weights need stronger gradient signal
- Always validate pipeline with 0.5B first (~45 min) before expensive 3B/7B runs

### 0.5B Validation Run (H100, ~45 min — do this first!)
```bash
cd /root/BitNet && nohup python3 distill/distill.py \
  --device cuda \
  --teacher_model Qwen/Qwen2.5-0.5B \
  --dataset alpaca \
  --batch_size 8 \
  --accumulation_steps 4 \
  --max_length 256 \
  --max_steps 5000 \
  --lr 5e-4 \
  --tau 2.0 \
  --save_every 500 \
  > distill_log.txt 2>&1 &
```

Test checkpoints to verify quality before investing in longer runs:
```bash
python distill/test_checkpoint.py distill/checkpoints/step_1000.pt -n 50 -t 0.7 -r 1.3
python distill/inspect_logits.py distill/checkpoints/step_1000.pt
```

### 3B → 3B Self-Distillation (H100 SXM 80GB)
```bash
cd /root/BitNet && nohup python3 distill/distill.py \
  --device cuda \
  --teacher_model Qwen/Qwen2.5-3B \
  --student_model Qwen/Qwen2.5-3B \
  --dataset slimorca \
  --batch_size 4 \
  --accumulation_steps 8 \
  --max_length 1024 \
  --max_steps 10000 \
  --lr 5e-4 \
  --tau 2.0 \
  --save_every 500 \
  > distill_log.txt 2>&1 &
```

### 7B → 7B Self-Distillation (H100 SXM 80GB, needs optimizer offload)
```bash
cd /root/BitNet && nohup python3 distill/distill.py \
  --device cuda \
  --teacher_model Qwen/Qwen2.5-7B \
  --student_model Qwen/Qwen2.5-7B \
  --dataset slimorca \
  --batch_size 2 \
  --accumulation_steps 16 \
  --max_length 512 \
  --epochs 2 \
  --max_steps 5000 \
  --lr 1e-4 \
  --save_every 500 \
  > distill_log.txt 2>&1 &
```

### Multi-GPU (if pod has multiple GPUs)
The script auto-detects multiple GPUs via FSDP. Launch with torchrun:
```bash
cd /root/BitNet && nohup torchrun --nproc_per_node=<NUM_GPUS> distill/distill.py \
  --device cuda \
  --teacher_model Qwen/Qwen2.5-3B \
  --student_model Qwen/Qwen2.5-3B \
  --dataset slimorca \
  --batch_size 8 \
  --accumulation_steps 2 \
  --max_length 512 \
  --epochs 2 \
  --max_steps 5000 \
  --lr 1e-4 \
  --save_every 500 \
  > distill_log.txt 2>&1 &
```

### Monitor

```bash
# Raw scrolling log
tail -f /root/BitNet/distill_log.txt

# Live color terminal dashboard (curses, press q to quit)
python distill/dashboard.py distill/training_log.jsonl

# Generate PNG plot of training curves
pip install matplotlib  # first time only
python distill/plot.py distill/training_log.jsonl -o distill/training_plot.png

# GPU stats
nvidia-smi
```

## 6. Download Results & Deploy

From the pod:
```bash
runpodctl send /root/BitNet/distill/checkpoints/final.pt
runpodctl send /root/BitNet/distill/training_log.jsonl  # optional, for plotting
```

On your local machine:
```bash
runpodctl receive <CODE>

# One command: export → GGUF → quantize → chat
python distill/deploy.py distill/checkpoints/final.pt
```

## 7. Stop the Pod

**Important**: Stop the pod from the RunPod dashboard to avoid ongoing charges.

- **Stop**: Preserves `/workspace` volume, stops billing for GPU.
- **Terminate**: Deletes everything except persistent volume.

## Cost Reference

| GPU | Price/hr | 3B→3B (5k steps) | 7B→7B (5k steps) |
|-----|----------|-------------------|-------------------|
| A100 80GB | ~$1.04 | ~$4-8 | ~$12-24 |
| H100 SXM 80GB | ~$2.49 | ~$5-10 | ~$10-20 |

## Troubleshooting

**Slow pip install / disk ops**: You're installing to `/workspace` (network storage). Everything should be on `/root` (local container disk). Set container disk to 70-80GB when creating the pod.

**"No module named 'torch'" / slow training / disk full**: Never `pip install torch` — the template already has it. Use `python3 -m venv --system-site-packages venv` so the venv inherits the pre-installed CUDA-optimized torch. Installing torch manually wastes ~5GB disk and may install a CPU-only or mismatched CUDA version, causing 2x slower training.

**OOM errors**: Reduce `--batch_size` (try 4 or 2) or `--max_length` (try 256).

**Pod disconnected**: Training continues if launched with `nohup ... &`. Reconnect and `tail -f /root/BitNet/distill_log.txt`.

**Not enough disk space**: Container disk defaults to 20GB which is too small. Set to 70-80GB when creating the pod to fit models, packages, and checkpoints on fast local storage.
