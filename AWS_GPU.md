# AWS GPU Setup for BitNet Distillation

## Prerequisites

- AWS CLI installed and configured (`aws configure`)
- An AWS account with EC2 GPU instance quota (request g5 quota increase if needed)

## 1. Create SSH Key Pair

```bash
aws ec2 create-key-pair --key-name bitnet-distill --query 'KeyMaterial' --output text > ~/.ssh/bitnet-distill.pem
chmod 400 ~/.ssh/bitnet-distill.pem
```

## 2. Create Security Group

Find your VPC ID:
```bash
aws ec2 describe-vpcs --query "Vpcs[0].VpcId" --output text
```

Create the security group (replace `vpc-XXXXX` with your VPC ID):
```bash
aws ec2 create-security-group \
  --group-name bitnet-distill-sg \
  --description "SSH access for BitNet distillation" \
  --vpc-id vpc-XXXXX \
  --query 'GroupId' --output text
```

Allow SSH access (replace `sg-XXXXX` with the security group ID returned above):
```bash
aws ec2 authorize-security-group-ingress \
  --group-id sg-XXXXX \
  --protocol tcp --port 22 --cidr 0.0.0.0/0
```

## 3. Find the Deep Learning AMI

```bash
aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
  --query "Images | sort_by(@, &CreationDate) | [-1].{ID:ImageId,Name:Name}" \
  --output table
```

## 4. Check Spot Pricing

```bash
aws ec2 describe-spot-price-history \
  --instance-types g5.2xlarge \
  --product-descriptions "Linux/UNIX" \
  --max-items 3 \
  --query "SpotPriceHistory[*].{AZ:AvailabilityZone,Price:SpotPrice}" \
  --output table
```

Find the cheapest availability zone, then get its subnet ID:
```bash
aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=vpc-XXXXX" "Name=availability-zone,Values=eu-west-1a" \
  --query "Subnets[0].SubnetId" --output text
```

## 5. Launch the Instance

Replace `ami-XXXXX`, `sg-XXXXX`, and `subnet-XXXXX` with your values:

```bash
aws ec2 run-instances \
  --image-id ami-XXXXX \
  --instance-type g5.2xlarge \
  --key-name bitnet-distill \
  --security-group-ids sg-XXXXX \
  --subnet-id subnet-XXXXX \
  --associate-public-ip-address \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"persistent","InstanceInterruptionBehavior":"stop"}}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=bitnet-distill}]' \
  --query "Instances[0].{InstanceId:InstanceId,State:State.Name}" \
  --output table
```

Get the public IP once it's running:
```bash
aws ec2 describe-instances \
  --instance-ids i-XXXXX \
  --query "Reservations[0].Instances[0].PublicIpAddress" --output text
```

## 6. Upload Code & Install Dependencies

```bash
# Upload the BitNet repo (excluding large files)
rsync -az --exclude='.git' --exclude='models' --exclude='build' \
  --exclude='distill/checkpoints' --exclude='__pycache__' --exclude='*.pyc' \
  -e "ssh -i ~/.ssh/bitnet-distill.pem" \
  /path/to/BitNet/ ubuntu@<IP>:~/BitNet/

# Install dependencies
ssh -i ~/.ssh/bitnet-distill.pem ubuntu@<IP> \
  "cd ~/BitNet && pip install -r distill/requirements.txt"
```

Verify GPU is working:
```bash
ssh -i ~/.ssh/bitnet-distill.pem ubuntu@<IP> \
  "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
```

## 7. Start Distillation

### Recommended Parameters

| Parameter | A10G 24GB | A100 80GB | Why |
|-----------|-----------|-----------|-----|
| `--batch_size` | 2 | 4-8 | Largest that fits in VRAM |
| `--accumulation_steps` | 16 | 4-8 | Target effective batch size 32 |
| `--max_length` | 512 | 1024 | Longer context = better quality (drop if OOM) |
| `--save_every` | 500 | 200 | Checkpoints are ~12GB each |

**Key insight**: Effective batch size = `batch_size × accumulation_steps`. Target 32 for stable training. Too small (e.g., 16) gives noisier gradients; too large wastes compute without quality gain.

### Single GPU (g5.2xlarge / A10G 24GB)

```bash
ssh -i ~/.ssh/bitnet-distill.pem ubuntu@<IP> \
  "cd ~/BitNet && nohup python3 distill/distill.py \
    --device cuda \
    --teacher_model Qwen/Qwen2.5-3B \
    --dataset slimorca \
    --batch_size 2 \
    --accumulation_steps 16 \
    --max_length 512 \
    --epochs 2 \
    --max_steps 5000 \
    --lr 1e-4 \
    --save_every 500 \
    > distill_log.txt 2>&1 &"
```

### Multi-GPU Distributed (p4d.24xlarge / 8x A100 80GB)

Uses PyTorch FSDP to shard the student model across all GPUs. Launch with `torchrun`:

```bash
ssh -i ~/.ssh/bitnet-distill.pem ubuntu@<IP> \
  "cd ~/BitNet && nohup torchrun --nproc_per_node=8 distill/distill.py \
    --device cuda \
    --teacher_model Qwen/Qwen2.5-3B \
    --student_model Qwen/Qwen2.5-3B \
    --dataset slimorca \
    --batch_size 4 \
    --accumulation_steps 4 \
    --max_length 1024 \
    --epochs 2 \
    --max_steps 5000 \
    --lr 1e-4 \
    --save_every 200 \
    > distill_log.txt 2>&1 &"
```

**Note**: With 8 GPUs, effective batch = `batch_size × accumulation_steps × 8` = 128. Adjust `--accumulation_steps` accordingly. Only rank 0 prints logs and saves checkpoints.

Monitor progress:
```bash
# Raw scrolling log
ssh -i ~/.ssh/bitnet-distill.pem ubuntu@<IP> "tail -f ~/BitNet/distill_log.txt"

# Live color terminal dashboard (curses, press q to quit)
ssh -i ~/.ssh/bitnet-distill.pem ubuntu@<IP> \
  "cd ~/BitNet && python distill/dashboard.py distill/training_log.jsonl"

# Generate PNG plot of training curves
ssh -i ~/.ssh/bitnet-distill.pem ubuntu@<IP> \
  "cd ~/BitNet && pip install matplotlib && python distill/plot.py"

# GPU stats
ssh -i ~/.ssh/bitnet-distill.pem ubuntu@<IP> "nvidia-smi"
```

## 8. Download Results & Deploy

```bash
scp -i ~/.ssh/bitnet-distill.pem \
  ubuntu@<IP>:~/BitNet/distill/checkpoints/final.pt \
  distill/checkpoints/final_3b.pt

# One command: export → GGUF → quantize → chat
python distill/deploy.py distill/checkpoints/final_3b.pt
```

## 9. Stop the Instance

**Important: stop the instance to avoid ongoing charges.**

```bash
aws ec2 stop-instances --instance-ids i-XXXXX
```

To restart later:
```bash
aws ec2 start-instances --instance-ids i-XXXXX

# Get the new public IP (changes on restart)
aws ec2 describe-instances \
  --instance-ids i-XXXXX \
  --query "Reservations[0].Instances[0].PublicIpAddress" --output text
```

To permanently delete (terminates instance and releases storage):
```bash
aws ec2 terminate-instances --instance-ids i-XXXXX
```

## 10. Export & Inference (Local)

After downloading the checkpoint, deploy with one command:

```bash
python distill/deploy.py distill/checkpoints/final_3b.pt -o models/distilled-bitnet-3b
```

This auto-detects the tokenizer, exports to HuggingFace format, converts to GGUF, quantizes, and launches interactive chat.

## Scaling Up: 7B BitNet Distillation

To produce a high-quality 7B-class BitNet model, we self-distill (7B → 7B BitNet). The ternary weight constraint is the bottleneck, not architecture size, so the student should match ~90-95% of the teacher's quality.

### VRAM Requirements (7B self-distillation)

| Component | VRAM |
|-----------|------|
| Teacher (7B, fp16, frozen) | ~14 GB |
| Student (7B, fp32, trainable) | ~28 GB |
| Gradients (fp32) | ~22 GB |
| AdamW optimizer states (2x fp32) | ~44 GB |
| Activations (grad checkpointing) | ~5-10 GB |
| **Total** | **~113-118 GB** |

Does not fit on a single A100 80GB. Two approaches:

### Option A: Multi-GPU (simplest, most expensive)

**Hardware**: AWS `p4d.24xlarge` — 8x A100 80GB (uses 2-3 GPUs)

```bash
python distill/distill.py \
  --device cuda \
  --teacher_model Qwen/Qwen2.5-7B \
  --student_model Qwen/Qwen2.5-7B \
  --dataset slimorca \
  --batch_size 2 \
  --accumulation_steps 16 \
  --max_steps 10000 \
  --save_every 500
```

**Requires**: Script changes for multi-GPU (DeepSpeed ZeRO-2 or model placement across GPUs).

| | Estimate |
|---|---|
| Time | 24-48 hours |
| Spot price | ~$12/hr |
| **Total cost** | **$290-580** |

### Option B: Single A100 80GB with optimizations (budget)

Load teacher in 8-bit quantization (`bitsandbytes`) and offload optimizer states to CPU RAM:

| Component | VRAM |
|-----------|------|
| Teacher (7B, 8-bit quantized) | ~7 GB |
| Student (7B, fp32) | ~28 GB |
| Gradients | ~22 GB |
| Optimizer states → CPU offload | ~0 GB GPU |
| Activations | ~5-10 GB |
| **Total** | **~62-67 GB** → fits on A100 80GB |

```bash
python distill/distill.py \
  --device cuda \
  --teacher_model Qwen/Qwen2.5-7B \
  --student_model Qwen/Qwen2.5-7B \
  --teacher_quantize 8bit \
  --optimizer_offload cpu \
  --dataset slimorca \
  --batch_size 1 \
  --accumulation_steps 32 \
  --max_steps 10000 \
  --save_every 500
```

**Requires**: Script changes for 8-bit teacher loading (`bitsandbytes`) and optimizer CPU offload.

| | Estimate |
|---|---|
| Time | 48-72 hours (slower from CPU offload) |
| Spot price | ~$1-2/hr (single A100 80GB) |
| **Total cost** | **$100-200** |

### Option C: Bigger teacher for best quality (14B → 7B BitNet)

Same as Option A/B but with `--teacher_model Qwen/Qwen2.5-14B`. Slightly better quality than self-distillation, but requires more VRAM (Option A hardware).

### Recommendation

Start with **Option B** for cost efficiency. If quality isn't sufficient, move to Option A with a 14B teacher.

## Instance Types Reference

| Instance | GPU | VRAM | Spot Price* | Best For |
|----------|-----|------|-------------|----------|
| g5.xlarge | A10G | 24 GB | ~$0.40/hr | 1.5B-3B teacher (batch_size=2) |
| g5.2xlarge | A10G | 24 GB | ~$0.63/hr | 3B teacher (more CPU/RAM) |
| p4d.24xlarge | 8x A100 | 8x 80 GB | ~$12/hr | 7B-14B teacher |

*Spot prices vary by region and time. Check current prices with Step 4 above.

## Troubleshooting

**CUDA out of memory**: Reduce `--batch_size` (try 1) or `--max_length` (try 256).

**Spot instance interrupted**: Use `--resume_from` to continue from the last checkpoint:
```bash
ssh -i ~/.ssh/bitnet-distill.pem ubuntu@<IP> \
  "cd ~/BitNet && nohup python3 distill/distill.py \
    --device cuda \
    --teacher_model Qwen/Qwen2.5-3B \
    --dataset slimorca \
    --resume_from distill/checkpoints/step_1000.pt \
    --batch_size 2 --accumulation_steps 16 --max_steps 5000 \
    > distill_log.txt 2>&1 &"
```

**SSH connection refused**: Wait 30-60 seconds after starting the instance for SSH to come up.

**No GPU quota**: Request a quota increase for "Running On-Demand G and VT instances" in the AWS Service Quotas console for your region.
