# AI-Toolkit - Viking Engine 
[![DOI](https://img.shields.io/badge/DOI.org/10.5281/zenodo.22700108-blue.svg?style=flat&logo=doi)](https://doi.org/10.5281/zenodo.22700108)
---

> [!IMPORTANT]
> # 🚀 ORAKUL STUDIO V2.1 MAJOR UPDATE
> ### Complete Architectural Overhaul | Maximum Speed ​​| Clean Terminal
>
> **Orakul Studio Toolkit** has transitioned to an entirely new, custom-built training core! We have unlocked peak performance on GPUs like the **RTX 4090 (24GB)** and higher.
>
> ⚡ **Training speed for Flux.3 image-klein-base:** up to **~1.5 seconds per iteration** (`1.5s/it`) using full weights with no latency or VRAM leaks!

---

## ⚠️ CRITICAL YAML SETTINGS

To achieve maximum speed on 24GB+ GPUs (RTX 4090 and more powerful), you **MUST** disable all legacy memory-saving workarounds in your `.yaml` configuration file:

```yaml
# ========================================================
# ORAKUL STUDIO v2.1 - OPTIMAL FLUX.3 CONFIGURATION
# ========================================================
engine:
offload: false   # DISABLED: Model stays entirely in VRAM; no offloading to CPU
low_vram: false  # DISABLED: Low-VRAM mode turned off for maximum FPS

### 🚀 Step-by-step guide to starting training

All commands should be executed sequentially from the repository's root directory.

Step 1. Activate the virtual environment
```bash
source venv/Scripts/activate

Step 2. Starting the training process with logs output to a file
python run.py viking_train/r128f3kleintest.yaml 2>&1 | tee "Log_r128f3kleintest.txt"

[!NOTE]

`source venv/Scripts/activate` activates an isolated Python environment.

`python run.py ...` initiates training based on the specified `.yaml` configuration file.

`2>&1 | tee "Log_....txt"` duplicates all console output into a text log file in real time.
=========================================================================================================
Note for Windows Users:
Do not use 2>&1 | tee inside native PowerShell or CMD. PowerShell's Tee-Object mutates \r carriage returns into newlines, ruining tqdm progress bars.

Recommended Launch Methods:

Git Bash / WSL2: python run.py ... 2>&1 | tee Log.txt (works correctly)

PowerShell (Direct): Run python run.py ... without shell redirection, or pass logging via Python/YAML config.
============================================================================================================
git clone https://github.com/OrakulStudio/ai-toolkit.git
cd ai-toolkit
python -m venv venv
.\venv\Scripts\activate
pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

If you encounter numpy.dtype errors or CUDA/Triton warnings, use the following commands to ensure your environment is set up correctly for high-performance training:

Fix NumPy/SciPy version conflicts:
pip install "numpy<2.0.0" scipy --force-reinstall
---

Configs go in viking_train/ folder.
---
training:
# Ready-to-use configuration templates with the correct flags are available in the /viking_train folder
```

> **Note for lower-end GPUs:** This configuration has been tested and optimized for the RTX 4090. For GPUs with less VRAM, these parameters will require individual testing. ---

## 🛠️ WHAT’S NEW IN V2.1?

### 1. 🚫 Complete Removal of Web-UI (Gradio / Streamlit)
We have **completely stripped out the web interface**. Say goodbye to memory-hungry browser UIs, freezing, and unnecessary CPU/VRAM overhead. The system now operates 100% autonomously, running locally and at maximum speed directly from the console.

### 2. 📊 Full Telemetry in the Progress Bar
Essential technical data is now displayed in real-time within the progress bar:
* **Current GPU voltage** and **operating frequency (MHz)**.
* Instantaneous **Loss** value.
* **System announcements:** Get notifications about system updates directly in your terminal perfect for those who don’t check GitHub often.

### 3. 📈 Orakul Viking Monitor (`RUN_MONITOR.bat`)
If a digital Loss readout in the console isn’t enough for you, launch the visual monitor!
* A graph with **EMA (Exponential Moving Average) smoothing** lets you track convergence phases and stop the training process before overfitting occurs.
* Launch it with a **single click** using the `RUN_MONITOR.bat` file no need to manually type commands into the CLI.

### 4. ⚡ Custom-Built Core
The entire logic for memory management and file streaming has been rewritten from scratch. The code is proprietary and optimized to achieve record-breaking speeds.

### 5. 🧬 SVD Health Monitor & Critical Alert System

A universal, real-time matrix diagnostic engine that inspects the internal structural geometry ($W = U \Sigma V^T$) of LoRA weights live during training. Compatible with **all ranks** (from Rank 8 up to Rank 2048+) and **any model architecture** (FLUX.1/2/3, Z-Image, HighDream, Video models, etc.).

---

### 📊 Diagnostic Parameters & Live Indicators

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`Layer [lora_down / lora_up]`** | Core Target | Separately evaluates the down-projection ($A$) and up-projection ($B$) weight matrices. |
| **`Active Axes`** | Health Status | Measures live rank utilization. Displays `(STABLE)` when feature dimensions remain healthy and active. |
| **`Dominant Feature (Max SVD)`** | Capacity Value | Tracks the maximum singular value ($\sigma_{\max}$) to monitor weight saturation and structural balance. |
| **`[!] SVD ALERT`** | **Critical Warning** | **Automated Early Warning System.** Triggers high-priority visual alerts (`critical changes at step X!`) instantly upon detecting gradient spikes, rank degradation, or structural shifts. |

---
![SVD Health Monitor](https://github.com/OrakulStudio/FLUX2-RANK-2048/blob/main/Offload%20Benchmarks/SVD%20Health%20Monitor/33.png)
![SVD Health Monitor](https://github.com/OrakulStudio/FLUX2-RANK-2048/blob/main/Offload%20Benchmarks/SVD%20Health%20Monitor/11.png)

### ⚡ Why It Matters

1. **Automated Gradient Explosion Prevention:** The `[!] SVD ALERT` system immediately flags weight instabilities and gradient spikes at the exact step they occur, preventing hours of wasted GPU compute before the loss graph even reflects the issue.
2. **Universal Compatibility:** Operates identically across all rank configurations (from lightweight Rank 16 to extreme Rank 2048) and any generative architecture (Image, Video, Multimodal).
3. **Beyond Standard Loss Curves:** Captures internal weight collapse, rank degradation, and overtraining long before visual artifacts appear in samples.

---
![SVD ALERT](https://github.com/OrakulStudio/FLUX2-RANK-2048/blob/main/Offload%20Benchmarks/SVD%20Health%20Monitor/22.png)

### 📊 Understanding the SVD Health Monitor (How to read the logs)

At startup and during training, the console displays 4 lines showing the status of the LoRA matrices:

```text
[SVD HEALTH MONITOR | Initial Baseline]
-> Layer [lora_down]: Active Axes = 128/128 | Dominant Feature (Max SVD) = 0.6617
-> Layer [lora_up]:   Active Axes = 128/128 | Dominant Feature (Max SVD) = 0.5185  <-- IMPORTANT POINT
-> Layer [lora_down]: Active Axes = 128/128 | Dominant Feature (Max SVD) = 0.6562
-> Layer [lora_up]:   Active Axes = 128/128 | Dominant Feature (Max SVD) = 0.3558
```
## 🔑 Key information for the user: 
The "Layer [lora_up]" line (the second line) is your primary indicator of efficiency. 
Regarding "Active Axes = X/Y": If your rank is set to 128 and it shows "128/128," the model is utilizing 100% of the vectors. 
If your rank is set to 128 but it shows "20/128," this is NOT an error. 
It simply means that, for the given task, the model only requires 20 vectors. 
The training is proceeding correctly; the remaining "excess" vectors are simply not being utilized. 
## 🚨 What should you do if an "[!] SVD ALERT" appears? 
If a red warning appears in the console "[!] SVD ALERT: 
critical changes at the step X!" do not panic, and do not immediately cancel the training. 
1. Open the `monitor.py` file.
2. On **line 13**, specify the name of your `.txt` log file:
LOG_FILE = "your_log.txt"
and examine the Loss graph: 
If the Loss value is trending steadily downward, everything is fine; the model has simply reorganized its vectors to accommodate a new element in the dataset. 
If the Loss value spikes toward infinity, a "gradient explosion" has occurred, and you should stop the training.

## 🛣️ ROADMAP

Over the coming month, this new, high-speed custom core will be progressively integrated into all other supported model architectures (including **Flux.2**, **SDXL**, and more). *Screenshots of actual logs and confirmation of a speed of 1.5s/it are included in the release materials on GitHub.*


## 

| | | |
|---|---|---|
Metrics in screenshots
![1](https://github.com/OrakulStudio/FLUX2-RANK-2048/blob/main/Offload%20Benchmarks/Flux2-klein-base-logs-images/1.png) | ![2](https://github.com/OrakulStudio/FLUX2-RANK-2048/blob/main/Offload%20Benchmarks/Flux2-klein-base-logs-images/2.png) | ![3](https://github.com/OrakulStudio/FLUX2-RANK-2048/blob/main/Offload%20Benchmarks/Flux2-klein-base-logs-images/3.png) |

---


## Supported Models
**Image**

- [black-forest-labs/flux-3-image-klein-base](https://huggingface.co/black-forest-labs/flux-3-image-klein-base) **(FLUX.3 Image Klein)**
- [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) **(FLUX.1)**
- [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) **(FLUX.2)**
- [black-forest-labs/FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) **(FLUX.2-klein-base-4B)**
- [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) **(FLUX.2-klein-base-9B)**
- [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) **(SDXL)**
- [HiDream-ai/HiDream-I1-Full](https://huggingface.co/HiDream-ai/HiDream-I1-Full) **(HiDream I1)**
- [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) **(Qwen-Image)**
- [Tongyi-MAI/Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) **(Z-Image)**
- [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) **(Z-Image Turbo)**
- [lodestones/Chroma1-Base](https://huggingface.co/lodestones/Chroma1-Base/tree/main/transformer) **(Chroma)**
- [OmniGen2/OmniGen2](https://huggingface.co/OmniGen2/OmniGen2) **(OmniGen2)**

**Instruction / Edit**

- [black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) **(FLUX.1-Kontext-dev)**
- [Qwen/Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) **(Qwen-Image-Edit)**
- [HiDream-ai/HiDream-E1-1](https://huggingface.co/HiDream-ai/HiDream-E1-1) **(HiDream E1)**



**Video**
- [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) **(Wan 2.2 14B)**
- [Wan-AI/Wan2.2-I2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) **(Wan 2.2 I2V 14B)**
- [Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) **(Wan 2.2 TI2V 5B)**
- [Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2) **(LTX-2)**
 
---



---


# 🚨 IMPORTANT UPDATE: RANK 2048 BREAKTHROUGH FOR FLUX.2 (32B) ON CONSUMER GPU (16.10.2026)

> **Update Date:** October 2026 (Follow-up to the September 21 post)  
> **Target Hardware:** RTX 4090 (24GB VRAM) + 128GB RAM + Intel i9-13900K  
> **Status:** ⚡ Full Stability, Zero OOM, Maximum Tensor Core Occupancy  

---

## 🚀 Executive Summary & Critical Findings

In our previous update (21.09), we successfully ran extra-high **Rank 2048** training on a 32-billion parameter FLUX.2 model using `offload: 0.91`. 

However, after deep-dive profiling via **NVIDIA Nsight Systems** (`nsys`) and driver-level benchmarking, we identified a **critical architectural conflict**:

When the framework's default `low_vram` mode is enabled alongside **Viking Engine** (our custom Asynchronous CUDA Memory Manager), the two engines compete for memory control and CUDA streams at the WDDM / pinned-memory layer. This competition induces severe latency spikes (up to 750ms kernel launch delays) and suboptimal GPU occupancy.

By disabling `low_vram` and letting Viking Engine take full control, we eliminated stream contention, resulting in smooth asynchronous memory transfers and maximum compute throughput.

---

## 🛠 Recommended Configuration (YAML)

To achieve maximum training speed and zero OOM errors on a single 24GB GPU at Rank 2048:

```yaml
job: extension
config:
  name: r2048ftranspskin
  process:
  - type: diffusion_trainer
    training_folder: D:\AI_Models\ai-toolkit\output
    sqlite_db_path: D:\AI_Models\ai-toolkit\aitk_db.db
    device: cuda
    trigger_word: r2048ftranspskin
    performance_log_every: 10
    network:
      type: lora
      linear: 2048
      linear_alpha: 64    
      conv: 32            
      conv_alpha: 64      
      lokr_full_rank: true
      lokr_factor: -1
      network_kwargs:
        ignore_if_contains: []
    save:
      dtype: bf16
      save_every: 50
      max_step_saves_to_keep: 2
      save_format: diffusers
      push_to_hub: false
    datasets:
    - folder_path:  D:\AI_Models\datasets\modeldata\transporen
      mask_path: null
      mask_min_value: 0.1
      default_caption: ''
      caption_ext: txt
      caption_dropout_rate: 0.05
      cache_latents_to_disk: true
      is_reg: false
      network_weight: 1
      resolution:
      - 512
      controls: []
      shrink_video_to_frames: true
      num_frames: 1
      flip_x: false
      flip_y: false
      num_repeats: 1
      control_path_1: null
      control_path_2: null
      control_path_3: null
    train:
      batch_size: 1
      bypass_guidance_embedding: false
      steps: 100
      gradient_accumulation: 1
      train_unet: true
      train_text_encoder: false
      gradient_checkpointing: true
      noise_scheduler: flowmatch
      optimizer: adamw8bit
      timestep_type: weighted
      content_or_style: balanced
      optimizer_params:
        weight_decay: 0.0001
      unload_text_encoder: true
      cache_text_embeddings: false
      lr: 0.0001
      ema_config:
        use_ema: false
        ema_decay: 0.99
      skip_first_sample: true
      force_first_sample: false
      disable_sampling: true
      dtype: bf16
      diff_output_preservation: false
      diff_output_preservation_multiplier: 1
      diff_output_preservation_class: person
      switch_boundary_every: 1
      loss_type: mse
    logging:
      log_every: 1
      use_ui_logger: false
    model:
      name_or_path: black-forest-labs/FLUX.2-dev
      quantize: true
      qtype: bf8
      quantize_te: true
      qtype_te: bf8
      arch: flux2
      low_vram: false
      model_kwargs:
        match_target_res: false
      layer_offloading: true
      layer_offloading_text_encoder_percent: 0.71
      layer_offloading_transformer_percent: 0.71
    
meta:
  name: r2048ftranspskin
  version: '1.0'

```

### Why `offload: 0.71` + `low_vram: false`?
1. **`low_vram: false`**: Hands total memory control to Viking Engine's double-buffered async pipeline, removing framework overhead.
2. **`offload: 0.71`**: Frees PCIe bandwidth, ensuring high-speed pre-staging of weights in host RAM without choking CUDA execution streams.

---

## 📊 Nsight Systems Profiling Metrics

Our profiling session (`orakul_report2048.nsys-rep`) validates complete decoupling of compute and memory transfers:

* **Stream 7 (Default Compute Stream):**
  * **83.2%** of active execution time saturated with heavy `cutlass` and `ampere_bf16` GEMM operations.
* **Stream 13 (Dedicated Async Memory Stream):**
  * **90.2%** of stream time executing asynchronous Host-to-Device transfers.
  * Over **1.07 TB (Terabyte!)** of memory transferred H2D seamlessly across the session without stalling GPU compute kernels.

---

## 📁 Attached Artifacts & Benchmarks

* 📄 [r2048ftranspskin.yaml](https://github.com/OrakulStudio/FLUX2-RANK-2048/blob/main/Report-Flux2D(32B)-Rank2048-ofload-0.71-Low-Ram-False-dataset-512x512/r2048ftranspskin.yaml)` — Complete training configuration file.
* 📈 [orakul_report2048.nsys-rep](https://github.com/OrakulStudio/FLUX2-RANK-2048/blob/main/Report-Flux2D(32B)-Rank2048-ofload-0.71-Low-Ram-False-dataset-512x512/orakul_report2048.nsys-rep) — Raw Nsight Systems trace report.
* 📋 [ORAKUL_stats_output.txt](https://github.com/OrakulStudio/FLUX2-RANK-2048/blob/main/Report-Flux2D(32B)-Rank2048-ofload-0.71-Low-Ram-False-dataset-512x512/ORAKUL_stats_output.txt) — Summary of CUDA API and Kernel execution statistics.

---

---

## Nsight Systems trace report

| | | |
|---|---|---|
Metrics in screenshots
![1](https://github.com/OrakulStudio/FLUX2-RANK-2048/blob/main/Report-Flux2D(32B)-Rank2048-ofload-0.71-Low-Ram-False-dataset-512x512/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%20(1210).png) | ![2](https://github.com/OrakulStudio/FLUX2-RANK-2048/blob/main/Report-Flux2D(32B)-Rank2048-ofload-0.71-Low-Ram-False-dataset-512x512/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%20(1211).png) | ![3](https://github.com/OrakulStudio/FLUX2-RANK-2048/blob/main/Report-Flux2D(32B)-Rank2048-ofload-0.71-Low-Ram-False-dataset-512x512/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%20(1210).png) |

---

---

## Test

| | 
|---|
![R2048](https://github.com/OrakulStudio/FLUX2-RANK-2048/blob/main/Report-Flux2D(32B)-Rank2048-ofload-0.71-Low-Ram-False-dataset-512x512/photo_2026-10-16_02-34-32.jpg) |

---


### 📊 FLUX.2 (32B BF8) VRAM & Offload Benchmarks (RTX 4090 24GB)

> **Hardware Context:** RTX 4090 (24GB GDDR6X) | Windows 11 WDDM Driver | Viking Async CUDA Memory Manager

---

#### 🎯 Quick Decision Matrix (Mode Selection)

| Mode | `offload` | Resolution | Rank | Cruising Speed ​​| Purpose |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Stability Mode** *(Set & Forget)* | **`0.71`** | **512×512 / 1024×1024** | **16 – 2048** | **~8 s/it** (512×512)<br>**~16–20 s/it** (1024×1024)<br>**~100 s/it** (Rank 2048) | **Universal preset.** Overcomes memory fragmentation after frequent restarts, prevents WDDM spinlocks, and completely eliminates OOM (Out of Memory) errors at any rank. |
| **Performance Mode** *(Clean Boot)* | **`0.65`** | **1024×1024** | **16 – 128** | **~12–15 s/it** | **Maximum speed.** Requires high computational density (4096 tokens) within a "fresh" CUDA context to overlap weight transfers over the PCIe bus. | ---

#### 💡 Operational Mechanics and WDDM Behavior

1. **`offload: 0.71` (Absolute Stability Mode):**
* Frees up an additional **~2 GB of VRAM**, preventing the Windows WDDM driver from needing to defragment video memory during the micro-pauses between asynchronous CUDA streams. 
* Allows the asynchronous buffer to clear itself instantly: even if the initial step is delayed due to a heavily loaded system, the timing stabilizes to optimal, sustained values ​​by the second or third step.

2. **`offload: 0.65` (Maximum Compression Mode):**
* The hard limit for a native resolution of 1024×1024. 
* At resolutions below 1024×1024 (e.g., 512×512) or during frequent restarts, computations may execute too rapidly, triggering an **Async Stream Underflow**. With no available VRAM headroom, the WDDM driver enters a spinlock state (90+ s/it). In such scenarios, `offload: 0.71` must be used instead.

---

## Logs

**Rank 128 @ 512x512 / 1024x1024 Recovery Log (`offload: 0.71`):**

Set offload to 0.65 | Rank 128 at 1024x1024
[Log](https://github.com/OrakulStudio/FLUX2-RANK-2048/blob/main/Offload%20Benchmarks/1024x1024/Rank128-dataset1024x1024-ofload-0.65.txt)
## 1024x1024

| | 
|---|
![R128 1204x1024](https://github.com/OrakulStudio/FLUX2-RANK-2048/blob/main/Offload%20Benchmarks/1024x1024/1024x1024.png) |


Set offload to 0.71 | Rank 128 at 512x512
[Log](https://github.com/OrakulStudio/FLUX2-RANK-2048/blob/main/Offload%20Benchmarks/512x512/Log_Rank128-dataset512x512-ofload-0.71.txt)
### 512x512

| | 
|---|
![R128 512x512](https://github.com/OrakulStudio/FLUX2-RANK-2048/blob/main/Offload%20Benchmarks/512x512/512x512.png) |


> **TL;DR:** Do not run `low_vram` alongside custom asynchronous offloaders. Set `low_vram: false` and `offload: 0.71` in Viking Engine to train 12B FLUX2 models at Rank 2048 on consumer hardware with zero bottlenecks.


---
---

**High-performance LoRA training on RTX 4090** 

# AI-Toolkit (Windows 11 & Viking Engine Edition)

> ### ⚡ HYPER-SPEED BREAKTHROUGH BENCHMARK (17.09.2026)
> **Proved impossible: FLUX.2 Dev (~8B) Rank 1280 @ ~39s/it on a single RTX 4090 (24GB VRAM).**
> Fully resolved PCIe bandwidth bottlenecks and VRAM overflows during extreme-rank LoRA training.

| Parameter | Standard / Default | Orakul Optimized |
| :--- | :--- | :--- |
| **LoRA Rank** | 128 / 256 | **1280 (Extreme Density)** |
| **Iteration Speed** | ~65–70s / step | **~39–40s / step** |
| **Transformer Offload** | 0.75 – 0.85 | **0.65 (PCIe Bottleneck Bypass)** |
| **Power Consumption** | ~280–350W | **~136W (FP8 E5M2 Efficiency)** |
| **VRAM Stability** | High Risk / OOM on Checkpoints | **Zero OOM (Manual Latent Cache Eviction)** |

> [!IMPORTANT]
> **EN**
> ⚡ **CRITICAL PERFORMANCE NOTE: TRANSFORMER OFFLOAD RATIO**
>
> * **Ranks from 32 to 1024 (Standard & High-Rank):** Always set `offload = 0.75` *(Golden Ratio)*. This provides the ultimate balance between VRAM utilization and PCIe bus throughput (~6.5s/it).
> * **Rank 1280 (Extreme 8B Parameter LoRA):** Switch `offload = 0.65`. Keeping an extra 10% of transformer blocks directly in VRAM circumvents the severe PCIe bus bottleneck for massive adapter matrices, unlocking peak training speed (~39s/it).
> 
> *Do not use `0.65` for low/medium ranks, as it causes VRAM fragmentation and PCIe queueing.*


> [!IMPORTANT]
> **RU**
> ⚡ **КРИТИЧЕСКИ ВАЖНО: НАСТРОЙКА TRANSFORMER OFFLOAD RATIO**
>
> * **Ранги от 32 до 1024 (Стандарт):** Строго **`0.75`** *(Golden Ratio)*. Обеспечивает идеальный баланс VRAM и шины PCIe, отдавая стабильные **~6.5s на шаг**.
> * **Ранг 1280 (Экстремальный 8B LoRA):** Переключайте на **`0.65`**. Удержание ключевых слоёв базовой модели в VRAM полностью снимает пробки на шине PCIe при вычислении гигантских матриц адаптера, выбивая скорость **~39s на шаг**.
> 
> *Внимание: Не используйте `0.65` для стандартных рангов (128–512)  это вызовет фрагментацию VRAM и лишние микропростои шины.*

#### 🔑 Key Engineering Highlights:
* **PCIe Bottleneck Bypass:** Lowering `layer_offloading_transformer_percent` to `0.65` kept critical matrices inside GDDR6X, removing GPU stall states.
* **Zero-OOM Latent Cache Clearing:** Async RAM/VRAM cache clearing prevents memory leak spikes during step checkpoint saves.

## Logs: [HYPER-SPEED 17.09.2026](https://github.com/OrakulStudio/FLUX2-RANK-2048/blob/main/Offload%20Benchmarks/Flux2D_Logs_yaml_17.09.2026/r1280f2aivazovsky.txt)

## Configuration file: [yaml](https://github.com/OrakulStudio/FLUX2-RANK-2048/blob/main/Offload%20Benchmarks/Flux2D_Logs_yaml_17.09.2026/r1280f2aivazovsky.yaml)

**AI-Toolkit (Windows 11)** is a user-friendly, high-performance toolkit designed for training diffusion models on consumer-grade hardware. 

Built upon the  AI-Toolkit framework, this version has been extensively re-engineered to deliver server-grade training speeds and exceptional VRAM efficiency within the Windows 11 environment.
*Server-class speed on consumer hardware*

We don't compromise.

Hardware Performance: The Viking Engine is optimized for Windows not out of laziness, but because it provides direct, low level access to the RTX 4090's resources without any slowdowns.

Death to workarounds: Linux is beautiful when you're browsing the web. But when it comes to working with heavyweights, asynchronous memory streaming, and real computing resources, it turns into a patchwork of patches. We value our time and studio resources more than "pretty" fonts in the terminal.

Unique Architecture: Our optimizations are the result of extensive work with the Ada Lovelace architecture. It works where it's supposed to fast, predictable, and without surprises like broken drivers or incompatible libraries.

Orakul Studio is designed for those who want to get things done, not just "administer" systems. Our base is Windows. Our goal is results. Anything else is just a waste of time.
Linux is great for servers and clean code. But when you have two hours of daylight and need to run 200 steps to rank 1280, you don't care about a "proper kernel contract." You just want it to work.

So yes, my code is written for Windows. And it runs. If anyone wants to port the Viking Engine to Linux, they're welcome, I'm not opposed. But for now, I'll stick with where it's less of a hassle and more rewarding.

⚠️ Nevertheless, Linux support for the specialized Viking Engine is under active development.



**Contact and Support:**

For questions regarding the proprietary memory management layer, integration nuances, or non-standard configurations, please email me at   [orakulstorm@gmail.com](https://mail.google.com/mail/u/0/#inbox)

*Orakul Studio - Chernihiv, Ukraine 🇺🇦*

---

## ⚠️ STRICTLY FOR TERMINAL / CLI RUNNING ONLY

If you're one of the 400+ people who cloned this repository, forget about any web UIs for this pipeline.

This code was designed, rewritten, and optimized exclusively for directly running configuration files (.yaml) via the console.

### What breaks when running via the Web UI:

1. Dynamic Alpha  DOESN'T WORK AT ALL**
* This repository implements dynamic Alpha recalculation logic for correct weight scaling (Scale = Alpha / Rank). For example, when working with high ranks (Rank 128, Rank 512, Rank 1024), the system automatically calculates a fair scale (down to Scale = 0.5000), allowing the model to deeply learn the structure and physics of the material.
* **The web UI completely ignores this logic.** Almost all web wrappers under the hood forcibly overwrite this parameter and force a fixed Alpha = 16. At high ranks, this turns training into a dud: weight changes are suppressed, gradients tend to zero, the model visually "learns" without errors, but produces default output.

2. **Asynchronous Memory Manager (Async CUDA Memory Manager)  CRASHED**
* The logic for memory retention and low-level logging is optimized for the terminal's stdout.
* Web interfaces attempt to intercept and parse the string stream for their browser consoles. At best, this leads to a crash of the backend interface due to custom security prints; at worst, to a hidden downcast of tensor precision and gradient castration, so that a casual user doesn't simply "get a memory error."

### How to use the repository correctly:
* Run strictly through the console, directly from your virtual environment.
* If your process crashes while working with high ranks and dynamic alpha, **don't look for compromises in the code; instead, increase the system swap/pagefile**. The terminal works with your hardware without censorship or hidden precision reductions.

## What This Is

The original ai-toolkit is an excellent, flexible framework. This fork takes it in one specific direction: **maximum performance on RTX 4090 (Ada Lovelace, sm_89)**.

Not about making weak hardware work. About making strong hardware **fly**.


### ⚡ Benchmark & Performance Verification (FLUX.2-Dev / RTX 4090)

| LoRA Configuration | Speed (s/it) | VRAM Memory Status ||
| :--- | :--- | :--- | :--- |
| **Rank 128 (Optimized)** | **6.70s / 6.50s** | 24 GB (Zero OOM / Stable) | 
| **Rank 512 (Deep Gesture)** | **8.97s** | 24 GB (Double Buffered) | 
| **Rank 1024 (Extreme)** | **22.45s** | 24 GB (Full 8-bit Stack Forced) |
| **Rank 1280 (Extreme)** | **65.80s** | 24 GB (Full 8-bit Stack Forced) | 

---
---
## Benchmark - Flux2-dev, RTX 4090, Rank 128
[orakul_report_folder_logs](https://github.com/OrakulStudio/FLUX2-RANK-2048/tree/main/Offload%20Benchmarks/orakul_report)

<img width="3840" height="2160" alt="10" src="https://github.com/OrakulStudio/FLUX2-RANK-2048/raw/main/Offload%20Benchmarks/orakul_report/10.png" />
<img width="3840" height="2160" alt="3" src="https://github.com/OrakulStudio/FLUX2-RANK-2048/blob/main/Offload%20Benchmarks/orakul_report/3.png" />
<img width="3840" height="2160" alt="8" src="https://github.com/OrakulStudio/FLUX2-RANK-2048/blob/main/Offload%20Benchmarks/orakul_report/8.png" />

[orakul_report.txt](https://github.com/OrakulStudio/FLUX2-RANK-2048/blob/main/Offload%20Benchmarks/orakul_report/orakul_report.txt)


## Benchmark - Flux2-dev, RTX 4090, Rank 512

| Version | Speed | What Changed |
|---------|-------|-------------|
| Baseline (original) | 179 s/it | — |
| Viking v1 | 37 s/it | Double-buffer async CUDA |
| Viking v2 | 14 s/it | + bf16 weight forcing |
| Oracle-60 | 8.7 s/it | + Hardware FP8 + CPU prequant |
| **LEGEND** | **7.3 s/it** | **+ Full 8-bit stack (AdamW 8-bit)** |

**24.5× faster than baseline. Same hardware. Zero OOM at rank 1280 (7.8B trainable params).**

---

## Modified Files

### [toolkit/manager_modules.pyd](https://github.com/OrakulStudio/AI-Toolkit-Windows11/blob/main/toolkit/memory_management/manager_modules.pyd) — Viking Engine
**Double-buffered async weight streaming.**

While GPU computes layer N, weights for layer N+1 transfer in a parallel CUDA stream. Transfer disappears from the profiler entirely.

```python
> 🔒 **Orakul Studio Proprietary Tech**  
> Core architecture and high-performance memory optimization layers are closed-source. Distributed exclusively via compiled binary module. The repository is open, and the pipeline is fully functional and stable..
```

Also: CPU pinned memory for direct DMA from DRAM without CPU cache copy.

---

### [toolkit/quantize.py](https://github.com/OrakulStudio/AI-Toolkit-Viking-Engine-Fork/blob/main/toolkit/util/quantize.py)  Protocol Orakul-60
**Native FP8 (E5M2) for Ada Lovelace + CPU pre-quantization.**

RTX 4090 has native FP8 Tensor Cores. This activates them.  
CPU pre-quantizes transformer blocks before GPU load  PCIe bus freed.

```python
"bf8": Float8WeightOnlyConfig(weight_dtype=torch.float8_e5m2)
```

```
>>> [ORACLE-60] BF8 NATIVE (E5M2) DETECTED - CPU PRE-QUANT MODE
```

**Why E5M2:** preserves dynamic range like BF16. No "muddy faces" from aggressive quantization. Skin texture survives.

---

### [toolkit/lora_special.py](https://github.com/OrakulStudio/AI-Toolkit-Viking-Engine-Fork/blob/main/toolkit/lora_special.py)  Viking Override
**LoRA matrices born in bfloat16, not converted later.**

```python
# === VIKING OVERRIDE: ЖЕСТКИЙ BFLOAT16 ДЛЯ ВЕСОВ ===
dtype = torch.bfloat16
```

All Linear and Conv2d LoRA layers initialized directly in bf16. Half the memory at birth. PCIe transfer halved from step one.

**Also fixed:** Alpha/scale calculation for rank 1024.  
Original code skipped `.alpha` keys during save, causing **64× signal drop** at rank 1024 with alpha 64. Fixed and locked:

```python
# === ORAkul STANDART PASS (FIXED) ===
alpha_val = alpha if alpha is not None and alpha != 0 else lora_dim
self.scale = alpha_val / self.lora_dim
```

---

### [toolkit/network_mixins.py](https://github.com/OrakulStudio/AI-Toolkit-Viking-Engine-Fork/blob/main/toolkit/network_mixins.py)  Orakul-60 Alpha Fix
**The alpha skip bug that corrupted high-rank training.**

Lines that skipped `.alpha` keys for non-LoKR networks are permanently commented with explanation. The `lora_special.py` autopilot handles scale correctly now.

```python
# === [ORAkul-60] BLOCK: ЗАЩИТА ОТ МЫЛА ===
# These lines skipped Alpha for all types except LoKR.
# At Rank 1024, signal dropped 64x (Scale 16/1024 instead of 64/1024).
# KEEP DISABLED  lora_special.py autopilot handles this correctly.
```

---

### [toolkit/style.py](https://github.com/OrakulStudio/AI-Toolkit-Viking-Engine-Fork/blob/main/toolkit/style.py) — Perceptual Loss (Rewritten)
**VGG19 perceptual loss without the VRAM leak.**

Original had a critical bug: VGG19 was computing gradients for itself during perceptual loss calculation  wasting gigabytes of VRAM on a frozen reference network.

Fixed:
```python
# Freeze VGG19 completely — it's a reference, not a trainee
for param in cnn.parameters():
    param.requires_grad = False
```

Also: Gram matrix calculation replaced with hardware `torch.bmm` (Batch Matrix Multiply)  faster, cleaner, no nested function overhead.

**Use perceptual loss for:** Aivazovsky, watercolor, oil painting, charcoal, any artistic style where brushstroke texture matters more than pixel accuracy.

**Use MSE for:** portraits, photorealism, identity training.

---

### [toolkit/timer.py](https://github.com/OrakulStudio/AI-Toolkit-Viking-Engine-Fork/blob/main/toolkit/timer.py)  Timer Silenced
**CPU overhead from time polling eliminated.**

Original timer called `time.time()` on every micro-step of the pipeline — `predict_unet`, `backward`, `optimizer_step`, etc. Parasitic CPU load, log spam, potential micro-freeze points on PCIe bus under heavy load.

```python
def start(self, timer_name):
    return  # CPU freed

def stop(self, timer_name):
    return  # bus cleared

def print(self):
    for hook in self._after_print_hooks:
        hook({})  # engine hooks get empty dict, nothing breaks
    return
```

Result: cleaner logs, lower CPU load, stable PCIe bus under sustained training.

---

### [jobs/process/BaseSDTrainProcess.py](https://github.com/OrakulStudio/AI-Toolkit-Viking-Engine-Fork/blob/main/jobs/process/BaseSDTrainProcess.py)  bf16 Forcing
**Network forced to bfloat16 before training starts.**

```python
# Viking method  before network.apply_to()
# todo switch everything to proper mixed precision like this  ← ostris left this todo
self.network.force_to(self.device_torch, dtype=torch.bfloat16)
```

The `# todo` comment was already in the original source. We read it and implemented it.

---

## Two Training Modes

### Mode 1: Photorealism (Maximum Speed)
For portraits, faces, identity training. Pixel-accurate, fast.

```yaml
content_or_style: balanced
loss_type: mse
```

### Mode 2: Art Styles (Perceptual)
For Aivazovsky, watercolor, oil, charcoal. Learns brushstroke, not pixels.  
Uses ~1-2 GB more VRAM, ~20-30% slower, higher GPU voltage.

```yaml
content_or_style: style
loss_type: perceptual
```

Switch between modes by commenting/uncommenting  no config rewrite needed.

---

## Recommended Config

```yaml
network:
  type: lora
  linear: 1280          # rank 1280 = 7.8B trainable params
  linear_alpha: 64
  conv: 32
  conv_alpha: 64
  lokr_full_rank: true
  lokr_factor: -1

model:
  qtype: bf8            # Orakul-60 native FP8
  quantize_te: true
  qtype_te: bf8
  layer_offloading: true
  layer_offloading_transformer_percent: 0.91
```

---

## Launch

Web UI works but **sampling is disabled for maximum speed**.  
For real performance  terminal only:

```bash
# Activate environment
source venv/Scripts/activate

# Run training with full log capture
python run.py viking_train/your_config.yaml 2>&1 | tee "log_training.txt"
```

Configs go in `viking_train/` folder.

---

## Installation & Requirements

- RTX 4090 (24 GB)  optimized for Ada Lovelace sm_89
- 64+ GB RAM recommended (128 GB for rank 1280)
- Python 3.10+ / Linux or Windows 11

```bash
git clone https://github.com/OrakulStudio/ai-toolkit.git
cd ai-toolkit
python -m venv venv
.\venv\Scripts\activate
pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

If you encounter numpy.dtype errors or CUDA/Triton warnings, use the following commands to ensure your environment is set up correctly for high-performance training:

Fix NumPy/SciPy version conflicts:
pip install "numpy<2.0.0" scipy --force-reinstall
---

Configs go in viking_train/ folder.
---

```

## Supported Models
**Image**

- [black-forest-labs/flux-3-image-klein-base](https://huggingface.co/black-forest-labs/flux-3-image-klein-base) **(FLUX.3 Image Klein)**
- [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) **(FLUX.1)**
- [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) **(FLUX.2)**
- [black-forest-labs/FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) **(FLUX.2-klein-base-4B)**
- [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) **(FLUX.2-klein-base-9B)**
- [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) **(SDXL)**
- [HiDream-ai/HiDream-I1-Full](https://huggingface.co/HiDream-ai/HiDream-I1-Full) **(HiDream I1)**
- [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) **(Qwen-Image)**
- [Tongyi-MAI/Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) **(Z-Image)**
- [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) **(Z-Image Turbo)**
- [lodestones/Chroma1-Base](https://huggingface.co/lodestones/Chroma1-Base/tree/main/transformer) **(Chroma)**
- [OmniGen2/OmniGen2](https://huggingface.co/OmniGen2/OmniGen2) **(OmniGen2)**

**Instruction / Edit**

- [black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) **(FLUX.1-Kontext-dev)**
- [Qwen/Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) **(Qwen-Image-Edit)**
- [HiDream-ai/HiDream-E1-1](https://huggingface.co/HiDream-ai/HiDream-E1-1) **(HiDream E1)**



**Video**
- [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) **(Wan 2.2 14B)**
- [Wan-AI/Wan2.2-I2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) **(Wan 2.2 I2V 14B)**
- [Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) **(Wan 2.2 TI2V 5B)**
- [Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2) **(LTX-2)**
 
---


## Links

- 🐙 [github.com/OrakulStudio](https://github.com/OrakulStudio)
- 🤗 [huggingface.co/OrakulStorm](https://huggingface.co/OrakulStorm)
- 🎨 [civitai.com/user/orakul_studio](https://civitai.com/user/ORAKUL_STUDIO)

```


*The smell of the iron is stable. 🦊⚡*

*Chernihiv, Ukraine 🇺🇦 · Orakul Studio · 2026*
