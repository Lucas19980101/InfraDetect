## Multimodal Feature Description Network
This repository implements the Multimodal Feature Description Network (MFDN) for aligning RGB–IR image pairs in building facade inspection.

This is the first part of the code for our paper:  
[Weakly-aligned Cross-modal Learning Framework for Subsurface Defect Segmentation on Building Facades Using Unmanned Aerial Vehicles](https://www.sciencedirect.com/science/article/abs/pii/S0926580524006824),  
published in ***Automation in Construction***.


## Usage

### Environment setup (RTX 5070 Ti / Blackwell)

Run these commands from the repository root. This configuration uses Python 3.11
and PyTorch 2.7.1 with CUDA 12.8, which supports Blackwell GPUs.
An NVIDIA driver that supports your GPU and CUDA 12.8 is required.

```bash
conda env create -f environment.yaml
conda activate MFDN
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); x = torch.randn(256, 256, device='cuda'); print((x @ x).mean().item())"
```

The environment file is UTF-8 and includes the PyTorch CUDA wheel index.
The training and feature extraction scripts currently require CUDA; `--gpu -1`
is not supported. ONNX packages are only needed for the separate ONNX workflow.

### 1. Data preparation
- **Project structure**: Place your data under the project root in a directory named after the project, e.g. `UST_Campus`, `Tower_D`, or `Facade_18`.
- **Train / Test split**: Each project directory should contain:
  - **`train/`**: training images (e.g. `train/VIS`, `train/IR`).
  - **`test/`**: testing images and metadata. The `test/` folder must include at least the following **5 subfolders**:
    - `VIS` – visible images
    - `IR` – infrared images
    - `ref` – reference images (if applicable)
    - `homo` – homography (`.mat`) files
    - `landmarks` – landmark (`.mat`) files

### 2. Extract multimodal features
- **Script**: `extract_MMFeat.py`  
- **Purpose**: Run MFDN to extract multimodal keypoints and descriptors for each VIS/IR image pair.
- **Output**: `.features.mat` files are saved under the `features/<Project>/MFDN/` directory.
- **Example**:
  ```bash
  python extract_MMFeat.py \
    --subsets UST_Campus \
    --model ./Pretrained/UST_Campus.pth \
    --gpu 0
  ```

### 3. Evaluate the model
- **Script**: `match_ust.py`  
- **Purpose**: Evaluate feature matching performance using the extracted features.
- **Input**: Features under `features/<Project>/MFDN/` and homographies/landmarks under `<Project>/test/`.
- **Example**:
  ```bash
  python match_ust.py \
    --feature_name MFDN \
    --subsets UST_Campus \
    --nums_kp 4096
  ```

#### Validate the pretrained model on MSRS

MSRS already provides aligned pairs, uses lowercase `vi`/`ir` folder names, and
does not include per-image homography files. Extract features from its official
test split and evaluate against the identity transform:

```bash
python extract_MMFeat.py \
  --subsets MSRS --split test --vis-dir vi --ir-dir ir \
  --model ./Pretrained/UST_Campus.pth --gpu 0

python match_ust.py \
  --subsets MSRS --split test --vis-dir vi --ir-dir ir \
  --identity-gt --feature_name MFDN --nums_kp 4096
```

This tests how the UST Campus pretrained descriptor transfers to MSRS; it does
not train or fine-tune the model. Since MSRS is distributed as aligned data and
does not provide exact registration ground truth, the identity-transform scores
also include any residual alignment error in the dataset.

To inspect homographies estimated from the descriptors, run:

```bash
python reproj_only.py \
  --subsets MSRS --split test --vis-dir vi --ir-dir ir \
  --feature_name MFDN --nums_kp 4096
```

#### ReDFeat VIS-IR registration benchmark

Build the official transformed VIS-IR test pairs from RoadScene and, when
available, the extracted LGHD RGB/LWIR archive:

```bash
python tools/prepare_redfeat_vis_ir.py \
  --benchmark /path/to/Multimodal_Feature_Evaluation/VIS_IR \
  --roadscene /path/to/RoadScene \
  --lghd /path/to/lghd_icip2015_rgb_lwir \
  --output VIS_IR
```

The LGHD argument is optional. Without it, the script builds the 39 RoadScene
pairs and records the 10 skipped LGHD pairs in `VIS_IR/manifest.json`.

```bash
python extract_MMFeat.py --subsets VIS_IR \
  --model ./Pretrained/UST_Campus.pth --gpu 0

python match_ust.py --subsets VIS_IR --redfeat-gt \
  --feature_name MFDN --nums_kp 4096
```

For `.21` samples the supplied transform maps VIS to the transformed IR image.
For `.12` samples evaluation uses its inverse so every score consistently uses
the current VIS-to-current-IR direction.

### 4. Generate reprojected (warped) images
- **Script**: `reproj_ust.py`  
- **Purpose**: Use the estimated homography to warp VIS images onto IR images (or vice versa) and visualize alignment.
- **Output**: Reprojected images and homography matrices saved under `results/<Project>/...`.
- `reproj_ust.py` requires `test/homo/<name>.mat` (key `H`) and
  `test/landmarks/<name>.mat` (keys `IRPoints` and `RGBPoints`).
  To warp images without these annotations, use `reproj_only.py` with the same arguments.
- For compatibility, the matrix output directory retains the original name
  `homo_ir2vis`, but the stored matrix maps VIS coordinates to IR coordinates.
- **Example**:
  ```bash
  python reproj_ust.py \
    --feature_name MFDN \
    --subsets UST_Campus \
    --nums_kp 4096
  ```

### 5. Model training
- **Script**: `train.py`  
- **Purpose**: Train MFDN on RGB–IR pairs.
- **Recommendation**:
  - **Combine** open-source datasets (e.g., [VIS-IR](https://github.com/ACuOoOoO/Multimodal_Feature_Evaluation)) with **your own data** to improve robustness.
  - `--datapath` accepts either a project directory (`UST_Campus/train/VIS`,
  `UST_Campus/train/IR`) or the original layout (`<datapath>/VIS_IR/train/VIS`,
  `<datapath>/VIS_IR/train/IR`). Paired files must have identical names.
- Training creates the checkpoint and `logs/` directories automatically.
- Adjust `--image_type`, `--datapath`, and training hyperparameters (`--num_epochs`, `--lr`, etc.) as needed.
- **Example**:
  ```bash
  python train.py \
    --image_type VIS_IR \
    --datapath ./UST_Campus/ \
    --gpu 0 \
    --num_epochs 50 \
    --batch_size 2
  ```

## Citation
If you find this work useful in your research or projects, please consider citing the following paper:

```text
@article{he2025weakly,
  title={Weakly-aligned cross-modal learning framework for subsurface defect segmentation on building facades using UAVs},
  author={He, Sudao and Zhao, Gang and Chen, Jun and Zhang, Shenghan and Mishra, Dhanada and Yuen, Matthew Ming-Fai},
  journal={Automation in Construction},
  volume={170},
  pages={105946},
  year={2025},
  publisher={Elsevier}
}
```

## Acknowledge
In this project, we use parts of codes in:

[RIFT](https://github.com/LJY-RS/RIFT-multimodal-image-matching)

[RedFeat](https://github.com/ACuOoOoO/ReDFeat)

[SAR-SIFT](https://github.com/yishiliuhuasheng/sar_sift)
