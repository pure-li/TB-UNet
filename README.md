# TB-UNet: Self-Supervised Deep Learning for Geomagnetic Data Reconstruction

This repository contains the source code for the paper submitted to *Computers & Geosciences*.

## Repository Structure

```
.
├── U-net.py                          # Core model: U-Net + Transformer backbone
├── U-net_transformer_final/          # TB-UNet full implementation
├── 第四章结果与分析/                  # Chapter 4: Experiments & Results
│   ├── ablation experiment/          # Ablation study (4 variants)
│   ├── comparative experiment/       # Comparison with CNN, GAN, Kriging
│   ├── noise experiment/             # Noise robustness tests
│   └── robustness experiment/        # Spatial generalization tests
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy, SciPy, scikit-learn, scikit-image
- Pandas, Matplotlib

## Data

The geomagnetic anomaly data used in this study is from the Afghanistan aeromagnetic survey (2006). A preprocessed CSV file should be placed at `afghanistan_full/Afghan_mag06A.csv`.

## Citation

If you use this code, please cite our paper:

> [Authors]. TB-UNet: Self-Supervised Deep Learning for Geomagnetic Data Reconstruction via Block-Mask Training. *Computers & Geosciences*, [Year].
