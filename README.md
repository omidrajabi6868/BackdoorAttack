# GhostBackdoor: A Resistant Backdoor Attack

PyTorch research code for training, evaluating, and analyzing resistant backdoor attacks in image classifiers. The repository implements the GhostBackdoor workflow, standard poisoned-model baselines, trigger generation, activation-oriented training, and several backdoor detection or mitigation experiments.

## Supported experiments

- Clean and poisoned classifier training
- Added-class and poison-augmentation variants
- GhostBackdoor/traced-model training
- Trigger optimization and visualization
- Clean-accuracy and attack-success evaluation
- Pruning-based evaluation
- ABS and additional backdoor-analysis scripts

## Models and datasets

The experiment code includes configurations for:

- CIFAR-10
- GTSRB
- MNIST
- ResNet-18
- VGG-16
- Small convolutional networks

The active dataset, architecture, target class, training flags, and hyperparameters are selected near the beginning of `main.py`.

## Installation

```bash
git clone https://github.com/omidrajabi6868/BackdoorAttack.git
cd BackdoorAttack
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision numpy scipy scikit-learn matplotlib pillow
mkdir -p models
```

Install a PyTorch build appropriate for your CUDA version if GPU execution is required.

## Run

1. Open `main.py` and select the model, dataset, training stages, and hyperparameters.
2. Ensure any referenced checkpoints exist under `models/`, or enable the appropriate training stage.
3. Run:

```bash
python main.py
```

The repository also includes shell scripts for SLURM/container environments:

```bash
sbatch main.sh
sbatch eval.sh
sbatch revisiting.sh
```

These scripts use cluster-specific modules and should be adapted to the target system.

## Repository structure

```text
main.py        # Main training and trigger workflow
Network.py     # Training, evaluation, and attack routines
utils.py       # Datasets, poisoning transforms, and helpers
resnet.py      # ResNet implementation
Experiments.py # Additional experiment orchestration
evaluation.py  # Evaluation utilities
pruning.py     # Pruning-based analysis
abs.py         # ABS-related evaluation
Revisiting.py  # Additional comparison experiments
images/        # Example learned mask, pattern, and poisoned image
```

## Citation

If you use this code or method, please cite:

```bibtex
@ARTICLE{11474530,
  author={Rostami, Omid Rajabi and Ning, Rui and Xin, Chunsheng and Cho, Jin-Hee and Li, Jiang and Wu, Hongyi},
  journal={IEEE Internet of Things Journal},
  title={GhostBackdoor: A Resistant Backdoor Attack},
  year={2026},
  volume={13},
  number={12},
  pages={28007-28017},
  keywords={Adversarial robustness; backdoor attacks; deep learning; Internet of Things; model security; secure artificial intelligence systems},
  doi={10.1109/JIOT.2026.3680800}
}
```

Paper DOI: [10.1109/JIOT.2026.3680800](https://doi.org/10.1109/JIOT.2026.3680800)
