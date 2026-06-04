# Video Dehazing

Simple deep Learning-based Video Dehazing Framework.

## Installation

```bash
git clone https://github.com/qkrwhdah03/EE499_Dehazing.git
cd EE499_Dehazing

conda create -n dehazing python=3.10
conda activate dehazing

pip install -r requirements.txt
```

---

## Dataset Structure

Organize the dataset as follows:

```text
DATA_ROOT/
├── Train/
│   ├── gt/
│   │   ├── video1/
│   │   ├── video2/
│   │   └── ...
│   └── hazy/
│       ├── video1/
│       ├── video2/
│       └── ...
│
└── Test/
    ├── gt/
    │   ├── video1/
    │   ├── video2/
    │   └── ...
    └── hazy/
        ├── video1/
        ├── video2/
        └── ...
```

Each video directory should contain the corresponding image frames.

---

## Training

`train_base.py` provides a baseline training example.

You are encouraged to freely develop and extend your ideas in `train.py`. The current implementation includes an example prior-based approach.

Run training with:

```bash
python train.py --config [yaml_file_path] --name [experiment_name]
```

### Arguments

| Argument   | Description                                               |
| ---------- | --------------------------------------------------------- |
| `--config` | Path to the YAML configuration file                       |
| `--name`   | Name of the experiment directory for logs and checkpoints |

---

## Evaluation

Evaluate a trained model using:

```bash
python eval.py --checkpoint [checkpoint_path] --data_root [data_root]
```

### Arguments

| Argument       | Description                    |
| -------------- | ------------------------------ |
| `--checkpoint` | Path to the trained checkpoint |
| `--data_root`  | Dataset root directory         |

---

## Inference & Visualization

Generate dehazed outputs for qualitative evaluation:

```bash
python inference.py --checkpoint [checkpoint_path] --data_root [data_root]
```

### Arguments

| Argument       | Description                    |
| -------------- | ------------------------------ |
| `--checkpoint` | Path to the trained checkpoint |
| `--data_root`  | Dataset root directory         |

---

## Notes

* `train_base.py` serves as a reference implementation.
* Feel free to modify the architecture, loss functions, priors, and training strategies.
* Experiment settings are managed through YAML configuration files.
* Checkpoints and logs are automatically saved under the directory specified by `--name`.

---

## License

This repository is intended for KAIST EE499 research project purposes.
