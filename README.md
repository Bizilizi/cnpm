# CNPM
This repository contains implementation of Color Neural Parametric Models. A learnable parametric model for object shape and color representations.

![reconstructions](https://i.postimg.cc/DzXycdS4/Untitled-Reconstruction.jpg)

# Repository structure
This repo contains two implementation for CNPM model, their source roots can me found in following directories:
- `src/cnpm`
- `src/cnpm_v2`

Each implementation root contains 4 modules:
- `src/{model_name}/callbacks` Are responsible for PyTorchLightning callbacks used by training loop
- `src/{model_name}/data` Implementation of DataSets and DataModules
- `src/{model_name}/model` Actual implementation of a model
- `src/{model_name}/traning` Training methods that setup callbacks, loggers and run training loop

# Training

Training configuration for each of the model can be found in corresponding root repositories. In order to run training, you should run following command:
```cmd
python3 train.py --model cnpm --seed 42
```
To get full list of commands:
```cmd
python3 train.py --help
```

# Dataset

Datasets are publicly available by the following links:
- CNPM-V2: https://disk.yandex.com/d/mqaClruNZQwhsA
- CNPM: https://disk.yandex.com/d/mqaClruNZQwhsA

# Optimization
Once the model is trained, you can use optimization scripts from `src/cnpm/scripts/optimization`, they generate artifacts with optimization process, that later can be used for statistical analysis.

An example of optimization process made on cloud points from multiple renders:
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/7IFfl-2-OdI/0.jpg)](https://www.youtube.com/watch?v=7IFfl-2-OdI)