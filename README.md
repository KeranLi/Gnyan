## Gnyan | **གཉན** : Using pretrained foundational model and data assimilation to link modern and plaeo Tibet plateau

This project, named **gnyan**, is designed to run the **GraphCast** model for analyzing and predicting climate data specific to the **Tibet** region.

### 1. Environment Setup

Follow these steps to set up the environment for **gnyan**:

#### 1.1 Install Conda

Make sure **Anaconda** or **Miniconda** is installed on your machine. If not, you can download and install it from [here](https://docs.conda.io/en/latest/miniconda.html).

#### 1.2 Create Conda Environment

Create the environment using the provided `env_gnyan.yml` file:

```bash
conda env create -f env_gnyan.yml
```

This will set up the required dependencies for **GraphCast**.

#### 1.3 Activate the Environment

Activate the environment:

```bash
conda activate gnyan
```

#### 1.4 Install Additional Dependencies (if needed)

If you encounter missing dependencies, install them using:

```bash
pip install nvidia-dali-cu12 scipy matplotlib
```

### 2. Running the Model

Once the environment is set up, run the training script:

```bash
python graphcast/train_graphcast.py --cfg job
```

### 3. Directory Structure

Here’s a quick overview of the directory structure:

```
gnyan/
├── graphcast/                             # GraphCast model directory
│   ├── train_graphcast.py                 # Training script
├── env_gnyan.yml                          # Conda environment configuration file
└── README.md                              # This file
```

### 4. Notes

* If you are using **GPU acceleration**, make sure to install the appropriate **CUDA** drivers.
* The project uses **NVIDIA DALI** for efficient data loading. Follow the installation instructions in the official [DALI documentation](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html).

---