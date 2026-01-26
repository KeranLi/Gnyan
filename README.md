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

### 5. Performance

* **ERA5 dataset visualization**:

<p align="center">
  <img src="https://private-user-images.githubusercontent.com/66153455/540486975-c5865863-f27d-4a14-b75e-181d874eb3e7.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njk0Mzk0MTgsIm5iZiI6MTc2OTQzOTExOCwicGF0aCI6Ii82NjE1MzQ1NS81NDA0ODY5NzUtYzU4NjU4NjMtZjI3ZC00YTE0LWI3NWUtMTgxZDg3NGViM2U3LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAxMjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMTI2VDE0NTE1OFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTEwZjlkNTk3Y2ZlMDRkNzUyYzJmN2EzMDAwZjAxYzY2OGQ1YzNlZjFjZDdmZjIwMmY0ZDYzMDRlNjZkZmYyZDkmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.hmZOmqU0LbQ3BP9er9rIJ2muDz5o7rnQ3sOKV4bmD-o" width="600">
</p>

* **ERA5 dataset statistics**:

<p align="center">
  <img src="https://private-user-images.githubusercontent.com/66153455/540487107-677fd578-df11-4b34-8c1c-06cd93066ddf.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njk0MzkxMzIsIm5iZiI6MTc2OTQzODgzMiwicGF0aCI6Ii82NjE1MzQ1NS81NDA0ODcxMDctNjc3ZmQ1NzgtZGYxMS00YjM0LThjMWMtMDZjZDkzMDY2ZGRmLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAxMjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMTI2VDE0NDcxMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTg3ZTEwOGRkYmE2M2ViNTMzNjdhMzY5MTk4MTgwZDg3MTdiM2RhOTk3NmJjYTNjNjMzNDNlMTlhNTAxZjAwYzQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.YvqZbsnAJw_rJEaVYSYYXrcAuufVFRFlPARO9CbqeB8" width="600">
</p>

* **Copernicus DEM-30m dataset: Regional**:

<p align="center">
  <img src="https://private-user-images.githubusercontent.com/66153455/540549456-6b6468d8-8566-4ae9-998a-b1730d0300dd.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njk0MzkxNzIsIm5iZiI6MTc2OTQzODg3MiwicGF0aCI6Ii82NjE1MzQ1NS81NDA1NDk0NTYtNmI2NDY4ZDgtODU2Ni00YWU5LTk5OGEtYjE3MzBkMDMwMGRkLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAxMjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMTI2VDE0NDc1MlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTQzNWZkMjNiYzQ2Y2ZmZTBiOTQ4NGQ0NjM3M2I4MDdkNGZmOGM5MmRmODZiMzFiZWFlZjFmMzNkOWNlMTU4NWQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.TL4-S2La0u9LVl8H28gnmtSRLCLlkbpMCrif4lGHZvA" width="600">
</p>

* **Copernicus DEM-30m dataset: Tibet plateau**:

<p align="center">
  <img src="https://private-user-images.githubusercontent.com/66153455/540549594-736cf02c-2ebb-446a-ae5b-ee13adb2235c.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njk0Mzk2MjksIm5iZiI6MTc2OTQzOTMyOSwicGF0aCI6Ii82NjE1MzQ1NS81NDA1NDk1OTQtNzM2Y2YwMmMtMmViYi00NDZhLWFlNWItZWUxM2FkYjIyMzVjLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAxMjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMTI2VDE0NTUyOVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWIwMTcyYTQ2OWMxZWRiYzUzZTMwZjBiMzIyYTFkNGMzMTQ1NmU4MmQ3ZmQxOTM3Y2FkYTA1ZWZlMDFjNjdjZTMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.QhRnk9xf8sZ_DysmqLlDIRcwv0y1WVQr8nKcZvyLvzA" width="600">
</p>
---