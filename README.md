# Deconvolution using single-cell and single-nucleus RNA-seq

This project evaluates transformation strategies for using **single-nucleus RNA-seq (snRNA-seq)** as a reference for **bulk RNA-seq deconvolution**, compared to **single-cell RNA-seq (scRNA-seq)**.

Because snRNA-seq captures primarily nuclear RNA, it differs systematically from scRNA-seq, which contains both nuclear and cytoplasmic RNA. We test whether these differences degrade deconvolution performance and whether specific transformations can mitigate them. Comparisons are performed using both **simulated** and **real bulk** datasets.

## Reproducing the results

### 1. Clone the repository

```bash
git clone https://github.com/greenelab/deconvolution_sc_sn_comparison.git
cd deconvolution_sc_sn_comparison
```

### 2. Create the conda environments

Create the required Python and R environments:

```bash
bash environments/create_envs.sh
```

This creates:
- `env_deconv` (Python)
- `env_deconv_R` (R / Bioconductor)

### 3. Download the data

Download all datasets into the `data/` directory using the appropriate dataset identifiers:

```
data/
 └── DATASET_ID/
```

All datasets are publicly available. Links, preprocessing steps, and metadata are provided in:

```
data/details/Data_Details.xlsx
```

### 4. Run the analysis pipeline

Run the following scripts **in order** (typically using `sbatch` on an HPC system).

#### Data preprocessing

```bash
scripts/0_preprocess_data.sh
```

Runs preprocessing and QC notebooks for all datasets.

#### Simulation pipeline

```bash
scripts/1_train_scvi_models_sim.sh
```

Trains scVI models (conditional and non-conditional; with and without DE genes).

```bash
scripts/2_prepare_deconvolution_sim.sh
```

Prepares transformed references and pseudobulks for simulations.

```bash
scripts/3_run_bayesprism_sim.sh
```

Runs BayesPrism / InstaPrism deconvolution.

```bash
scripts/4_process_results_sim.sh
```

Processes simulation results and computes RMSE and Pearson correlation.

```bash
scripts/5_results_notebook_sim.sh
```

Generates notebooks and figures for simulation results.

#### Real bulk comparison

```bash
scripts/6_comparison_with_sc_and_bulks.sh
```

Trains models on real bulks and generates comparison notebooks.

### 5. Inspect results

All figures used in the manuscript are generated in the result notebooks located in:

```
notebooks/
```

## Adding your own method

### Adding a new transformation

1. Preprocess data using:

```bash
scripts/0_preprocess_data.sh
```

2. If training is required, add your code to:

```
scripts/train_scvi_models_allgenes.py
```

3. Add your transformation to:

```
scripts/prepare_deconvolution_sim.py
```

(see the section marked “Add your transformation here”).

### Adding new datasets

1. Add a preprocessing notebook to:

```
notebooks/
```

2. Add the dataset under:

```
data/YOUR_DATASET_ID/
```

3. Register the new dataset ID in the relevant shell scripts, for example:

```bash
datasets=("ADP" "PBMC" "MBC" "MSB")
```

Change to:

```bash
datasets=("ADP" "PBMC" "MBC" "MSB" "YOUR_DATASET_ID")
```

4. For real bulk analyses, add the dataset ID to scripts that include `Real_ADP`.

## Data access and processing

Detailed information on all datasets used in this study — including download links, filtering steps, and preprocessing details — is available in:

```
data/details/Data_Details.xlsx
```
