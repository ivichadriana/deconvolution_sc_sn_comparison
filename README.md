# **Deconvolution using single-cell RNA sequencing dataset combined with a single-nucleus cell type.**

The aim of this research project is to evaluate methods of transformation that can be applied to single-nucleus RNA sequencing dataset in order to improve deconvolution of bulk RNA-seq. Single-cell RNA-seq and bulk RNA-seq share similar expression (cytoplasmic and nuclear RNA), while single-nucleus RNA-seq only contains nuclear RNA. Our central hypothesis is that differences makes single-nucleus RNA-seq a poor deconvolution reference. We compare the two modalities in both simulations and real data, and we present some transformation options. 

# **Reproducing the results**

- Fork or branch and clone the repository. [Github has many tutorials on this](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)
- Run the (bash script to create the conda environments)[/environments/create_env.sh] needed.
    This creates the conda environments from the (enviornments folder)[/environments/] using the yml files (two environments, one for R (env_deconv_r) and one for Python (env_deconv)). 
- Download the data we use, putting in the appropiate folder (data/ID). All data is publicly available and easily downloadable. All links and details can be found on the Excel sheet [here](data/details/Data_Details.xlsx) - same Esxcel is also a supplemental figure on paper.
-  After downaloading the data, run the shell scripts, in order:
    - [0_preprocess_data.sh](scripts/0_preprocess_data.sh)
        - Runs [preprocessing notebooks](notebooks/). Preprocessing and QC for all datasets. 
    - [1_train_scvi_models_sim.sh](scripts/1_train_scvi_models_sim.sh) 
        - Runs [training scripts](scripts/train_scvi_models_allgenes.py) Trains scVI models (conditional and not conditional), with and without genes.
    - [2_prepare_deconvolution_sim.sh](scripts/2_prepare_deconvolution_sim.sh) 
        - Runs [script to prepare files for deconvolution](scripts/prepare_deconvolution_sim.py) (only simulations). Files needed are one reference for each transform, where we transform one cell type at a time. 
    - [3_run_bayesprism](scripts/3_run_bayesprism_sim.sh)
        - Runs (script for deconvolution)[BayesPrism_sim.R] through BayesPrism/InstaPrism using the references and pseudobulks created on 2.Tutorials on InstaPrism available [here](https://github.com/humengying0907/InstaPrismSourceCode).
    - [4_process_results_sim.sh](scripts/4_process_results_sim.sh)
        - Runs (script to process the results from deconvolution)[scripts/process_results.py] (only of simulation), computes RMSE and Pearson, and puts it in a format for analysis.

- You can look at the results in the results notebooks after! All plots included in all figures of the paper will be available in these.

Example on running bash on HPC: 
> sbatch sciripts/0_preprocess_data.sh

# Contribute to the research!

## **Instructions for adding your own method to the analysis:**
- Preprocess data [as usual](scripts/0_preprocess_data.sh) (if you want to add more data, see instructions below). 
- If your method requieres training, you can add the training code to the [same script where we train scVI models](scripts/train_scvi_models_allgenes.py). You can also train idependently.
- You can now create references for deconvolution with your method:
    - Add your transformation to the same datasets as seen in the [simulations script](scripts/prepare_deconvolution_sim.py) See where we highlight "Add your transformation here!" line 698.

- Each of the notebooks have a "Settings" cell at the top. Add your reference identifier to these variables, add a color in the palette, and add a name for it for the plots. You might need to adjust size if the plots don't look right depending on the number of transforms you add.
 
## **Instructions for adding more data to the analysis:**
- Start by preprocessing single cell and single nucleus datasets. You can add our own Jupyter notebook to the [notebooks folder](notebooks). Then, run the [preprocessing shell](scripts/0_preprocess_data.sh). Choose a new identifier for your dataset, add it to the data folder: data/YOURS.
- After, it's just a matter of adding your new dataset identifier to the shell scripts:
Example:
> datasets=("ADP" "PBMC" "MBC" "MSB") to datasets=("ADP" "PBMC" "MBC" "MSB" "YOURS")
- If you only want to add data to the "real bulk" analysis, add it to the shell scripts that contain "Real_ADP", and add a array to the job at the top (we only use one real bulk dataset):
Example:
> dataset=("Real_ADP") to dataset=("Real_ADP" "YOURS)