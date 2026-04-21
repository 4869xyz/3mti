# FMMDIFF
FMM-Diff: A Feature Mapping and Merging Diffusion Model for MRI Generation with Missing Modality.

## 🔧 Configuration

Modify the configuration file at (config.yml) to customize your MRI inputs.

```yaml

mri:
  # MRI sequences as input
  modalities_name: ['Flair.nii.gz','T1.nii.gz','T1c.nii.gz','T2.nii.gz' ]
  # number of modality

folder_path:
  data_store_path: "PATH/TO/DATA"

```
Each patient's folder must include all corresponding .nii.gz files.
> 📌 Make sure that the file names match those specified in `modalities_name` in your config file.
```
PATH/TO/DATA/
├── patient1/
│   ├── mri_type1.nii.gz
│   ├── mri_type2.nii.gz
│   └── ...
├── patient2/
│   ├── mri_type1.nii.gz
│   ├── mri_type2.nii.gz
│   └── ...

