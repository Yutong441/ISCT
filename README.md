# Predicting sympton onset time from NCCT
## Dataset
Preprocessing is done is 4 steps:
1. `process_init.sh` identifies NCCT scans and performs skullstripping
2. `view/disambiguate.ipynb` selects NCCT scans that the first step failed to assign
3. `process_final.sh` processes the manually selected scans
4. `quantify.sh` obtains Hounsfield intensity in different arterial territories

## CNN
Refer to `setup_env` folder for installatin instruction
Training and testing a model can be done by:
```bash
conda activate ISCT
python utils/one_exp.py
```
Refer to `utils/one_exp.py` for how to specify configs
To test multiple model variations, refer to the scripts in the `train` folder
