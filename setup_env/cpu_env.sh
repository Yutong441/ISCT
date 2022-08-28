#!/bin/bash
#BSUB -J test
#BSUB -o output/test-%J.out
#BSUB -e output/test-%J.err
#BSUB -q short
#BSUB -n 1
#BSUB -R rusage[mem=3000]

module load python/3.8
module load R/4.1.0
module load cmake/3.19.2
module load anaconda/4.11.0
env=ISCT

# --------------------setup python virtual environment--------------------
conda init bash
source ~/.bashrc
conda create -n $env python=3.8
module unload anaconda/4.11.0

conda activate $env
conda install numpy pandas matplotlib scikit-image ipywidgets -c conda-forge
conda install nibabel dcm2niix -c conda-forge
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=$env
conda deactivate

# --------------------install R packages--------------------
my_r=~/Documents/R_libs
mkdir $my_r
for library in [ 'tidyverse' 'caret' 'kableExtra' 'ggpubr' 'ggthemes' ]; do
    Rscript -e "install.packages ('$library',repos='http://cran.r-project.org',lib="$my_r")"
done

cat "
if [ -n $R_LIBS ]; then
   export R_LIBS="$my_r:"$R_LIBS
else
   export R_LIBS="$my_r"
   fi
" >> ~/.bashrc

# --------------------install CTseg--------------------
# white matter segmentation
cd ~/Documents
# install spm12
cd spm12/src
make distclean
make USE_OPENMP=1 && make install MEXBIN=/apps/lib-osver/matlab/2021b/bin/mex

cd spm12/toolbox
git clone https://github.com/WTCN-computational-anatomy-group/mb
cd mb
make MEXBIN=/apps/lib-osver/matlab/2021b/bin/mex

cd ../../..
git clone https://github.com/WCHN/CTseg 

# Download arterial territorial atlas from 
# https://www.nitrc.org/projects/arterialatlas
# place it on template
mkdir template
cd template
unzip Atlas_MNI.zip
