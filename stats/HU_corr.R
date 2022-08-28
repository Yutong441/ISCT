# correlation between Housefield units from brain segmentation and different
# clinical outcomes
setwd('..')
library (tidyverse)
source ('ML/utils.R')
phenotype <- read.csv ('data/original_labels/all_imputed.csv')%>% 
    column_to_rownames ('anoID') %>% select (!patientID)

ycol <- 'LSW'
xcols <- c("Age", "Sex", "HTN", "HLD", "DM2", 'CAD', 'AF', 'DBP', 'SBP', 'NIHSS', 
           'diff_GM', 'diff_WM', 'diff_CSF', 'avg_GM', 'avg_WM', 'avg_CSF',
           'diff_ACA2', 'diff_MCA', 'diff_PCA', 'diff_LV2', 'diff_VB',
           'avg_ACA2', 'avg_MCA', 'avg_PCA', 'avg_LV2', 'avg_VB')
lm_df <- lm_sum_all (phenotype, xcols, ycol)
lm_df %>% slice_max (abs(correlation), n=10) %>%
    data.table::data.table() %>%
    kableExtra::kbl (format='latex', align='r') %>%
    kableExtra::kable_minimal(full_width = F)
#       variable correlation padj
# cor15  avg_CSF       0.215    0
# cor22  avg_MCA       0.163    0
# cor24  avg_LV2       0.162    0
# cor21 avg_ACA2       0.158    0
# cor23  avg_PCA       0.157    0
# cor13   avg_GM       0.138    0
