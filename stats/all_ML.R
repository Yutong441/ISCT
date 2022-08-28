setwd ('..')
library (tidyverse)
source ('ML/utils.R')

label_dir <- 'data/labels_mRSdc'
save_dir <- 'results/ML/NCCT'

# --------------------test all models--------------------
label_root <- paste (label_dir, c('train.csv', 'test.csv', 'val.csv'), sep='/')
load_train_test (label_root, paste (save_dir, 'mRSdc.csv', sep='_'), NULL,
    ycols='mRS_dc', pos_label=3)

results <- read.csv (paste (save_dir, 'mRSdc.csv', sep='_'))
head (results)
ggplot (results, aes_string (x='method', y='AUC', fill='mode'))+
    geom_bar (stat='identity', position=position_dodge2())+
    theme_ctp ()

# --------------------linear regression--------------------
out_df <- read.csv ('data/original_labels/all_sel.csv')
lin_df <- lm_sum_all (out_df, get_xvar(), 'mRS_dc')
lin_df %>% filter (!grepl ('(^avg|^diff)', variable)) %>% plot_lm_val () 
#+theme (aspect.ratio=1.7, plot.margin = unit(c(0,0,0,0),"mm"))
write.csv (lin_df, 'report/1lm_non_imaging.csv')

