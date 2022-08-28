# convert CNN performance to latex table
setwd('..')
library(tidyverse)
library (grid)
source ('stats/utils.R')
source ('stats/arrange_figures.R')

sel_col <- c('name', 'AUC', 'RMSE')
models <- data.table::fread ('results/summary.csv') %>%
    mutate (RMSE =get_RMSE_CI (sqrt (MSE_test0), 283)) %>%
    mutate (AUC = get_AUC_CI (AUC_test0, 162, 122)) %>%
    select (all_of (sel_col))

# dataset choice
models %>% filter(grepl ('^dense_reg', name)) %>%
    mutate (name=c('unregistered', 'registered', 'unregistered*')) %>%
    arrange (name) %>%
    magrittr::set_colnames (c('dataset', 'AUC', 'MSE')) %>%
    kableExtra::kbl (format='latex', align='r') %>%
    kableExtra::kable_minimal(full_width = F)

# depth choice
models %>% filter(grepl ('depths', name) | name == 'dense_reg_regi') %>%
    mutate (name=c(18, 20, 22)) %>%
    arrange (name) %>%
    magrittr::set_colnames (c('image depths', 'AUC', 'MSE')) %>%
    kableExtra::kbl (format='latex', align='r') %>%
    kableExtra::kable_minimal(full_width = F)

# model choice
models %>% filter(grepl ('^_model_type', name) | name == 'dense_reg_regi') %>%
    filter (name != '_model_type_0') %>%
    arrange (name) %>%
    mutate (name=c('densenet169', 'resnet18', 'resnet34', 'CNN (5 layer)', 
                   'CNN (6 layer)', 'densenet121')) %>%
    arrange (name) %>%
    kableExtra::kbl (format='latex', align='r') %>%
    kableExtra::kable_minimal(full_width = F)

# loss function choice
models %>% filter (grepl ('loss', name)| name == 'dense_reg_regi') %>%
    mutate (name=c('regression', 'regression (log)', 'classification')) %>%
    arrange (name) %>%
    kableExtra::kbl (format='latex', align='r') %>%
    kableExtra::kable_minimal(full_width = F)

# --------------------Overfitting--------------------
train_acc <- data.table::fread ('results/dense_reg_regi/NCCT_metric_train.csv')
test_acc <- data.table::fread ('results/dense_reg_regi/NCCT_metric_test.csv')
acc <- data.frame (train=train_acc$AUC, validation=test_acc$AUC)
acc$epoch <- seq(1, 2*dim(acc)[1], by=2)
acc %>% gather (key='dataset', value='AUC', -epoch) -> acc

cairo_pdf ('results/figures/AUC_epoch.pdf', width=6, height=6)
ggplot (acc, aes(x=epoch, y=AUC, fill=dataset, color=dataset)) +
    geom_point (shape=21, size=3) +
    geom_smooth () +
    theme_publication () 
dev.off()

# --------------------time to symptom onset--------------------
all_df <- data.table::fread ('data/original_labels/all_sel.csv')
ggplot (all_df, aes (x=LSW))+
    geom_histogram (fill='black', color='white')+
    theme_publication () -> p1
ggplot (all_df, aes (x=LSW))+
    geom_histogram (fill='black', color='white')+
    scale_x_log10()+
    theme_publication () -> p2
arrange_plots (list(p1,p2), 'results/figures/LSW_distri.pdf', matrix(c(1,2), nrow=1))
