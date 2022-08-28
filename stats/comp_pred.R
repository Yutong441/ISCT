# compare predicted and actual variables
setwd ('..')
library (tidyverse)
library (grid)
source ('stats/utils.R')
source ('stats/arrange_figures.R')
save_dir <- 'results/figures/'

# --------------------age--------------------
model_dir <- 'results/dense_age_lr_20220821065138/'
ypred <- read.csv (paste (model_dir, 'NCCT_tmp/ypred_test1.csv', sep='/'))
test_df <- read.csv ('data/labels/test.csv')
test_df %>% rename(true_age = Age) %>%
    mutate (pred_age = ypred$X0) -> plot_df

plot_df %>% mutate (diff_age = (true_age - pred_age)^2) %>%
    pull (diff_age) %>% mean() %>% sqrt()
#5.45

ggplot (plot_df, aes (x=true_age, y=pred_age))+
    geom_point (size=3, shape=21, color='white', fill='black')+
    geom_smooth (method='lm') +
    xlab ("Patient's age [years]")+
    ylab ("Predictd patient's age [years]")+
    theme_publication () -> p2

#stats
cor.test (plot_df$true_age, plot_df$pred_age)
# t = 24.385, df = 282, p-value < 2.2e-16
# 0.8235985 

table (plot_df$true_age>70)
# FALSE  TRUE 
#   122   162 
pROC::auc(as.numeric (plot_df$true_age>70), as.numeric (plot_df$pred_age>70))
# Area under the curve: 0.8451
get_AUC_CI (0.8451, 162, 122)
# [1] "0.845 (0.801-0.890)"

# --------------------LSW--------------------
model_dir <- 'results/dense_reg_regi/'
ypred <- read.csv (paste (model_dir, 'NCCT_tmp/ypred_test1.csv', sep='/'))
test_df <- read.csv ('data/labels/test.csv')
test_df %>% rename(true_time = LSW) %>%
    mutate (pred_time = ypred$X0 ) -> plot_df
ggplot (plot_df, aes (x=true_time, y=pred_time))+
    geom_point (size=3, shape=21, color='white', fill='black')+
    geom_smooth (method='lm') +
    xlab ("Time from onset [hours]")+
    ylab ("Predicted time from onset [hours]")+
    theme_publication () -> p3

#stats
cor.test (plot_df$true_time, plot_df$pred_time)
# t = 8.962, df = 282, p-value < 2.2e-16
# 0.4708257 

table (plot_df$true_log_time>1.5)
# FALSE  TRUE 
#   161   123 
pROC::auc(as.numeric (plot_df$true_time>4.5), plot_df$pred_time)
# Area under the curve: 0.6402
get_AUC_CI (0.640, 123, 161)
# [1] "0.640 (0.575-0.705)"
plot_df %>% mutate (diff_time = abs(true_time - pred_time)) %>% 
    pull (diff_time) %>% median()

# --------------------make figure--------------------
p1 <- png::readPNG (paste (save_dir, 'patient_selection.drawio.png', sep='/'))
plot_list <- list(p1, p3, p2)
lay_mat <- matrix(c(1, 2, 
                    1, 3),
                  nrow=2) %>% t()
arrange_plots (plot_list, paste (save_dir, 'figure1.pdf', sep='/'), lay_mat, 
    plot_width=6, plot_height=5, panel_spacing=0.1)
