setwd ('..')
library (tidyverse)
all_df <- data.table::fread ('data/original_labels/all_sel.csv') %>% data.frame()
all_df %>% pull (Sex) %>% table ()
all_df %>% pull (Age) %>% mean ()
all_df %>% filter (LSW <12) %>% dim()

cor.test (all_df$LSW, all_df$NIHSS, use='na.or.complete', method='spearman')
all_df %>% ggplot (aes(x=LSW, y=NIHSS)) +
    geom_point()+
    geom_smooth (method='lm')

median(all_df$mRS_dc, na.rm=T)
quantile(all_df$mRS_dc, 0.25, na.rm=T)
quantile(all_df$mRS_dc, 0.75, na.rm=T)
