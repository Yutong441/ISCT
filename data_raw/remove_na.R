library (tidyverse)
args <- commandArgs(trailingOnly=TRUE)
xdf <- read.csv (args[1])
xdf %>% filter (!grepl ('^NA_', directory)) %>%
        write.csv (gsub ('.csv$', '2.csv', args[1]))
