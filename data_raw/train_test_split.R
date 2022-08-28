library (tidyverse)
source ('ML/utils.R')
args <- commandArgs(trailingOnly=TRUE)
df_path <- args[1]
save_dir <- args[2]
variable <- args[3] # e.g., 'LSW_bool'
thromb <- as.numeric (args[4]) # 0 for no discrimination
thres <- as.numeric (args[5]) #threshold for distributing the variable into datasets

#df_path <- '../data/original_labels/all_sel.csv'
#save_dir <- '../data/labels_Stay'
#variable <- 'LSW'
#throm <- 0

read_csv (df_path) %>% data.frame () %>%
    filter (!is.na (!!as.symbol (variable) )) -> pdata

if (variable %in% c('LSW', 'Stay')){
    pdata %>% filter (!!as.symbol (variable)>=0) -> pdata
}
if (args[4] == 1){ pdata %>% filter (IVtPA == 1) -> pdata }

# --------------------train test split--------------------
# create test set
if (!dir.exists (save_dir)){dir.create (save_dir)}
set.seed (100)
ind <- caret::createDataPartition (pdata [, variable]>thres, p=0.8, list=F)

# create train and val sets
set.seed (100)
ind2 <- caret::createDataPartition (pdata[ind, variable]>thres, p=0.8, list=F)

train_df <- pdata[ind,][ind2,]
val_df <- pdata[ind,][-ind2,]
test_df <- pdata[-ind,]

print (paste (dim (train_df)[1], 'train samples'))
print (paste (dim (val_df)[1], 'val samples'))
print (paste (dim (test_df)[1], 'test samples'))

write.csv (test_df, paste (save_dir, 'test.csv', sep='/'))
write.csv (train_df, paste (save_dir, 'train.csv', sep='/'))
write.csv (val_df, paste (save_dir, 'val.csv', sep='/'))
