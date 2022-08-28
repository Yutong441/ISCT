# remove unavailable scans from the phenotype dataframe 
library (tidyverse)
args <- commandArgs(trailingOnly=TRUE)
df_path <- args[1]
shape_path <- args[2]
seg_path <- args[3]

convert_quantile <- function (xx){
    yy <- xx
    yy [xx>=0 & xx<1.5] = 0
    yy [xx>=1.5 & xx<3] = 1
    yy [xx>=3 & xx<4.5] = 2
    yy [xx>=4.5 & xx<8] = 3
    yy [xx>8] = 4
    return (yy)
}

meta <- read_csv (df_path) %>% data.frame ()
# filter away images that are not present 
shape <- read.csv (shape_path, header=F)
shape %>% filter (V4 >15) %>% pull (V1) -> selected

meta %>% select (!one_of ('...1')) %>%
    filter (anoID %in% selected) %>%
    mutate (LSW_bool = ifelse (LSW>4.5,1,0)) %>%
    mutate (LSW_log = log(LSW)) %>%
    mutate (LSW_quant = convert_quantile (LSW)) %>%
    mutate (mRS_diff = mRS_dc - mRS_48h) %>%
    mutate (Stay_log= log (Stay+1e-3)) %>%
    mutate (NIHSS_log = log (NIHSS + 1e-3)) -> meta

# append segmentation info
if (!is.null (seg_path)){
    seg_df <- read.csv (seg_path, row.names=1)
    meta <- cbind (meta, seg_df [match (meta$anoID, rownames(seg_df)),])
}
write.csv (meta, paste (dirname (df_path), 'all_sel.csv', sep='/'),row.names=F)

# --------------------imputation--------------------
xvar <- c('Age', 'Sex', 'Race', 'HTN', 'HLD', 'DM2', 'CAD', 'AF', 'DBP', 'SBP', 'NIHSS')
ymeta <- meta %>% select (!all_of (xvar))
xmeta <- meta %>% select (all_of (xvar))
caret::preProcess(xmeta, method=c("center", "scale", 'knnImpute'),
                  cutoff=0.6) -> transformations
xmeta <- predict (transformations, xmeta)
meta <- cbind (xmeta, ymeta)
write.csv (meta, paste (dirname (df_path), 'all_imputed.csv', sep='/'),row.names=F)
