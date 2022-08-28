# summarise missing studies
source ('data_raw/utils.R')
args <- commandArgs(trailingOnly=TRUE)
filedir<- dirname (args[1])
root_dir <- args[2]
shape <- read.csv (paste (filedir, 'unreg_trim_shape.csv', sep='/'), header=F)
# V1 is the anoID, V2, V3 and V4 are height, width and depth respectively
shape %>% filter (V4 <15) %>% pull (V1) -> small_depths
cta <- paste (filedir, 'missing_studies.txt', sep='/')
if (file.exists (cta)){
        cta_df <- read.table (cta, header=F)
        small_depths <- c(small_depths, gsub ('.nii.gz$', '', cta_df[,1]))
}

ori_df <- read.csv (args[1])
conversion <- read.csv (gsub ('.csv$', '_conversion.csv', args[1]))
missingID <- conversion$patientID [match (small_depths, conversion$anoID)]

# basic filtering to reduce workload
ori_df %>% filter (num>20 & num <350) %>%
        filter (accession != 'NA') %>%
        filter (!grepl (get_exclude_str(), view, ignore.case=T)) %>%
        # recreate patientID to match those in the missing studies
        mutate (accession = gsub ('\\.0$', '', as.character (accession))) %>%
        mutate(MRN=sprintf('%07d', patientID)) %>%
        unite ('patientID', c('MRN', 'accession'), remove=F, sep='_') %>%
        filter (patientID %in% missingID) %>% 
        mutate (directory=paste (MRN, accession, mode, view,sep='/')) %>%
        mutate (directory=paste (root_dir, directory, sep='/')) %>%
        mutate (thick = infer_thick (view, num, SliceThickness)) %>%
        select (one_of (retain_col())) -> missing_df

# append the image information to the disambiguation list
missing_df$anoID <- small_depths[match (missing_df$patientID, missingID)]
dup <- read.csv (gsub ('.csv$', '_dup.csv', args[1]), row.names=1)
rbind (dup, missing_df) %>% write.csv (gsub ('.csv$', '_dup2.csv', args[1]))
