library (tidyverse)
root <- '../data/original_labels/'
pro_data <- read.csv ( paste (root, 'registry.csv', sep='/'), row.names=1)

# ====================Imaging metadata====================
# compile imaging metadata derived from dicom tags
# by combining automatically selected scans (first pass) and 
# manually selected scans (second pass)
first_pass <- read.csv (paste (root, 'dicom_path.csv', sep='/'), row.names=1)
second_pass <- read.csv (paste (root, 'dicom_disamb2.csv', sep='/'), row.names=1)
# remove scans from the first pass that have been reselected in the second pas
first_pass %>% filter (!anoID %in% second_pass$anoID) -> first_pass
imaging <- plyr::rbind.fill (first_pass, second_pass) %>%
        mutate (MRN = sprintf ('%07d', MRN)) 

# ====================Generate date per visit====================
# assign visit
# 1. make the MRN unique
# 2. extract the unique index, assign it to visit
# 3. spread the column: datetime x visit
# 4. append the dataframe to the imaging df
# 5. subtract image scan date from the dataframe 
# 6. choose the one with least time difference (use apply)
# 7. append the index back to MRN
# 8. assign the visit info to the imaging df

pro_data %>% mutate (MRN = sprintf ('%07d', MRN)) %>%
       filter (MRN %in% imaging$MRN) %>%
       mutate (uniqID = make.unique (MRN)) -> pro_data

# create a dataframe by datetime x visit
# visit number starts from 0, may not reflect the order of visit,
# simply an index of assignment
pro_data %>% mutate (Visit = str_extract (uniqID, '\\.[0-9]+$')) %>%
        mutate (Visit = gsub('^\\.', '', Visit)) %>%
        mutate (Visit = ifelse (is.na (Visit), 0, Visit)) %>%
        mutate (Visit = paste ('Visit', Visit, sep='')) %>%
        select (MRN, onset_date, Visit) %>%
        spread (Visit, onset_date) -> visit_df

# align the dataframe with the imaging metadata
visit_df [match (imaging$MRN, visit_df$MRN),] %>% 
        magrittr::set_rownames(imaging$anoID) %>%
        select(!MRN) -> visit_date
# compute time difference with the scan date
for (i in 1:dim(visit_date)[2]){
        visit_date[,i] <- difftime (imaging$scan_date, visit_date[,i], 
        unit='days') %>% as.numeric() %>% abs()
}
# choose the smallest absolute time difference
abs_min <- function(x){which (x==min(x,na.rm=T)) [[1]]}
apply(visit_date, 1, abs_min) -> sel_visit

# re-create the uniqID used to distinguish different visits in `pro_data`
imaging %>% mutate (Visit = sel_visit [match (imaging$anoID, names(sel_visit))]) %>%
        # R indexing starts from 1, needs to minus 1
        mutate (Visit = as.character (Visit-1)) %>%
        unite('uniqID', c('MRN', 'Visit'), remove=F, sep='.') %>%
        mutate (uniqID = gsub ('\\.0$', '', uniqID)) -> img_merge

pro_merge <- pro_data [match (img_merge$uniqID, pro_data$uniqID),] %>% 
        select (-uniqID, -MRN)
keep_col <- c('anoID', 'patientID', 'Age', 'Sex', 'Race', 'TIA', 'HTN', 'HLD',
              'DM2', 'CAD', 'AF', 'ICH', 'IVtPA', 'IAtPA', 'mRS_48h', 'mRS_dc',
              'BI_dc', 'NIHSS', 'DBP', 'SBP', 'Subtype', 'Stay', 'LSW')
cbind (img_merge, pro_merge) %>%
        mutate (scan_date=as.POSIXct (scan_date, format="%Y-%m-%d %H:%M:%S")) %>%
        mutate (Onset = as.POSIXct (Onset, format="%Y-%m-%d %H:%M:%S")) %>%
        mutate (LSW = difftime (scan_date, Onset, units='hours')) %>%
        mutate (LSW = as.numeric (LSW)) %>%
        # if LSW is smaller than 0, it means that the date differs by 1 thus
        # plus 24 hours
        mutate (LSW = ifelse (LSW<0, 24+LSW, LSW)) %>%
        select (one_of (keep_col)) %>%
        write.csv(paste (root, 'all.csv', sep='/'))
