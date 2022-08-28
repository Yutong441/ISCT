# Rscipt phenotype_raw/clean_phenotype.R \
# /data/rosanderson/is_ct/phenotype_files/MGH IS Registry/211209 Mayerhofer Ischemic CT Main Ischemic Stroke Registry list.xlsx
library (tidyverse)
args <- commandArgs(trailingOnly=TRUE)
#args <- c('../data/phenotype_files/MGH IS Registry/211209 Mayerhofer Ischemic CT Main Ischemic Stroke Registry list.xlsx')
ori_data <- readxl::read_excel (args[1]) %>% data.frame()

#' The coding of some of the variables is that 1 stands for absence, 2 stands
#' for presence, and '7777', '8888' or '9999' stand for data unavailable or
#' absent. This function converts the coding into 0 for absence, 1 for
#' presence, and NA for not available
binarize <- function (xx){
        xx [xx>7700] <- NA
        xx [xx == 1] <- 0
        xx [xx == 2] <- 1
        return (xx)
}

convert_na <- function (xx){
        xx [xx>7700] <- NA
        return (xx)
} 

set_na <- function (xx, index){
        xx [index] <- NA
        return (xx)
}

bin_fea <- c('HTN', 'HLD', 'DM2', 'CAD', 'AF', 'ICH', 'IVtPA', 'IAtPA')
na_fea <- c('mRS_48h', 'mRS_dc', 'BI_dc', 'NIHSS', 'Race')
keep_fea <- c('MRN', 'Age', 'Sex', 'Onset', 'onset_date', 'TIA',
              bin_fea, na_fea, 'SBP', 'DBP', 'Subtype', 'Stay')
ori_data %>% mutate (prev_TIA = TIA) %>%
        mutate (TIA=ifelse (X211117.Mayerhofer.Power.1_Description=='TIA',1,0)) %>%
        mutate (Sex = ifelse (LU.Sex_Description=='Male', 1, 0)) %>%

        # make some of the names shorter and more readable
        mutate (ICH=Cerebral.Hemorrhage) %>%
        mutate (Subtype=Toast) %>%
        mutate (mRS_48h = X48HRmRS) %>%
        mutate (mRS_dc = Discharge.mRS) %>%
        mutate (BI_dc = Discharge.BI) %>%
        mutate (SBP = Systolic.BP) %>%
        mutate (DBP = Diastolic.BP) %>%

        # create NA fields for '7777', '8888' or '9999'
        mutate_at (bin_fea, binarize) %>%
        mutate_at (na_fea, convert_na) %>%

        # calculate age at the onset of stroke
        mutate (Age=difftime (Date.Presentation, Date.of.Birth, unit='days')) %>%
        mutate (Age = as.numeric(Age)/365) %>%

        # length of stay in the hospital
        mutate (Stay=difftime (DischargeDate, Date.Presentation, unit='days')) %>%
        mutate (Stay=as.numeric (Stay)) %>%

        # combine onset date and time into a single column
        mutate (onset_time = strftime (First.Known.Symptoms.Time, 
                                            format='%H:%M:%S')) %>%
        mutate (no_time = onset_time == '00:00:00') %>%
        mutate (onset_date = strftime (Date.Presentation,
                                               format='%Y-%m-%d')) %>%
        unite('Onset', c('onset_date', 'onset_time'), sep=' ', remove=F) %>%
        mutate (Onset = set_na (Onset, no_time)) %>%
        mutate (Onset = as.POSIXct (Onset, format = '%Y-%m-%d %H:%M:%S')) %>%
        select(one_of (keep_fea)) -> pro_data

write.csv (pro_data, 'data/original_labels/registry.csv')
