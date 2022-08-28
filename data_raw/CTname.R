source ('data_raw/utils.R')
#run CTname.py to obtain a file containing the scan accession of each patient
args = commandArgs(trailingOnly=TRUE)
fil_df <- filter_scans (args[1])
fil_df <- filter_meta (fil_df, args[2])
fil_df %>% dplyr::mutate (directory=paste (args[3], directory,sep='/')) -> fil_df

dupID <- fil_df %>% filter (duplicated (patientID)) %>% pull (patientID)
fil_df %>% dplyr::filter (patientID %in% dupID) %>%
        write.csv(gsub ('.csv', '_dup.csv', args[1]))
fil_df %>% filter (!patientID %in% dupID) %>%
        write.csv (gsub ('.csv', '_path.csv', args[1]))

rbind (read.csv (gsub ('.csv', '_path.csv', args[1])),
       read.csv (gsub ('.csv', '_dup.csv', args[1]))
) %>% dplyr::filter (!duplicated (patientID)) %>%
        select (anoID, patientID) %>% 
        write.csv (gsub ('.csv', '_conversion.csv', args[1]))
