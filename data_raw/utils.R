library (tidyverse)
# --------------------image naming--------------------
get_exclude_str <- function (){
        contrast_str <- 'cta|iodine|angi|perfusion|cine|injection'
        view_str <- 'sag|cor|parallel|perpendicular' #sagittal vs coronal view
        recon_str <- 'bone|overlay|mip|virtual|recon|scout|topo|surview|monitor|screensave'
        region_str <- 'car|cow|neck|cca|skull'
        return (paste ('(', contrast_str, '|', view_str, '|', 
                       recon_str, '|', region_str, ')', sep=''))
}

retain_col <- function (){
        return (c('anoID','MRN', 'num', 'view', 'mode', 'accession',
                  'patientID', 'AcquisitionTime', 'scan_date', 'thick',
                  'DeviceSerialNumber', 'directory', 'out_file'))
}

#' Infer the thickness of CT slices based on naming
infer_thick <- function (xx, slices, initial){
        thickness <- as.numeric (initial)
        thickness [is.na (thickness)] <- 1000
        for (i in c(0:10)){
                regex <- paste (i, 'MM', sep='')
                for (j in c(0:10)){
                        if (!(i==0 & j==0)){
                                except_num <- '([a-z]|_|^)'
                                regex <- paste (except_num, i, '_', j, except_num, sep='')
                                thickness [grepl (regex, xx, ignore.case=T)] <- i+0.1*j
                        }
                }
        }
        for (i in c(1:10)){
                regex <- paste (i, '([a-z]|_)?mm', sep='')
                thickness [thickness == 1000 & grepl (
                        regex, xx, ignore.case=T)] <- i
        }
        thickness [grepl ('1_25_?mm', xx, ignore.case=T)] <- 1.25
        thickness [grepl ('625_?mm', xx, ignore.case=T)] <- 0.625
        # scans with slice number between 50 and 100 are likely to be between
        # 2.5 to 3mm thick
        thickness [thickness == 1000 & slices >20 & slices <50] <- 5
        thickness [thickness == 1000 & slices >=50 & slices <100] <- 3
        thickness [thickness == 1000 & slices >=100 & slices <250] <- 1
        thickness [thickness == 1000 & slices >=250] <- 0.5
        return (thickness)
}

#' Certain features in the names of the scan suggest it is more or less likely
#' to be the correct scan
assign_priority <- function (xx){
        priority1 <- rep (0, length(xx))
        priority2 <- rep (0, length(xx))
        priority3 <- rep (0, length(xx))
        priority4 <- rep (0, length(xx))
        priority5 <- rep (0, length(xx))
        priority7 <- rep (0, length(xx))

        priority1 [grepl ('thins|std|stnd|standard', xx, ignore.case=T)] <- 1
        priority2 [grepl ('5mm|5_mm|i__head', xx, ignore.case=T)] <- 1.5
        # asir tends to be used in contrast CT
        # similarly contrast CT tends to indicate kV values
        # images containing 'S3','S4','A4' are the same as those without, so they
        # can be removed if they are redundant (I have checked multiple examples)
        priority3 [grepl ('(asir|kv|delay|2_5_mm)', xx, ignore.case=T)] <- -0.5
        priority4 [grepl ('(s4|s3|a4|a3|soft|filter|axial)', xx, ignore.case=T)] <- -0.5

        # H60s tends to contain low resolution images, similarly Hr64
        priority5 [grepl ('(reformat|processed|subacute|smart_prep|h60s|hr64|j70|h70|j80)', 
                          xx, ignore.case=T)] <- -3

        # higher radiation seems to produce better resolution
        priority6 <- gsub ('kV', '', str_extract (xx, '[0-9]+kV') ) %>% as.numeric()
        priority6 [is.na (priority6)] <- 0
        priority7 [grepl ('^[0-9]+$', xx)] <- -0.1 # bad naming system
        return (priority1 + priority2 + priority3 + priority4 + priority5 +
        priority6/1000 + priority7)
}

anonymise <- function (df_inp, patient){
        patient_MRN <- unique (df_inp [, patient])
        N <- length (patient_MRN)
        print_digit <- paste ('%0', ceiling(log10(N)), 'd', sep='')
        df_inp$anoID <- c(1:N) [match (df_inp [,patient], patient_MRN)]
        df_inp %>% mutate (anoID=paste ('MGH', sprintf(print_digit, anoID),
                                        sep='')) %>% return ()
}

filter_scans <- function (filename){
        exclude_str <- get_exclude_str ()
        imgname <- read_csv (filename) %>% 
                data.frame () %>%  
                mutate (accession = gsub ('\\.0$', '', as.character (accession))) %>%
                mutate (num = as.numeric (num)) %>%
                filter (accession != 'NA') %>%
                filter (num>20 & num<350)

        # with respect to different accession numbers
        imgname <- imgname %>% mutate (MRN = sprintf ('%07d', patientID)) %>%
                mutate (patientID = paste (MRN, accession, sep='_'))
        imgname <- anonymise (imgname, 'patientID')
        imgname %>% pull (patientID) %>% unique () %>% length () -> ori_num

        imgname %>% filter (scan_date != 'no data') %>%
                filter (!grepl (exclude_str, SeriesDescription, ignore.case=T)) %>%
                filter (!grepl ('(derive|second|reformat|average|mip)', 
                                ImageType, ignore.case=T)) %>%
                #filter (!grepl ('neck', ProtocolName, ignore.case=T)) %>%
                filter (!grepl (exclude_str, view, ignore.case=T)) %>%
                # filter out contrast image, keep noncontrast
                filter (!(grepl ('trast', view, ignore.case=T) & !grepl (
                        'non', view, ignore.case=T))  ) -> fil_df

        fil_df %>% mutate (thick = infer_thick (view, num, SliceThickness)) %>%
                mutate (priority= assign_priority (view)) %>%
                mutate (directory=paste (MRN, accession, mode, view,sep='/')) %>%
                group_by (patientID) %>%
                slice_max (priority) %>% 
                data.frame () -> fil_df
        fil_df %>% pull (patientID) %>% unique () %>% length ()-> new_num
        print (paste ('originally ', ori_num, ' patients, now ', new_num, sep=''))
        print (paste (dim(fil_df)[1], 'scans left'))
        return (fil_df %>% select (all_of (retain_col())))
}

#' Only include those scans that have corresponding labels (for training or
#' testing)
filter_meta <- function (fil_df, meta_path){
        read_csv(meta_path) %>% data.frame() %>%
                mutate (MRN = sprintf ('%07d', MRN)) %>%
                unite('patientID', c('MRN', 'Accession_Number'), 
                      remove=F, sep='_') -> metadata
        fil_df %>% filter (patientID %in% metadata$patientID) %>% return ()
}
