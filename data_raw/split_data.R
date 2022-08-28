# Rscript split_data.R <path to csv> <number of splits>
args <- commandArgs(trailingOnly=TRUE)
all_df <- read.csv (args[1], row.names=1)
N <- dim(all_df)[1]
step <- ceiling(N/as.numeric (args[2]))
save_dir <- gsub ('.csv$', '', args[1])
if (!file.exists (save_dir)){dir.create(save_dir)}
for (i in seq (1, N, by=step)){
        end <- pmin (i+step-1, N)
        write.csv (all_df[i:end,], 
                   paste (save_dir, '/sample', (i-1)/step, '.csv', sep=''))
}
