preprocess_img <- function (img_path, modality='train'){
        img_list <- list()
        for (i in 1:length(img_path)){
                prefix <- gsub ('/$', '', img_path[i])
                prefix <- gsub ('^.*/', '', prefix)
                prefix <- gsub ('_.*$', '', prefix)
                img_dat_path <- paste (img_path[i], '/CTP_act_',
                                       modality, '.csv', sep='')
                img_fea <- read.csv (img_dat_path, row.names=1) 
                colnames (img_fea) <- paste (prefix, 1:ncol(img_fea), sep='')
                img_list [[i]] <- img_fea
        }
        if (length (img_list)==1){return (img_list[[1]])
        }else{do.call ('cbind', img_list) %>% return ()
        }
}

#' Obtain dataframe for machine learning
#'
#' @param label_path path to the rds file which is a list of 2 elements: x
#' variables and y variables
#' @param img_path path to the csv file containing features extracted by deep
#' neural network
#' @examples
#' train_df <- get_img_data ('data/CTP/labels/train.rds', 
#'                  'results/CTP_re/CTP_act_train.csv')
get_img_data <- function (label_path, img_path=NULL, joining=TRUE,
                          modality='train', na_rm_by=NULL){
        label_df <- read.csv (label_path)
        label_df <- list (x=label_df [, get_xvar()], y=label_df [, get_yvar()])
        if (!is.null (na_rm_by)){label_df <- remove_na (label_df, na_rm_by)}
        if (!is.null (img_path)){
                img_fea <- preprocess_img (img_path, modality)
                if (joining){ label_df$x <- cbind (label_df$x, img_fea)
                }else{label_df$x <- img_fea}
        }
        for (i in colnames (label_df$y) ){
                if ( length (unique (label_df$y[,i])) == 2 ){
                        label_df$y [,i] <- as.factor (label_df$y[,i])
                }
        }
        if ('mRS_3m' %in% colnames (label_df$y)){
                label_df$y$mRS_clas <- as.factor (label_df$y$mRS_3m)
        }
        return (label_df)
}

train_model <- function (train_df, y_col, method='rf'){
        set.seed(42)
        seeds <- vector(mode = "list", length = 26)
        for(i in 1:25) {seeds[[i]] <- sample.int(1000, 50)}
        seeds[[26]] <- sample.int(1000,1)
        caret::trainControl(method="repeatedcv", number = 5, repeats =5, seeds=seeds,
                            preProcOptions=list(cutoff=0.75)) -> train_control 
        set.seed (42)
        caret::train(train_df$x, train_df$y[,y_col], method=method, 
                     trControl=train_control) -> model
        return (model)
}

nmse <- function (ytrue, ypred){
        mean((ytrue - ypred)^2, na.rm=T) %>% sqrt () %>% return ()
}
mae <- function (ytrue, ypred){ mean (abs(ytrue - ypred), na.rm=T) }
mae_CI <- function (ytrue, ypred){
        mae_boot <- function (xx, idx){ mean (abs(xx [idx]), na.rm=T) }
        boot_obj<- boot::boot (ytrue - ypred, mae_boot, R=100)
        boot_conf<- boot::boot.ci (boot_obj, type='norm')
        paste (sprint1 (boot_conf$t0), ' (', sprint1 (boot_conf$normal[2]), '-', 
               sprint1 (boot_conf$normal[3]), ')', sep='') %>% return ()
}

test_stat <- function (ytrue, ypred, pos_label, ycol){
        if (grepl('log', ycol)){
                ytrue <- 10^ytrue - 1
                ypred <- 10^ypred - 1
        }
        if (is.na (pos_label)){pos_label <- median (ytrue)}
        acc_df <- list(
                NMSE= nmse (as.numeric (ytrue), as.numeric (ypred) ),
                MAE=mae(as.numeric (ytrue), as.numeric (ypred) ),
                MAE_CI=mae_CI (as.numeric (ytrue), as.numeric (ypred) )
        ) %>% as.data.frame ()
        if (length (unique (ypred) ) >2 ){ 
                ypred <- ifelse (as.numeric (ypred) > pos_label, 1, 0)
                ytrue <- ifelse (as.numeric (ytrue) > pos_label, 1, 0)
        }
        if (length (unique (ypred)) >1  & length (unique (ytrue) ) >1){
                acc_df$AUC <- pROC::auc (as.numeric (ytrue), as.numeric (ypred))
                acc <- caret::confusionMatrix (as.factor (ypred), as.factor (ytrue)) 
                acc_df$accuracy <- acc$overall ['Accuracy']
                acc_df$sensitivity <- acc$byClass ['Sensitivity']
                acc_df$specificity <- acc$byClass ['Specificity']
        }else{
                acc_df$AUC <- 0.5
                acc_df$accuracy <- 0.5
                acc_df$sensitivity <- 0.5
                acc_df$specificity <- 0.5
        }
        acc_df %>% mutate_if (is.numeric, function(xx){round (xx, 3)}) %>% return ()
}

test_model <- function (model, df_list, ycol, pos_label=2.){
    N <- length (df_list)
    acc_df <- list()
    for (i in 1:N){
        set.seed (42)
        pred <- predict(model, df_list[[i]]$x)
        acc_df [[i]]<- test_stat (df_list[[i]]$y[,ycol], pred, pos_label, ycol)
    }

    acc_df <- do.call(rbind, acc_df)
    col_order <- c('method', 'mode', 'AUC', 'accuracy', 'sensitivity',
                   'specificity', 'NMSE', 'MAE', 'MAE_CI', 'ycol', 'sample')
    acc_df$method <- model$method
    acc_df$ycol <- ycol

    acc_df$mode <- c('train', paste ('test', 1:(N-1), sep=''))
    acc_df$sample <- sapply (df_list, function(i){dim(i$x)[1]})
    return (acc_df [, col_order])
}

remove_na <- function (df_list, ycol){
        remove_rows <- is.na (df_list$y [,ycol])
        return (list (x=df_list$x[!remove_rows,], y=df_list$y[!remove_rows,]))
}

train_test <- function (df_list, ycol, method, pos_label) {
        classification <- class (df_list[[1]]$y[,ycol]) == 'factor'
        proceed <- TRUE
        if (classification & method %in% c('lasso', 'blasso')){proceed <- FALSE}
        if (!classification & method %in% c('vglmAdjCat', 'plr')){proceed <- FALSE}

        N <- length (df_list)
        for (i in 1:N){
            df_list [[i]] <- remove_na (df_list [[i]], ycol)
        }
        if (proceed){
            model <- train_model (df_list[[1]], ycol, method=method)
            return (test_model (model, df_list, ycol, pos_label))
        }
}

all_models <- function (df_list, ycols=NULL, model_names=NULL, pos_label=2.,
                        save_dir=NULL){
        if (is.null (model_names)){
                model_names <- c('knn', 'svmRadial', 'svmLinear', 'lasso',
                 'gaussprRadial')
        }
        if (is.null (ycols)){get_yvar() }
        acc_list <- list ()
        if (length (pos_label)==1){pos_label <- rep (pos_label, length (ycols))}
        for (i in 1:length (model_names)){
                all_acc <- lapply (as.list (1:length (ycols)), function (j){
                        train_test (df_list, ycols[j], model_names[i], pos_label[j])
                })
                acc_list[[i]] <- do.call('rbind', all_acc)
                print (acc_list[[i]])
        }
        final_df <- do.call('rbind', acc_list)
        if (is.null (save_dir)){return (final_df)
        }else{write.csv (final_df, save_dir, quote=F, row.names=F)}
}

get_xvar <- function (){
    return (c("Age", "Sex", "Race", "HTN", "HLD", "DM2", "CAD","AF",
              'SBP', 'DBP', 'NIHSS',
              "diff_WM","diff_CSF","diff_ACA2","diff_MCA","diff_PCA",
              "diff_VB","diff_LV2","avg_GM","avg_WM","avg_CSF","avg_ACA2",
              "avg_MCA","avg_PCA","avg_VB","avg_LV2", "diff_ACA","diff_MLS",
              "diff_LLS","diff_MCAF","diff_MCAP","diff_MCAT","diff_MCAO",
              "diff_MCAI","diff_PCAT","diff_PCAO","diff_PCTP","diff_ACTP",
              "diff_BA","diff_SC","diff_IC","diff_LV","avg_GM","avg_WM",
              "avg_CSF","avg_ACA2","avg_MCA","avg_PCA","avg_VB","avg_LV2",
              "avg_ACA","avg_MLS","avg_LLS","avg_MCAF","avg_MCAP","avg_MCAT",
              "avg_MCAO","avg_MCAI","avg_PCAT","avg_PCAO","avg_PCTP",
              "avg_ACTP","avg_BA","avg_SC","avg_IC","avg_LV"))
}

get_yvar <- function (){
    return (c('LSW', 'LSW_quant', 'Stay', 'mRS_dc', 'mRS_48h', 'mRS_diff'))
}

load_all_data <- function (label_paths, img_root=NULL, joining=T, na_rm_by=NULL, 
                           xvar=NULL, impute=F){
    df_list <- list ()
    N <- length (label_paths)
    imgnames <- c('train', paste('test', 0:(N-2), sep=''))
    if (!is.null (img_root) ){
        for (i in 1:N){
            get_img_data (label_paths[i], img_root, joining=joining,
                          modality=imgnames[i], na_rm_by = na_rm_by)-> df_list [[i]]
        }
    }else{
        for (i in 1:N){
            get_img_data (label_paths[i], modality=imgnames[i])-> df_list[[i]]
        }
    }
    if (impute){
        if (is.null(xvar)){xvar <- get_xvar ()}
        caret::preProcess(df_list[[1]]$x [, xvar], 
                          method=c("center", "scale", 'knnImpute'),
                          cutoff=0.6) -> transformations
        for (i in 1:N){
            df_list[[i]]$x <- predict(transformations, df_list[[i]]$x [, xvar])
        }
    }else{
    caret::preProcess(df_list[[1]]$x, 
                      method=c("center", "scale", "nzv"),
                      cutoff=0.75) -> transformations
    for (i in 1:N){
        df_list[[i]]$x <- predict(transformations, df_list[[i]]$x )
    }}
    return (df_list)
}

load_train_test <- function (label_paths, save_dir, img_root=NULL, joining=T,
        model_names=NULL, ycols=NULL, 
        pos_label= c(4.5, 2, 4, 3, 3, 0), na_rm_by=NULL){
        out_df <- load_all_data (label_paths, img_root, joining, na_rm_by)
        all_models (out_df, ycols, model_names=model_names, save_dir=save_dir)
}
#if (method %in% c('rf', 'vglmAdjCat')){ print (caret::varImp (model)) }

plot_results <- function (save_dir, y_col='mRS_3m'){
        img_nimg <- read.csv (paste (save_dir, 'img_nimg.csv', sep='_'))
        img_nimg$inp <- 'combined'
        nimg_only <- read.csv ('results/ML/nimg_only.csv')
        nimg_only$inp <- 'non_CNN'
        img_only <- read.csv (paste (save_dir, 'img_only.csv', sep='_'))
        img_only$inp <- 'CNN_only'
        plot_df <- do.call ('rbind', list(img_nimg, nimg_only, img_only))
        method_order <- c('knn', 'svmRadial', 'svmLinear', 'lasso', 'gaussprRadial')
        plot_df %>% filter (ycol == y_col & mode == 'test' & method != 'nnet') %>%
                mutate (inp = factor (inp, levels = c('non_CNN', 'CNN_only', 'combined') )) %>%
                mutate (method= factor (method, levels=method_order)) %>% return ()
}

lm_sum <- function (xx, xcol, ycol){
    #lm(as.formula (paste (xcol, '~', ycol)), xx) %>% summary () -> lin_model
    pearson <- cor.test (xx [,xcol], xx[,ycol], method='pearson')
    spearman<- cor.test (xx [,xcol], xx[,ycol], method='spearman')
    data.frame (
        normality=shapiro.test (xx [, xcol])$p.value,
        pval_pear=pearson$p.value,
        cor_pear=pearson$estimate,
        pval_spea=spearman$p.value,
        cor_spea=spearman$estimate
    ) %>% return ()
}

lm_sum_all <- function (xx, xcols, ycol){
    xcols %>% as.list () %>% lapply (function (ii){
            lm_sum (xx, ii, ycol)
    }) -> df_list
    keep_col <- c('variable', 'correlation', 'padj')
    do.call ('rbind', df_list) %>% mutate (variable=xcols) %>% 
    mutate (pval=ifelse (normality>0.05, pval_pear, pval_spea)) %>%
    mutate (correlation=ifelse (normality>0.05, cor_pear, cor_spea)) %>%
    mutate (padj = p.adjust (pval, method='bonferroni')) %>% 
    select (all_of (keep_col)) %>% 
    mutate_if (is.numeric, function (xx){round (xx, 3)}) %>% 
    arrange (desc(correlation)) %>% return ()
}

theme_ctp <- function (fontsize=15, font_fam='arial'){
        list (ggplot2::theme(
              panel.grid.major = element_line(color='grey92'), 
              panel.grid.minor = element_line(color='grey92'), 
              panel.background = element_blank(),
              panel.border = element_blank(),

              axis.ticks.x = element_blank(),
              axis.ticks.y = element_blank(),
              axis.text.x = element_text(family=font_fam, hjust=0.5,
                                         size=fontsize, color='black'),
              axis.text.y = element_text(family=font_fam, size=fontsize,
                                         color='black'),
              axis.title.x = element_text(family=font_fam, size=fontsize),
              axis.title.y = element_text(family=font_fam, size=fontsize),
              
              legend.background= element_blank(),
              legend.key = element_blank(),
              strip.background = element_blank (),
              text=element_text (size=fontsize, family=font_fam),
              plot.title = element_text (hjust=0.5, size=fontsize*1.5,
                                         family=font_fam, face='bold'),
              aspect.ratio=1,
              strip.text = element_text (size=fontsize, family=font_fam,
                                         face='bold'),
              legend.text = element_text (size=fontsize, family=font_fam),
              legend.title = element_text (size=fontsize, family=font_fam, 
                                           face='bold') )
        )
}

theme_publication <- function(base_size=15, base_family="arial") {
      (ggthemes::theme_foundation(base_size=base_size, base_family=base_family)
       + theme(plot.title = element_text(face = "bold",
                                         size = rel(1.2), hjust = 0.5),
               text = element_text(),
               panel.background = element_rect(colour = NA),
               plot.background = element_rect(colour = NA),
               panel.border = element_rect(colour = NA),
               axis.title = element_text(face = "bold",size = rel(1)),
               axis.title.y = element_text(angle=90,vjust =2),
               axis.title.x = element_text(vjust = -0.2),
               axis.text = element_text(), 
               axis.line = element_line(colour="black"),
               axis.ticks = element_line(),
               panel.grid.major = element_line(colour="#f0f0f0"),
               panel.grid.minor = element_blank(),
               legend.key = element_rect(colour = NA),
               legend.position = "bottom",
               legend.direction = "horizontal",
               legend.key.size= unit(0.2, "cm"),
               legend.margin = unit(0, "cm"),
               legend.title = element_text(face="italic"),
               plot.margin=unit(c(10,5,5,5),"mm"),
               strip.background=element_rect(colour="#f0f0f0",fill="#f0f0f0"),
               strip.text = element_text(face="bold")
          ))
      
}

plot_lm_val <- function (lin_df, yval='R2'){
        lin_df %>% arrange (!!as.symbol (yval)) %>%
                mutate (variable= factor (variable, levels=variable)) %>%
                mutate (logP = pmin(-log10 (padj), 3) ) %>%
                ggplot (aes_string (x='variable', y=yval, fill='logP')) +
                        geom_bar (stat='identity') + 
                        scale_fill_viridis_c ()+
                        coord_flip () + theme_ctp ()
}

plot_var_imp <- function (model){
        var_imp <- caret::varImp (model)
        var_imp$importance %>% rownames_to_column ('variable') %>%
                arrange (Overall) %>%
                mutate (variable= factor (variable, levels=variable)) %>%
                ggplot (aes (x=Overall, y=variable)) +
                        geom_bar (stat='identity') + 
                        xlab ('importance') +theme_ctp ()
}

load_multimod <- function (save_dirs){
        acc_list <- list()
        for (i in 1:length (save_dirs)){
                acc_list [[i]] <- read.csv (paste (save_dirs[i], 'img_only.csv', sep='_'))
                acc_list [[i]]$modality <- gsub ('^.*/', '', save_dirs[i])
        }
        return (do.call ('rbind', acc_list))
}

rescale_g2 <- function (xx){
        xx [xx >= 3.75] <- 6
        xx [xx >= 3.25 & xx < 3.75] <- 5
        xx [xx >= 2.75 & xx < 3.25] <- 4
        xx [xx >= 2.25 & xx < 2.75] <- 3
        xx [xx < 2.25] <- 2
        return (xx)
}

rescale_mRS <- function (xx){
        xx [xx < 2] <- round (xx [xx <2], 0)
        xx [xx > 2] <- rescale_g2 (xx [xx >2])
        return (xx)
}

sprint1 <- function (xx){sprintf ('%.1f', round (xx,1))}
sprint3 <- function (xx){sprintf ('%.3f', round (xx,3))}

factor2numeric <- function (xx){
        if (is.factor (xx)){
                sum_col <- xx %>% as.character () %>% as.numeric ()
        }else{sum_col <- xx}
        return (sum_col [!is.na (sum_col)])
}

get_conv <- function (){
    c(Sex='Sex', Age='Age', HTN='Hypertension', HC='Hypercholesterolaemia',
      AF='Atrial fibrillation', HF='Heart failure', Smoke='Smoking',
      Diabetes='Diabetes',MI.Angina='Ischaemic heart disease', 
      Side='Side of stroke',
      Total='Core + penumbra', Core='Core volume', 
      Penumbra='Penumbra volume', Mismatch='Mismatch ratio', 
      BslmRS ='Baseline mRS', BslNIH
      ='Baseline NIHSS', mRS_3m = 'mRS after 3 months', 
      tPAdelay= 'tPay delay', X24hNIH = 'NIHSS after 24h', 
      ICH = 'Absence of intracranial haemorrhage') %>% return ()
}

sum_stat <- function (xdf){
        sum_list <- list ()
        for (i in colnames (xdf)){
                if (! is.character (xdf [,i]) ){
                        # convert factors to numeric
                        sum_col <- factor2numeric (xdf [, i])
                        # percentages for 2-level features
                        if (length (unique (sum_col) ) == 2 ){
                                sum_col %>% as.factor () %>% levels () -> sum_levels
                                num_lev <- sum (sum_col == sum_levels [2])
                                perc_lev <- sprint1 (mean(sum_col == sum_levels [2])*100)
                                sum_text <- paste (num_lev, ' (', perc_lev, '%)', sep='')
                        }else {
                                sum_text <- paste (
                                sprint1(mean (sum_col, na.rm=T)), ' \u00B1', 
                                sprint1(sd (sum_col, na.rm=T)), sep='')
                        }
                        sum_list [[i]] <- sum_text
                }
        }
        do.call (c, sum_list) %>% data.frame () %>% 
                rownames_to_column ('Variable') %>% 
                magrittr::set_colnames (c('Variable', 'Value')) %>%
                mutate (Variable=as.character (Variable)) -> sum_df
        conversion <- get_conv ()
        sum_df <- sum_df [match (names (conversion), sum_df$Variable),]
        sum_df$Variable  <- as.character (conversion)
        rownames (sum_df) <- sum_df$Variable
        return (sum_df %>% select (!Variable))
}

chi2_test <- function (xvec, yvec){
        xvec_pos <- sum (xvec == unique (xvec)[1])
        xvec_neg <- length (xvec) - xvec_pos
        yvec_pos <- sum (yvec == unique (xvec)[1])
        yvec_neg <- length (yvec) - yvec_pos
        xy_df <- data.frame (x=c(xvec_pos, xvec_neg), y=c(yvec_pos, yvec_neg))
        return (chisq.test (xy_df))
}

sum_stat_pval <- function (xdf, ydf){
    sum_list <- list ()
    conv_names <- names (get_conv ())
    xdf <- xdf %>% select (all_of (conv_names))
    ydf <- ydf %>% select (all_of (conv_names))
    for (i in colnames (xdf)){
        if (! is.character (xdf [,i]) ){
            # convert factors to numeric
            sum_colx <- factor2numeric (xdf [, i])
            sum_coly <- factor2numeric (ydf [, i])
            # percentages for 2-level features
            if (length (unique (sum_colx) ) == 2 ){
                    pval <- chi2_test (sum_colx, sum_coly)$p.value
            }else {
                    pval <- t.test (sum_colx, sum_coly)$p.value
            }
            sum_list [[i]] <- pval
        }
    }
    do.call (c, sum_list) %>% round (digit=3) -> sum_list
    sum_list <- data.frame (pval= sum_list)
    rownames (sum_list) <- get_conv ()
    return (sum_list)
}

# --------------------confidence interval--------------------
get_AUC_CI <- function (AUC, ypos, yneg, interval=F){
        Q1 <- AUC / (2 - AUC)
        Q2 <- 2*AUC**2 / (1 + AUC)
        SE_AUC <- sqrt((AUC*(1 - AUC) + (ypos - 1)*(Q1 - AUC**2) + 
                    (yneg - 1)*(Q2 - AUC**2)) / (ypos*yneg))
        if (interval){return (SE_AUC)
        }else{
                lower <- AUC - 1.96*SE_AUC
                upper <- AUC + 1.96*SE_AUC
                paste (sprint3 (AUC), ' (', sprint3 (lower), '-', sprint3 (upper), ')', 
                   sep='') %>% return ()
        }
}

get_acc_CI <- function (xx, num){
        interval <- sqrt (xx*(1-xx)/num)
        lower <- xx - 1.96*interval
        upper <- xx + 1.96*interval
        paste (sprint3 (xx), ' (', sprint3 (lower), '-', sprint3 (upper), ')', 
           sep='') %>% return ()
}

get_RMSE_CI <- function (xx, deg_free, p_upper=0.975, p_lower=0.025, interval=F){
        lower <- sqrt(deg_free / qchisq(p_upper, df = deg_free)) *xx
        upper <- sqrt(deg_free / qchisq(p_lower, df = deg_free)) *xx 
        if (!interval){
                paste (sprint3 (xx), ' (', sprint3 (lower), '-', sprint3 (upper), ')', 
                   sep='') %>% return ()
        }else{ return (list (upper, lower)) }
}

add_CI <- function (csv_path, ypos, yneg, save_csv=T){
        acc_df <- read.csv (csv_path)
        acc_df$AUC <- get_AUC_CI (acc_df$AUC, ypos, yneg)
        N <- as.numeric (as.character (acc_df$sample))
        acc_df$NMSE <- get_RMSE_CI (acc_df$NMSE, N -1)
        acc_df$MAE <- acc_df$MAE_CI
        acc_df %>% select (!MAE_CI) -> acc_df
        for (i in c ('accuracy', 'sensitivity', 'specificity')){
                acc_df [, i] <- get_acc_CI (acc_df [,i], ypos+yneg)
        }
        if (!save_csv) {return (acc_df) 
        }else{
                filename <- gsub ('\\.csv$', '_CI.csv', csv_path)
                write.csv (acc_df, filename, quote=F, row.names=F)
        }
}

format_table <- function (csv_path, ypos, yneg, save_csv=T,
                          exclude_col='method'){
        acc_df <- read.csv (csv_path, stringsAsFactors=F)
        if (is.numeric (acc_df$AUC)){
                acc_df <- add_CI (csv_path, ypos, yneg, save_csv=F)
        }
        new_df <- list ()
        acc_df %>% mutate_if (is.factor, as.character) -> acc_df
        for (i in 1:nrow (acc_df)){
                upper_reg <- '\\(.+-.+\\)$'
                upper_row <- gsub (upper_reg, '', acc_df [i,]) %>% trimws ()
                lower_row <- gsub ('\\)$', '', gsub ('^.*\\(', '', as.character (acc_df [i,]))
                                ) %>% trimws ()
                lower_row <- paste ('(', lower_row, ')', sep='')
                if (!is.null (exclude_col)){
                        for (j in exclude_col){
                                col_ind <- match (j, colnames (acc_df))
                                lower_row [col_ind] <- ''
                        }
                }
                new_df [[i]] <- rbind (upper_row, lower_row)
        }
        new_df <- do.call (rbind, new_df)
        colnames (new_df) <- paste (colnames (acc_df), ' (95% CI)', sep='')
        for (j in exclude_col){
                col_ind <- match (j, colnames (acc_df))
                colnames (new_df)[col_ind] <- colnames (acc_df) [col_ind]
        }
        rownames (new_df) <- 1:nrow (new_df)
        if (!save_csv) {return (acc_df) 
        }else{
                filename <- gsub ('\\.csv$', '_format.csv', csv_path)
                write.csv (new_df, filename, quote=F, row.names=F)
        }
}
