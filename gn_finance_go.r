#!/usr/bin/env Rscript
## to debug:  source("go_year.r", echo=T)
options(echo=T)
RUNNER_BASE = "/usr1/finance/backedup/git/glmnet_runner"
source(sprintf("%s/r.r", RUNNER_BASE))      # R doesn't have __FILE__ equivalent, sad

config = get_config(final_test=2001, alpha=1, dfmax=5000, year_window=5,
          refine=TRUE,
          input_dir="/usr2/finance/reproducible/feat_bigrams/prune_30k",
          input_ext="tf.num",
          log=TRUE,
          more_nontext_data_file="",
          nohist=FALSE,
          outfile="results.RData")

cat("\n")
if (file.exists(f<- sprintf("%s/%s.relprune", config$input_dir, config$final_test))) {
  cat("Switching input dir to the relprune subdir\n")
  config$input_dir = f
}

cat("Config\n")
print(config)
attach(config)

my_bind_xy = if (nohist) {
  function(...) bind_xy(..., nontext=NULL)
} else if (more_nontext_data_file != "") {
  more_nontext_data = read.csv(more_nontext_data_file, stringsAsFactors=F)
  function(...) bind_xy(..., more_nontext_data=more_nontext_data)
} else { 
  bind_xy
}
# print(my_bind_xy)

year_config = list()
year_config$final_test = final_test
# final split, e.g. 01-05 => 06  or  96-00 => 01
year_config$final_train = (final_test - year_window):(final_test-1)
# dev split, e.g. 01-04 => 05  or  96-99 => 00
year_config$dev_test = final_test-1
year_config$dev_train = (final_test - year_window):(final_test-2)

cat("Year split config\n")
print(year_config)
attach(year_config)

cat("\nLoading data\n")
load_vol()
texts = timeit( load_data(input_dir, input_ext, years=c(final_train,final_test), log=config$log) )

## Train the dev split
data = my_bind_xy(dev_train)
penalty.factor = c(rep(1, data$num_text_vars), rep(0, ncol(data$x) - data$num_text_vars))
gc()
cat("\nDev Training set years=",data$years, "  dim=",dim(data$x),"\n")
cat("\nTraining starting now ",format(Sys.time(),"%Y-%m-%d %H:%M:%S %Z"),",  Should be <30 minutes.\n")
dev_m = timeit( glmnet(data$x, data$y, alpha=alpha, dfmax=dfmax, lambda.min=1e-4, penalty.factor=penalty.factor) )

test_data = my_bind_xy(dev_test)
cat("\nDev Test years=",data$years,"  dim=",dim(test_data$x),"\n")
dev_pred = predict(dev_m, test_data$x)

err = sapply(1:ncol(dev_pred), function(i) mse(dev_pred[,i], test_data$y))


path_info = data.frame(df=dev_m$df, lambda=dev_m$lambda, err)
#if (refine) {
  coarse_path_info = path_info
  coarse_err = err; rm(err)
  best3 = which(rank(coarse_err)<=3)
  best3_lambda_bounds = c(min(dev_m$lambda[best3]), max(dev_m$lambda[best3]))
  best3_dfrange = max(dev_m$df[best3]) - min(dev_m$df[best3])
  cat("Refining grid search on best 3 path points",best3,"\n")
  fine_m = timeit(glmnet(data$x,data$y, alpha=alpha, dfmax=dfmax,
    lambda=seq(best3_lambda_bounds[1], best3_lambda_bounds[2], length.out=2*best3_dfrange)))
  fine_pred = predict(fine_m, test_data$x)
  fine_err = sapply(1:ncol(fine_pred), function(i) mse(fine_pred[,i], test_data$y))
  fine_path_info = data.frame(df=fine_m$df, lambda=fine_m$lambda, err=fine_err)
  coarse_path_info$run='coarse'
  fine_path_info$run='fine'
  path_info = rbind(coarse_path_info, fine_path_info)
  path_info = path_info[order(-path_info$lambda),]
  row.names(path_info)=NULL
#}
path_info$best = with(path_info, ifelse(1:nrow(path_info) == which.min(err), "** BEST **"," "))
cat("\nLambda path w/ heldout-dev-error\n")
print(path_info)
# conplot(path_info$err)
cat("\nIf the above error column doesn't look U-shaped, something is wrong. [e.g.i dfmax too low and/or lambda.min too high]\n")

dev_best_lambda = path_info$lambda[which.min(path_info$err)]

rm(data); rm(test_data)
gc()
with(list(data=NA),
  save.image("checkpoint.RData"))


## Real run now
data = my_bind_xy(final_train)
penalty.factor = c(rep(1, data$num_text_vars), rep(0, ncol(data$x) - data$num_text_vars))
gc()
cat("\nFinal training years=",data$years,"  dim=",dim(data$x),"\n")

cat("\nTraining, should be <10 minutes starting",format(Sys.time(),"%Y-%m-%d %H:%M:%S %Z"),"\n")
final_m = timeit( glmnet(data$x, data$y, lambda=dev_best_lambda, nlambda=1, alpha=alpha, dfmax=dfmax, penalty.factor=penalty.factor) )
print(final_m)
cat("\n\n")
cat("Final DF =", final_m$df, "\n")

test_data = my_bind_xy(final_test)
cat("\nFinal Test years=",data$years,"  dim=",dim(test_data$x),"\n")
final_pred = predict(final_m, test_data$x)[,1]
final_mse = mse(final_pred, test_data$y)

cat("Final MSE =", final_mse,"\n")

rm(data, test_data, texts)
gc()
vars_to_save = ls()[ grep("config|final_|dev_|refine_|path_|input_", ls()) ]
cat("Saving: ",vars_to_save,"\n")
save(list=vars_to_save, file=outfile)
printf("\n\nLoad in an R session later with:
  load('%s')
  ls()
  for (name in ls()){ print(name); str(environment()[[name]]) } \n\n\n", outfile)

