library(glmnet)
if (!exists('RUNNER_BASE')) RUNNER_BASE='/usr1/finance/backedup/git/glmnet_runner'
source(sprintf("%s/util.r", RUNNER_BASE))
# source("/afs/cs.cmu.edu/user/brendano/sw/conplot/conplot.r")
# conplot(cumsum(rnorm(100)))


load_vol = function() {
  vol <<- NULL
  for (year in 1996:2006) {
    before = read.table(sprintf("%s/../clean_vol/%s.logvol.-12.txt", RUNNER_BASE, year), stringsAsFactors=F)
    after = read.table(sprintf("%s/../clean_vol/%s.logvol.+12.txt", RUNNER_BASE, year), stringsAsFactors=F)
    stopifnot(all(  before$V2 == after$V2  ))
    x =  data.frame(year=year, before=before$V1, after=after$V1, kfile=before$V2)
    vol <<- lax_rbind(vol,x)
  }
}

# load_text = function(years) {
#   texts <<- load_data("../feat3","tf.num",years=years,log=TRUE)
# }

load_data = function(dir, ext, years=1996:2006, log=FALSE) {
  data = list()
  morecmd = if (log) "mawk '{print $1,$2,log(1+$3)}'" else "cat"
  vocab = readLines(sprintf("%s/vocab",dir))
  for (year in years) {
    cmd = sprintf("cat %s/%s.%s.mm; (cat %s/%s.%s | %s)", dir,year,ext, dir,year,ext, morecmd)
    print(cmd)
    data[[year]] = readMM(pipe(cmd))
    data[[year]] = as(data[[year]], 'dgCMatrix')
    colnames(data[[year]]) = vocab[1:ncol(data[[year]])]
  }
  gc()
  data
}

multiyear = function(yeardata, years) {
  # slow compared to concatting on the commandline .. but then have to fool around with .mm headers
  do.call(rBind, lapply(years, function(y) yeardata[[y]]))
}

bind_xy = function(years, nontext=NA, target='after', more_nontext_data=NULL) {
  stopifnot(target %in% names(vol))
  # cat("Binding text + nontext for ",years,"\n")
  args = list(multiyear(texts, years))
  print(dim(args[[1]]))
  num_text_vars = ncol(args[[1]])
  nontext_data = vol
  if (!is.null(more_nontext_data)) {
    stopifnot('kfile' %in% colnames(more_nontext_data))
    nontext_data = merge(nontext_data, more_nontext_data, 
      by='kfile', all.x=T,all.y=F, suffixes=c("","__dup"))
  }
  if (!is.null(nontext) && is.na(nontext)) {
    extras = colnames(nontext_data)
    extras = setdiff(extras, colnames(vol))
    extras = extras[!bgrep("__dup$",extras)]
    nontext = c('before', extras)
  }
  cat("Nontext variables:\n")
  print(nontext)
  for (nt_var in nontext) {
    name = sprintf("NT_%s", nt_var)
    args[[name]] = nontext_data[nontext_data$year %in% years, nt_var]
  }
  x = do.call(cBind, args)
  y = nontext_data[nontext_data$year %in% years, target]
  list(x=x, y=y, num_text_vars=num_text_vars, years=years)
}

mse = function(...) UseMethod("mse")

mse.numeric = function(x,y) mean((x-y)**2)

mse.lm = function(model,data=NULL) {
  if (is.null(data)) {
    cat("Using training data mse\n")
    mean( model$residuals**2 )
  } else {
    pred = predict(model, data)
    predvar = as.c(model$terms[[2]])
    cat("Using prediction var ",predvar,"\n")
    mse(pred, data[,predvar])
  }
}

rsq = function(pred,y) {
  1 - mse(pred,y) / var(y)
}

history_train_test = function(target_year, window) {
  train=subset(vol,year %in% (target_year-window):(target_year-1))
  m=lm(after~before,train)
  test=subset(vol, year==target_year)
  mse(predict(m,test), test$after)
}

history_train_test_0bias = function(target_year, window) {
  train=subset(vol,year %in% (target_year-window):(target_year-1))
  m=lm(after~0+before,train)
  test=subset(vol, year==target_year)
  mse(predict(m,test), test$after)
}


# vim:ft=r
