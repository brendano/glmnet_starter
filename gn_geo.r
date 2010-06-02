# Text regressions with glmnet
# For 2d space: latitude and longitude as completely separate models.
# Yes this is really dumb.
# brendan o'connor, June 2010 or so, brenocon@gmail.com / http://anyall.org

library(glmnet)
library(Matrix)
source("util.r")  # http://github.com/brendano/dlanalysis/raw/master/util.R

library(maps)   ## then do map('usa')   before further plotting

load_features = function(filename) {
  # filename     is (row, column, value)  triples.
  # filename.mm  is the stupid MatrixMarket header
  
  mm_file = sprintf("%s.mm", filename)
  if (!file.exists(mm_file))  stop(sprintf("Need to create %s MatrixMarket header file  e.g. via http://github.com/brendano/slmunge/blob/master/coord2mm", mm_file))
  cmd = sprintf("cat %s; pv %s",  mm_file, filename)
  print(cmd)
  x = readMM(pipe(cmd))
  x = as(x, 'dgCMatrix')
  if (ncol(x)!=length(vocab)) {
    print(dim(x))
    print(length(vocab))
    stop("number of columns doesn't line up with vocab size")
  }
  colnames(x) = vocab  ## global!
  x
}

# "no-nonsense" TSV format

read_nns = function(filename, sep="\t",stringsAsFactors=F,quote='',comment='', ...)  
          read.table(filename, sep=sep,stringsAsFactors=stringsAsFactors,quote=quote,comment=comment, ...)
write_nns= function(x,file, sep="\t",quote=F,row.names=F,col.names=F,...)
          write.table(x,file, sep=sep,quote=quote,row.names=row.names,col.names=col.names,...)


###   load stuff   ###


load_stuff = function(prefix = "data_geo") {
  p = function(f) sprintf("%s/%s", prefix,f)
  v <<- read_nns(p("vocab"), col.names=c("vocab","count"))
  vocab <<- v$vocab
  user_info <<- read_nns(p("user_info"), col.names=c('username','lat','long'))

  all_x <<- load_features(p("user_word_tf"))

  stopifnot(nrow(user_info) == nrow(all_x))
  row.names(all_x) <<- user_info$username

  folds <<- rep(1:5, nrow(all_x))[1:nrow(all_x)]

  ###  Make the splits
  dev_lat <<- list(
    train = list( x= all_x[folds %in% 1:3,], y=user_info$lat[folds %in% 1:3]),
    test  = list( x= all_x[folds==4,], y=user_info$lat[folds==4])
  )

  dev_long<<- list(
    train = list( x= all_x[folds %in% 1:3,], y=user_info$long[folds %in% 1:3]),
    test  = list( x= all_x[folds==4,], y=user_info$long[folds==4])
  )  
}


load_small = function(prefix) {
  p = function(f) sprintf("%s/%s", prefix,f)
  v <<- read_nns(p("vocab_wc_dc"), col.names=c("vocab","word_count","doc_count"))
  vocab <<- v$vocab
  user_info <<- read_nns(p("user_info"), col.names=c('username','lat','long'))
}


fold_split = function(data, test_fold) {
  ## for simple cross-validation
  list(
    train = list(
      x = data$x[ data$folds != test_fold, ],
      y = data$y[ data$folds != test_fold ],
      inds = which(data$folds != test_fold)
    ),
    test = list(
      x = data$x[ data$folds == test_fold, ],
      y = data$y[ data$folds == test_fold ],
      inds = which(data$folds == test_fold)
    )
  )
}

active = function(b) b[b!=0]

# add_counts = function(data) {
#   data$word_df = apply(data$x,2,function(x)

predict_2d = function(dev_lat,dev_long, m_lat,m_long, lambda_lat, lambda_long) {
  pred_long=with(dev_long$test,predict(m_long,x,s=lambda_long))
  pred_lat=with(dev_lat$test,predict(m_lat,x,s=lambda_lat))
  data.frame(lat=as.vector(pred_lat), long=as.vector(pred_long))
}

eval_2d = function(pred, real_lat, real_long) {
  lat_dev = pred$lat - real_lat
  long_dev= pred$long - real_long
  
  mean(sqrt(lat_dev**2 + long_dev**2))
  # sq_dists = laply(1:nrow(user_info), function(i)
  #   { (pred[i,] - real_loc[i,])**2 },   .progress='text'
  # )
  # mean(sqrt(sq_dists))
}

eval_center_guess = function(sp_lat, sp_long) {
  lat_ctr = mean(sp_lat$train$y)
  long_ctr= mean(sp_long$train$y)
  
  lat_dev =  sp_lat$test$y - lat_ctr
  long_dev= sp_long$test$y - long_ctr
  mean(sqrt(lat_dev**2 + long_dev**2))
}

run_split = function(split, ...) {
  m = with(split$train, timeit(
    glmnet(x,y, family='gaussian', ...)
  ))
  pred = with(split$test, predict(m,x))
  mse = with(split$test, sapply(1:ncol(pred), function(i) 
    mean(  (pred[,i]-y)**2  )))
  rsq = 1 - mse / with(split$test, mean((y-mean(y))**2))
  # acc = with(split$test, sapply(1:ncol(pred), function(i) mean(  (pred[,i]>0) == y)))

  list( m = m, pred = pred, mse=mse, mse = mse)
}


eval_run = function(res) {
  z = data.frame(df=res$m$df,lambda=res$m$lambda,mse=res$mse)
  best = which.min(z$mse)
  print(z[best,])
  plot(z$df, z$mse)
  invisible(z)
}



run_all_splits = function(data, ...) {
  lapply(1:5, function(fold) {
    sp = fold_split(data, fold)
    list(
      r = run_split(sp, ...),
      sp= sp)
  })
}




show_best_words = function(x) {
  best = which.max(x$rsq)
  show_words(x$m$beta[,best])
}

show_words = function(x) {
  active = which(x != 0)
  x = x[active]
  x = sort(x)
  data.frame(x)
}

word_report = function(beta, train, test=NULL) {
  stopifnot(!is.null(train$df))
  active = which(beta != 0)
  weights = beta[active]
  z = data.frame(weights, train_df=train$df[active])
  if (!is.null(test)) {
    z$test_df = test$df[active]
  }
  z = z[order(abs(z$weights)),]
z
}

add_word_df = function(data) {
  data$df = timeit(apply(data$x, 2, function(col) sum(col>0)))
  data
}




write_models = function(multi_res, prefix) {
  for (f in 1:5) {
    m = multi_res[[f]]$r$m
    write_model(m$beta[,1], sprintf("%s.%d", prefix,f))
  }
}

write_model = function(beta, filename) {
  active = which(beta != 0)
  beta = beta[active]
  beta = sort(beta)
  d = data.frame(names(beta), beta)
  write.table(d,filename, row.names=F,col.names=F,sep="\t",quote=F)
}

