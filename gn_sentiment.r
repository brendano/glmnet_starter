# some routines for doing text classification with glmnet logistic regression.
# brendan o'connor, april 2010 or so, brenocon@gmail.com / http://anyall.org
# was used for pang&lee movie review sentiment classification

# HOW TO RUN
# source("gn_sentiment.r")
# sp = fold_split(data,1)
# res = run_split(sp)
# eval_run(res)


library(glmnet)
library(Matrix)
source("util.r")  # http://github.com/brendano/dlanalysis/raw/master/util.R


## all_x=readMM("all.mm")  ##basic version

## Initial load from MatrixMarket triples format
pp = "data_sentiment"
p= function(f) sprintf("%s/%s", pp, f)
cmd = sprintf("cat %s/all_x.mm && (cat %s/all_x | awk '{print $1,$2,log(1+$3)}')",pp,pp)
all_x=readMM(pipe(cmd))
all_x=as(all_x,'dgCMatrix')
vocab = readLines(p("vocab.txt"))
colnames(all_x) = vocab


## Wrap it up with cross-validation folds
data = list()
data$x = all_x[c(1:500, 1001:1500),]
data$y=c(rep(1,500),rep(0,500))
data$doc_id = c(readLines(p("pos_doc_ids"))[1:500], readLines(p("neg_doc_ids"))[1:500])
set.seed(42)
data$folds = shuffle( rep(1:5, nrow(data$x))[1:nrow(data$x)] )


fold_split = function(data, test_fold) {
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


run_split = function(split, ...) {
  m = with(split$train, timeit(
    glmnet(x,y, family='binomial', ...)
  ))
  pred = with(split$test, predict(m,x))
  acc = with(split$test, sapply(1:ncol(pred), function(i) mean(  (pred[,i]>0) == y)))

  list( m = m, pred = pred, acc = acc)
}


eval_run = function(res) {
  z = data.frame(df=res$m$df,lambda=res$m$lambda,acc=res$acc)
  best = which.max(z$acc)
  print(z[best,])
  plot(z$df, z$acc)
}

run_all_splits = function(data, ...) {
  lapply(1:5, function(fold) {
    sp = fold_split(data, fold)
    list(
      r = run_split(sp, ...),
      sp= sp)
  })
}

# uses fold's trained model against its test fold

multi_predict = function(multi_res, lambda, ...) {
  inds = sapply(1:5, function(i) multi_res[[i]]$sp$test$inds)
  multi_pred = matrix(nrow=length(inds), ncol=length(lambda))
  for (fold in 1:5) {
    m = multi_res[[fold]]$r$m
    sp = multi_res[[fold]]$sp
    p = predict(m, sp$test$x, lambda, ...)
    
    # print(p)
    # print(sp$test$inds)
    multi_pred[sp$test$inds,] = p
  }
  multi_pred
}

eval_multi_run = function(multi_res, data, xlim=NULL,ylim=NULL) {
  lambdas = unlist(sapply(1:5, function(i) multi_res[[i]]$r$m$lambda))
  accs = unlist(sapply(1:5, function(i) multi_res[[i]]$r$acc))

  if (is.null(xlim)) xlim=range(lambdas)
  if (is.null(ylim)) ylim=range(accs)
  plot.new(); plot.window(xlim=xlim,ylim=ylim)
  box(); axis(1); axis(2); axis(3)
  for (i in 1:5) {
    z = multi_res[[i]]$r
    points(z$m$lambda, z$acc, col=rainbow(5)[i])
  }
  
  cross_lambda = seq(min(lambdas),max(lambdas),length.out=100)
  cross_pred = multi_predict(multi_res, cross_lambda)
  cross_acc = sapply(1:ncol(cross_pred), function(i) mean(  (cross_pred[,i]>0) == (data$y)))
  
  points(cross_lambda, cross_acc, type='o')
  
  fold_bests = dfapply(1:5, function(i) with(multi_res[[i]]$r, {
    best = which.max(acc)
    list(lambda=m$lambda[best], acc=acc[best], df=m$df[best])
  }))
  
  cat("Multi best settings\n")
  print(fold_bests)
  
  cat("Best lambda\n")
  print(cross_lambda[which.max(cross_acc)])
  cat("Best acc\n")
  print(max(cross_acc))
}

show_best_words = function(x) {
  best = which.max(x$acc)
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

