# clear all workspace
rm(list = ls())

library(mlr3)
library(tidyverse)
library(ggplot2)
library(mlr3learners)
library(data.table)
library(mlr3viz)
library(mlr3tuning)
library(mlr3pipelines)
library(paradox)
library(skimr)
library(smotefamily)
library(gridExtra)

setwd("C:/Users/user/Documents/R-projects/i2ml_final_project")
setwd("/home/alex/Desktop/i2ml_final_project/")

# suppress package making warning by start up in train 
# Warning: "package ??kknn?? was built under R version 3.6.3"
suppressPackageStartupMessages(library(kknn))

# read data with different encoding
dl_iv_data <- read.csv2("credit_card_prediction/iv_data/dl_iv_data.csv") %>% mutate(y = as.factor(y)) %>% mutate_if(is.integer,as.numeric)
mf_iv_data <- read.csv2("credit_card_prediction/iv_data/mf_iv_data.csv") %>% mutate(y = as.factor(y)) %>% mutate_if(is.integer,as.numeric)
mice_iv_data <- read.csv2("credit_card_prediction/iv_data/mice_iv_data.csv") %>% mutate(y = as.factor(y)) %>% mutate_if(is.integer,as.numeric)

dl_oh_data <- read.csv("credit_card_prediction/oh_data/dl_oh_data.csv") %>% mutate(y = as.factor(y)) %>% mutate_if(is.integer,as.numeric)
mf_oh_data <- read.csv("credit_card_prediction/oh_data/mf_oh_data.csv") %>% mutate(y = as.factor(y)) %>% mutate_if(is.integer,as.numeric)
mice_oh_data <- read.csv("credit_card_prediction/oh_data/mice_oh_data.csv") %>% mutate(y = as.factor(y)) %>% mutate_if(is.integer,as.numeric)



# load data directly into tasks for further training
tasks <- list(
  TaskClassif$new("dl_iv", backend = dl_iv_data, target = "y"),
  TaskClassif$new("mf_iv", backend = mf_iv_data, target = "y"),
  TaskClassif$new("mice_iv", backend = mice_iv_data, target = "y"),
  TaskClassif$new("dl_oh", backend = dl_oh_data, target = "y"),
  TaskClassif$new("mf_oh", backend = mf_oh_data, target = "y"),
  TaskClassif$new("mice_oh", backend = mice_oh_data, target = "y")
)

rm(dl_iv_data, mf_iv_data, mice_iv_data, dl_oh_data, mf_oh_data, mice_oh_data)

# knn learner
knn_learner <- lrn("classif.kknn", predict_type = "prob")

# setting the tunning for parameters, and terminator
knn_param_set <- ParamSet$new(params = list(ParamInt$new("k", lower = 3, upper = 5)))
terms <- trm("evals", n_evals=1)
#measure <- msr("classif.auc")

# creat autotuner, using the inner sampling and tuning parameter with random search
inner_rsmp <- rsmp("cv", folds = 2L)
knn_auto <- AutoTuner$new(learner = knn_learner, resampling = inner_rsmp, 
                          #measure = measure, 
                          search_space = knn_param_set,
                          terminator = terms, tuner = tnr("grid_search"))

# outer_rsmp <- rsmp("cv", folds = 3L)
outer_rsmp <- rsmp("holdout")
design = benchmark_grid(
  tasks = tasks,
  learners = knn_learner,
  resamplings = outer_rsmp
)

# set seed before traing, then run the benchmark
# save the results afterwards
set.seed(2020)
knn_bmr <- benchmark(design, store_models = TRUE)

# autoplot auc for all tasks (merged in one plot)
multiplot_roc <- function(models, type="roc"){
  plots <- list()
  thm <- theme(axis.text.x = element_blank(), axis.text.y = element_blank())
  model <- models$clone(deep = TRUE)$filter(task_id = "dl_iv")
  auc <- round(model$aggregate(msr("classif.auc"))[[7]], 4)
  plots[[1]] <- autoplot(model, type = type) + ggtitle(paste("dl_iv:", auc)) + thm
  
  model <- models$clone(deep = TRUE)$filter(task_id = "mf_iv")
  auc <- round(model$aggregate(msr("classif.auc"))[[7]], 4)
  plots[[2]] <- autoplot(model, type = type) + ggtitle(paste("mf_iv:", auc)) + thm
  
  model <- models$clone(deep = TRUE)$filter(task_id = "mice_iv")
  auc <- round(model$aggregate(msr("classif.auc"))[[7]], 4)
  plots[[3]] <- autoplot(model, type = type) + ggtitle(paste("mice_iv:", auc)) + thm
  
  model <- models$clone(deep = TRUE)$filter(task_id = "dl_oh")
  auc <- round(model$aggregate(msr("classif.auc"))[[7]], 4)
  plots[[4]] <- autoplot(model, type = type) + ggtitle(paste("dl_oh:", auc)) + thm
  
  model <- models$clone(deep = TRUE)$filter(task_id = "mf_oh")
  auc <- round(model$aggregate(msr("classif.auc"))[[7]], 4)
  plots[[5]] <- autoplot(model, type = type) + ggtitle(paste("mf_oh:", auc)) + thm
  
  model <- models$clone(deep = TRUE)$filter(task_id = "mice_oh")
  auc <- round(model$aggregate(msr("classif.auc"))[[7]], 4)
  plots[[6]] <- autoplot(model, type = type) + ggtitle(paste("mice_oh", auc)) + thm
  do.call("grid.arrange", plots)
}

multiplot_roc(knn_bmr)

# -----------------

learner = lrn("classif.kknn", predict_type = "prob")
pred = learner$train(tasks[[1]])$predict(tasks[[1]])
ggplot2::autoplot(pred, type = "roc")

# -----------------

task = tasks[[1]]
learner = lrn("classif.kknn", predict_type = "prob")
resampling = rsmp("holdout")
a = msr("classif.auc")
param_set = paradox::ParamSet$new(
  params = list(paradox::ParamInt$new("k", lower = 3, upper = 5)))
terminator = trm("evals", n_evals = 5)
tuner = tnr("grid_search", resolution = 10)

at = AutoTuner$new(learner, resampling, measure = measure,
                   param_set, terminator, tuner = tuner)


resampling_outer = rsmp("cv", folds = 3)
rr = resample(task = task, learner = at, resampling = resampling_outer)
autoplot(rr, type = "roc")


# -------------------- 

# get some example tasks
#tasks = lapply(c("german_credit", "sonar"), tsk)

tasks_t = list(tasks[[1]], tasks[[2]])

#learners = c("classif.kknn")
#learners = lapply(learners, lrn,
#                  predict_type = "prob", predict_sets = c("train", "test"))

learner = lrn("classif.kknn", predict_type = "prob", predict_sets = c("train", "test"))

# compare via 3-fold cross validation
resamplings = rsmp("cv", folds = 2)

# create a BenchmarkDesign object
design = benchmark_grid(tasks_t, learner, resamplings)
print(design)

bmr = benchmark(design)

measures = list(
  msr("classif.auc", id = "auc_train", predict_sets = "train"),
  msr("classif.auc", id = "auc_test")
)
bmr$aggregate(measures)

autoplot(bmr$clone(deep = TRUE)$filter(task_id = "dl_iv"), type = "roc")
