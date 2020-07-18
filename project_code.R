# Course Project code


# source: http://groupware.les.inf.puc-rio.br/har

# Six young health participants were asked to perform one set of 10 repetitions of the
# Unilateral Dumbbell Biceps Curl in five different fashions:
#     exactly according to the specification (Class A),
#     throwing the elbows to the front (Class B),
#     lifting the dumbbell only halfway (Class C),
#     lowering the dumbbell only halfway (Class D)
#     and throwing the hips to the front (Class E).

# Aim: build a model for predicting how well the exercise is being done
# (variable: "classe") in each measurement.


# Loading data

setwd("~/Desktop/Data Science course/8. Practical Machine Learning/Project")

if (!file.exists("training.csv")){
    fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    download.file(fileUrl, destfile="training.csv", method="curl")
}

if (!file.exists("testing.csv")){
    fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
    download.file(fileUrl, destfile="testing.csv", method="curl")
}

trainingdat <- read.csv("training.csv")
quizdat <- read.csv("testing.csv")

dim(trainingdat)    # 19622 x 160
dim(quizdat)        # 20 x 160

names(quizdat)[!names(quizdat) %in% names(trainingdat)]

# Both datasets have 160 columns and they all have same names except:
    # trainingdat has a column "classe"
    # quiztestdat has a column "problem_id" 

# IMPORTANT: this test set does not contain a "classe" column.
# It is merely relevant for the course project quiz I'll take
# after I build my predictor model.



# libraries

library(caret)
library(nnet)   # for multiclass log reg
library(randomForest)
library(e1071)  # for svm models
library(gbm)    # for boosting



# splitting data into training/test/validation

set.seed(3250234)
inTrain <- createDataPartition(trainingdat$classe, p=.7, list=FALSE)
validation <- trainingdat[-inTrain,]
trainingdat <- trainingdat[inTrain,]

inTrain <- createDataPartition(trainingdat$classe, p=.7, list=FALSE)
testing <- trainingdat[-inTrain,]
training <- trainingdat[inTrain,]

dim(training)   # 9619 x 160
dim(testing)    # 4118 x 160
dim(validation) # 5885 x 160

summary(training)
str(training)



# PREPROCESSING

str(training)
# Looking a bit more closely we can see that
    # some factor variables have MANY levels, e.g. kurtosis_roll_belt has 397 levels.
    # some have strange levels, e.g. kurtosis_yaw_belt has 2 levels: "" or "#DIV/0!"
    # some variables have MANY NA values


names(training)[colSums(is.na(training)) > 0]
# 67 columns have NA values

names(training)[colSums(is.na(training)) == dim(training)[1]]
# no columns has ONLY NA values

table( colSums(is.na(training)) )
# 93 variables have no NA values
# 67 variables have 9407 NA values
    # which is a LOT considering there are 9619 observations
    # about 9407/9619*100 = 97.8% missing values

names(training)[colSums(is.na(training)) > 0]

# Upon investigating, these all seem to be statistic summaries of other variables,
# namely: max, min, amplitude, var, avg, stddev.
# Therefore, I decided to drop these.

badvars <- names(training)[colSums(is.na(training)) > 0]
length( grep("^(max|min|amplitude|var|avg|stddev)_", badvars) )
# result: 67, meaning that all these features start with
# max_, min_, amplitude_, var_, avg_, stddev_

# But other variables of the data set also begin like this.
# Let's investigate them!
vars <- names(training)[colSums(is.na(training)) == 0]

str(training[,vars])

# Close inspection once again proves that all variables with problems mentioned above
# are precisely the ones that start with
# max_, min_, amplitude_, var_, avg_, stddev_, kurtosis_, skewness_

# Note:
# kurtosis is a measure of variability in a distribution, more specifically
# the combined weight of a distribution's tails relative to the center of the distribution.


# Additionally the first 7 features are:
    # X: unique sequential number
    # user_name: participant name
    # raw_timestamp_part_1: first part of timestamp when observation was collected
    # raw_timestamp_part_2: second part of timestamp when observation was collected
    # cvtd_timestamp: timestamp values converted to mm/dd/yyyy hh:mm format
    # new_window: (yes/no) row represents a new time window for sliding window feature extraction
    # num_window: numeric identifier for feature extraction window

# X is completely irrelevant for modelling class

table(training$new_window)
# no  yes 
# 9419  200 
# so 98% is "no" and 2% is "yes" (highly skewed variable)
# so we don't use it


# bad features
ind <- grep("^(max|min|amplitude|var|avg|stddev|kurtosis|skewness)_", names(training))
str(training[,ind])

ind <- c(1, 6, ind)



str(training[,ind])
# 88 variables selected

training <- training[, -ind]
dim(training)   # 9619 x 60
# 160 - 100 = 60 variables left!


str(training)
# seems OK!


# Preprocessing testing and validation data too
testing <- testing[, -ind]
validation <- validation[, -ind]





# PCA
preproc <- preProcess(training, method="pca", pcaComp=3)
trainPC <- predict(preproc, training)



# Prediction models

# fit multiclass logistic regression model (actually via neural networks)
# https://www.rdocumentation.org/packages/nnet/versions/7.3-14/topics/multinom
logreg.fit <- multinom(classe~., data=training)

# fit random forest model
rf.fit <- randomForest(classe~., data=training)

# fit a support vector machine model
svm.fit <- svm(classe~., data=training)


# (too slow)
# fit a boosting model
# boost.fit <- train(classe~., data=training, method="gbm", verbose=FALSE)

# (the following models gave errors)
# fit a linear discriminant analysis model
# lindisc.fit <- train(classe~., data=training, method="lda")

# fit a Naive Bayes model
# naive.fit <- train(classe~., data=training, method="nb")


# Predicting on testing data set
logreg.pred <- predict(logreg.fit, testing)
rf.pred <- predict(rf.fit, testing)
svm.pred <- predict(svm.fit, testing)

confusionMatrix(logreg.pred, testing$classe)
confusionMatrix(rf.pred, testing$classe)
confusionMatrix(svm.pred, testing$classe)

# combined predictions in a new dataframe
comb.data <- data.frame(logreg = logreg.pred,
                       rf = rf.pred,
                       svm = svm.pred,
                       classe = testing$classe)

# combined model: random forest
comb.fit <- train(classe~., data=comb.data, method="rf")

# prediction of resultant model on testing set
comb.pred <- predict(comb.fit, comb.data)

# accuracy on testing set
confusionMatrix(comb.pred, testing$classe)



# out-of-sample error
# error on the validation set
logreg.val <- predict(logreg.fit, validation)
rf.val <- predict(rf.fit, validation)
svm.val <- predict(svm.fit, validation)

comb.val <- data.frame(logreg = logreg.val,
                       rf = rf.val,
                       svm = svm.val)

comb.pred <- predict(comb.fit, comb.val)
confusionMatrix(comb.pred, validation$classe)

# out-of-sample accuracy: .9975
# out-of-sample error: .0025



# quiz answers

names(quizdat)
quizdat <- quizdat[,-c(ind)]
    # also removing col 160 (problem_id)
str(quizdat)

# some variables in quizdat have different class than
# the corresponding one in training/test/validation sets
# so we need to fix that

# check where classes differ
# which( sapply(quizdat, class) != sapply(training, class) )
# result:
# magnet_dumbbell_z  magnet_forearm_y  magnet_forearm_z classe 
# 46                58                59                60

# sapply(quizdat[,c(46,58,59)], class)
# sapply(training[,c(46,58,59)], class)

# quizdat[,46] <- as.numeric(quizdat[,46])
# quizdat[,58] <- as.numeric(quizdat[,58])
# quizdat[,59] <- as.numeric(quizdat[,59])

# Most importantly, factor variables have different levels
# in training and quiz datasets.

levels(quizdat$cvtd_timestamp) <- levels(training$cvtd_timestamp)
levels(quizdat$new_window) <- levels(training$new_window)



logreg.quiz <- predict(logreg.fit, quizdat)
rf.quiz <- predict(rf.fit, quizdat)
svm.quiz <- predict(svm.fit, quizdat)

logreg.quiz
rf.quiz
svm.quiz

comb.quiz <- data.frame(logreg = logreg.quiz,
                       rf = rf.quiz,
                       svm = svm.quiz)




predict(comb.fit, comb.quiz)

# result: B A B A A E D B A A B C B A E E A B B B


