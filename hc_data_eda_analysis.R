
################################################################################
# Data Preparation
################################################################################

library(dplyr)
library(caret)
library(earth)
library(Formula)
library(plotmo)
library(plotrix)
library(TeachingDemos)
library(iml)
library(randomForest)
library(rpart)
library(rpart.plot)
library(ROCR)
library(dlookr)
library(ggplot2)
library(DataExplorer)

#library(readr)

### read Data


library(rstudioapi)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# hc_data <- read.csv("hc_data_v2.csv", header = F, na.strings=c("-999", "NA", ""))


hc_data <- read.csv("HNC_DATA_v4.csv", fileEncoding = 'euc-kr', header = T, na.strings=c("-999", "NA", ""))

#hc_data <- hc_data[,-1]

### select Useful Columns

hc <- hc_data[,c(1, 67, 68, 69, 14, 33,66, 24, 71, 25, 72, 70, 62, 36, 39 )]
names(hc) <- c("Order_Number", "Sex", "Age", "Age_Group", "Price", "Monthly_Install", "Delivery", "CB", "CB_Class", "SP","SP_Class", "Payment", "Previous_Contracts", "Periods","Overdue")


#hc <- hc_data[,c(2, 43, 44, 45, 47, 46, 12, 19, 29, 31, 33)]
#names(hc) <- c("Order_Number", "Delivery", "Payment", "Age", "Sex", "Age_bin","Price", "Credit_CB", "Periods", "Overdue", "Monthly_Install")

### shipment
hc$Delivery <- ifelse(hc$Delivery=="Shipment 인수후 설치", "Shipment", "Install")

### Payment
#hc$Payment <- ifelse(hc$Payment==0, "CMS", "Credit Card")

### Payment
hc$Sex <- ifelse(hc$Sex=="Man", "Male", "Female")


################################################################################
# EDA
################################################################################

#df = subset(hc, select = -c(4, 9, 11) )

#colnames(df) <- c("주문번호", "성별", "연령", "가격", "월납입액", "배송방법", "CB", "SP", "결재수단", "과거거래경험", "연체개월")

#create_report(df)

#dlookr::plot_outlier(hc, Overdue)

################################################################################
# Defining Target Variable 1 : Overdue >= 1, periods >= 3, CB & SP available 
################################################################################


### remove overdue = 1
#hc <- hc[hc$Overdue==0 | hc$Overdue>1,]

### add Target Variable
hc$default <- ifelse(hc$Overdue>0, "Yes", "No")

### add CB Variable
#hc$CB <- ifelse(hc$Credit_CB==1 | hc$Credit_CB==2 | hc$Credit_CB==3, "Class 1", "Class 2")

hc$Previous_Contracts_YN <- ifelse(hc$Previous_Contracts>0, "Yes", "No")

### Observed Periods is 6 months
hc <- hc[hc$Periods>2,]


# Remove NA in SP, CB

hc_cb_sp <- hc[!is.na(hc$SP) & !is.na(hc$CB),]

hc <- hc_cb_sp

write.csv(hc, "hc_eda_1.csv")

################################################################################
# Defining Target Variable 2 : Overdue >= 2, Periods >= 6 
################################################################################

### remove overdue = 1
hc <- hc[hc$Overdue==0 | hc$Overdue>1,]

### add Target Variable
hc$default <- ifelse(hc$Overdue>2, "Yes", "No")

### add CB Variable
#hc$CB <- ifelse(hc$Credit_CB==1 | hc$Credit_CB==2 | hc$Credit_CB==3, "Class 1", "Class 2")

hc$Previous_Contracts_YN <- ifelse(hc$Previous_Contracts>0, "Yes", "No")

### Observed Periods is 6 months
hc <- hc[hc$Periods>5,]


# Remove NA in SP, CB

hc_cb_sp <- hc[!is.na(hc$SP) & !is.na(hc$CB),]

hc <- hc_cb_sp

write.csv(hc, "hc_eda_2.csv")




################################################################################
# transforming variables
################################################################################

names <- c("Delivery", "Payment", "Sex", "default","CB_Class", "SP_Class", "Previous_Contracts_YN")
hc[,names] <- lapply(hc[,names] , factor)

names <- c("Age", "Price", "Monthly_Install", "CB", "SP", "Previous_Contracts", "Periods")
hc[,names] <- lapply(hc[,names] , as.numeric)


################################################################################
# Sampling
################################################################################

# sum(hc$default=="Yes")
# dim(hc)


set.seed(2020)

modeling_set_1 <- hc[hc$default=="Yes",]
modeling_set_2 <- sample_n(hc[hc$default=="No",], 350)

modeling_set <- rbind(modeling_set_1, modeling_set_2)  

#order_case <- cbind.data.frame(as.numeric(rownames(modeling_set)), modeling_set$Order_Number)  
#names(order_case) <- c("id","Order_Number")


################################################################################
# Model Version
################################################################################

modeling_set_v1 <- modeling_set[, -c(1, 4, 5, 7, 9, 11, 14, 15, 17)]


write.csv(hc, "hc_v1.csv")

################################################################################
# Decision Tree
################################################################################


library(rpart)
library(rpart.plot)

# training decision tree
tree <- rpart(default~., data=modeling_set_v1, cp=.02, method = 'class')

# decision tree plot
rpart.plot(tree, box.palette="RdBu", shadow.col="gray", nn=TRUE)




################################################################################
# Ensemble : Random Forest
################################################################################

# First, we split the data into training and testing sets
intrain <- createDataPartition(modeling_set_v1$default,p=0.7,list=FALSE)
set.seed(2020)
training <- modeling_set_v1[intrain,]
testing <- modeling_set_v1[-intrain,]


#training <- training %>% dplyr::select(-Order_Number, -Delivery, -Age_bin, -Price, -Credit_CB, -Periods, -Overdue)
#testing <- testing %>% dplyr::select(-Order_Number, -Delivery, -Age_bin, -Price, -Credit_CB, -Periods, -Overdue)
testing.X <- testing %>% dplyr::select(-default)

### training by random forest with caret

fitControl <- trainControl(
  method = 'cv',                   # k-fold cross validation
  number = 5,                      # number of folds
  savePredictions = 'final',       # saves predictions for optimal tuning parameter
  classProbs = T,                  # should class probabilities be returned
  summaryFunction=twoClassSummary  # results summary function
) 



rf <- caret::train(default ~ ., data = training, method = 'rf', trControl = fitControl, metric = 'ROC')

### prediction 
rf.probs <- predict(rf, testing, type="prob", class = "Yes")
pred_dt <- cbind(testing, rf.probs)

### rf variable importance 
varImp(rf)

### ROC Curve
library(ROCR)
predictions <- as.vector(rf.probs[,2])
pred <- ROCR::prediction(predictions, testing$default)

perf_AUC <- performance(pred,"auc") 
AUC <- perf_AUC@y.values[[1]]

perf_ROC <- performance(pred,"tpr","fpr") 

plot(perf_ROC, main="ROC plot", colorize=TRUE)
text(0.5,0.5,paste("AUC = ",format(AUC, digits=5, scientific=FALSE)))

### confusion matrix with adjusting threshold
rf.pred <- predict(rf, testing,  type = "prob")

plot(density(rf.pred$Yes))

threshold <- 0.5
pred <- factor(ifelse(rf.pred[,"Yes"] > threshold, "Yes", "No") )
caret::confusionMatrix(pred, testing$default)


################################################################################
# IML : Interpretable Machine Learning
################################################################################

### Feature importance

varImp(rf)

predictor_1 <- Predictor$new(rf, data = testing.X, y = testing$default, type = "prob")
imp <- FeatureImp$new(predictor_1, loss = "ce")
plot(imp)



### Partial Dependence & Individual Conditional Expectation plots (ICE)

predictor_2 = iml::Predictor$new(rf, data = testing.X, y = testing$default, class = "Yes", type = "prob")


### ALE
ale_monthly <- FeatureEffect$new(predictor_2, feature = "Monthly_Install")
plot(ale_monthly)

ale_age <- FeatureEffect$new(predictor_2, feature = "Age")
plot(ale_age)

ale_payment <- FeatureEffect$new(predictor_2, feature = "Payment")
plot(ale_payment)

### interaction

interact <- Interaction$new(predictor_2)
plot(interact)

interact_month <- Interaction$new(predictor_2, feature = "Monthly_Install")
plot(interact_month)


### Partial Dependence & Individual Conditional Expectation plots (ICE)

hist(hc$Monthly_Install)

predictor_2 <- iml::Predictor$new(rf, data = testing.X, y = testing$default, class = "Yes", type = "prob")

ice_monthly <- iml::FeatureEffect$new(predictor_2,  method = "ice", feature = "Monthly_Install")
ice_age <- iml::FeatureEffect$new(predictor_2,  method = "ice", feature = "Age")
ice_payment <- iml::FeatureEffect$new(predictor_2,  method = "ice", feature = "Payment")

plot(ice_monthly)
plot(ice_age)
plot(ice_payment)

pdp_monthly <- iml::FeatureEffect$new(predictor_2,  method = "pdp", feature = "Monthly_Install")
pdp_age <- iml::FeatureEffect$new(predictor_2,  method = "pdp", feature = "Age")
pdp_payment <- iml::FeatureEffect$new(predictor_2,  method = "pdp", feature = "Payment")

plot(pdp_monthly)
plot(pdp_age)
plot(pdp_payment)


pdp_ice_monthly <- iml::FeatureEffect$new(predictor_2,  method = "pdp+ice", feature = "Monthly_Install")
pdp_ice_age <- iml::FeatureEffect$new(predictor_2,  method = "pdp+ice", feature = "Age")
pdp_ice_payment <- iml::FeatureEffect$new(predictor_2,  method = "pdp+ice", feature = "Payment")

plot(pdp_ice_monthly)
plot(pdp_ice_age)
plot(pdp_ice_payment)

pdp_monthly <- pdp_monthly$results[order(pdp_monthly$results[,1]),]

### tree surrogate : decision tree model

tree <- TreeSurrogate$new(predictor_2, maxdepth = 2)
plot(tree)

# prediction with decision tree

tree$predict(testing.X)

tree_predict <- cbind(testing.X$Order_Number, tree$predict(testing.X))
names(tree_predict) <- c("Order Number","Prob")





################################################################################
# Lime : Local interpretable model-agnostic explanations
################################################################################

### lime ###

library(lime)

explainer_caret <- lime(testing.X, rf, n_bins = 5)


explanation_caret <- lime::explain(
  x = testing.X, 
  explainer = explainer_caret, 
  n_permutations = 5000,
  dist_fun = "gower",
  kernel_width = .75,
  n_features = 10, 
  feature_select = "highest_weights",
  labels = "Yes"
)



tibble::glimpse(explanation_caret)

pred_table <- cbind.data.frame(testing,pred)

plot_features(explanation_caret[explanation_caret$case==138,])

### For each Order_Number, lime explanation

# case = 19
# order_case[order_case$id==19,]
# order_case[order_case$id==19,]
#      id Order_Number
#  13  19       523521

order_num <- 523521

case_num <- order_case[order_case$Order_Number==order_num,]$id

plot_features(explanation_caret[explanation_caret$case==case_num,])



explanation_caret[explanation_caret$label_prob>0.4,]

plot_features(explanation_caret[explanation_caret$case==139,])


### plot all lime explanations

plot_explanations(explanation_caret)






################################################################################
# EDA
################################################################################

library(dplyr)
library(dlookr)
library(ggplot2)
library(DataExplorer)

hc_1 <- hc %>% dplyr::select(-Order_Number, -Status, -Zip)

plot_str(hc_1)
plot_histogram(hc_1)
#plot_density(hc_1)
plot_bar(hc_1)

DataExplorer::create_report(hc_1, y = "default")

hc_tbl <- table(hc$default, hc$Credit_CB)
barplot(hc_tbl, xlab='Credit Rating',ylab='Frequency',main="Default by Credit",
        col=c("darkblue","lightcyan")
        ,legend=rownames(hc_tbl), args.legend = list(x = "topleft"))

hc %>% filter(Credit_CB == "7등급") %>%  filter(default == "Yes")


# distribution of CB, SP

attach(hc)
table(default, Credit_CB)
table(default, Credit_SP)
table(Credit_CB, Credit_SP)


# binning 

library(smbinning)
hc_2 <- hc
hc_2$target <- ifelse(hc_2$default=="Yes", 1,0)

result = smbinning(hc_2,y="target",x="Monthly_Install",p=0.05) 
result$ivtable

result = smbinning(hc_2,y="target",x="Credit_CB",p=0.05) 
result$ivtable

result = smbinning(hc_2,y="target",x="Credit_SP",p=0.01) 
result$ivtable


