library("rpart")
library("RWeka")
library("partykit")

training <- read.csv("kddcup.data_10_percent_corrected", header=F)

corrected <- read.csv("corrected", header=F)

testdata = corrected[,1:41]

weka_label_result = training[,42]
weka_training_data = training[,1:41]

weka_tree_model = J48(as.factor(weka_label_result)~.,weka_training_data)
e = predict(weka_tree_model, newdata = testdata)

golden_answer = corrected[,42]
tree_predict = e
golden_answer = factor(golden_answer, levels =levels(tree_predict))
mean(golden_answer == tree_predict,na.rm = TRUE)
summary(tree_predict)
summary(golden_answer)