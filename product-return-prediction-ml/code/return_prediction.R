# =====================================================
# Machine Learning Model for Predicting Product Returns
# Author: Shaik Naved Ahmed
# Description: Random Forest model to predict product
#              returns in e-commerce using customer
#              usage and satisfaction data.
# =====================================================

# ===============================
# Install packages (run once)
# ===============================
install.packages(c("readxl","dplyr","caret","randomForest","pROC","ggplot2"))

# ===============================
# Load Libraries
# ===============================
library(readxl)
library(dplyr)
library(caret)
library(randomForest)
library(pROC)
library(ggplot2)

# ===============================
# 1. Load Dataset
# ===============================
data <- read_excel("Expanded_CustomerUsageData.xlsx")

# Inspect dataset
str(data)
summary(data)

# ===============================
# 2. Data Preprocessing
# ===============================

# Remove unnecessary ID columns
data <- data %>%
  select(-CustomerID, -ProductID)

# Convert categorical variables to factors
data$Category <- as.factor(data$Category)
data$Region <- as.factor(data$Region)
data$WarrantyStatus <- as.factor(data$WarrantyStatus)
data$ReturnFlag <- as.factor(data$ReturnFlag)

# Check missing values
colSums(is.na(data))

# ===============================
# 3. Exploratory Data Analysis
# ===============================

# Return rate by product category
ggplot(data, aes(x = Category, fill = ReturnFlag)) +
  geom_bar(position = "fill") +
  ylab("Return Rate") +
  ggtitle("Return Rate by Product Category")

# Customer satisfaction distribution
ggplot(data, aes(x = CustomerSatisfactionScore, fill = ReturnFlag)) +
  geom_histogram(bins = 10) +
  ggtitle("Customer Satisfaction Distribution")

# Purchase amount distribution
ggplot(data, aes(x = PurchaseAmount)) +
  geom_histogram(bins = 20, fill = "skyblue") +
  ggtitle("Purchase Amount Distribution")

# ===============================
# 4. Train-Test Split
# ===============================

set.seed(123)

trainIndex <- createDataPartition(data$ReturnFlag,
                                  p = 0.8,
                                  list = FALSE)

trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# ===============================
# 5. Train Random Forest Model
# ===============================

rf_model <- randomForest(ReturnFlag ~ .,
                         data = trainData,
                         ntree = 500,
                         importance = TRUE)

print(rf_model)

# ===============================
# 6. Predictions
# ===============================

predictions <- predict(rf_model, testData)

# Confusion Matrix
conf_matrix <- confusionMatrix(predictions, testData$ReturnFlag)
print(conf_matrix)

# ===============================
# 7. ROC Curve and AUC
# ===============================

prob <- predict(rf_model, testData, type = "prob")

roc_curve <- roc(testData$ReturnFlag, prob[,2])

# Plot ROC curve
plot(roc_curve,
     main = "Random Forest ROC Curve",
     col = "darkgreen",
     lwd = 3)

# Display AUC score
auc_value <- auc(roc_curve)
print(auc_value)

# Save ROC curve image
png("roc_curve.png", width = 800, height = 600)

plot(roc_curve,
     main = "Random Forest ROC Curve",
     col = "darkgreen",
     lwd = 3)

dev.off()

# ===============================
# 8. Feature Importance
# ===============================

importance(rf_model)

# Plot feature importance
png("feature_importance.png", width = 800, height = 600)

varImpPlot(rf_model)

dev.off()

# ===============================
# 9. Model Comparison
# Logistic Regression
# ===============================

log_model <- glm(ReturnFlag ~ .,
                 data = trainData,
                 family = binomial)

log_pred <- predict(log_model,
                    testData,
                    type = "response")

log_pred_class <- ifelse(log_pred > 0.5, 1, 0)

confusionMatrix(as.factor(log_pred_class),
                testData$ReturnFlag)

# ===============================
# End of Script
# ===============================