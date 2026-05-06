# ============================================================
# DS7003 PROJECT
# Random Forest vs Artificial Neural Network
# Myocardial Infarction Death Prediction
# ============================================================

# -----------------------------
# 1. LIBRARIES
# -----------------------------

library(randomForest)
library(caret)
library(nnet)
library(pROC)
library(ggplot2)
library(reshape2)

set.seed(123)

# -----------------------------
# 2. LOAD DATA
# -----------------------------

setwd(dirname(file.choose()))
df <- read.csv("MI.csv", header = FALSE, na.strings = "?")

cat("Raw dataset dimensions:", nrow(df), "rows x", ncol(df), "columns\n")


# -----------------------------
# 3. DEFINE FEATURES AND TARGET
# -----------------------------
# Columns 2 to 112 = clinical features/independent variable
# Column 124 = LET_IS, lethal outcome cause/ target/dependent variable

X <- df[, 2:112]
y_raw <- df[, 124]

# Give simple names to predictors because the CSV has no column headers.
colnames(X) <- paste0("Feature_", 1:ncol(X))
colnames(X)[1] <- "Age"


# Convert predictors to numeric.
X[] <- lapply(X, function(col) as.numeric(as.character(col)))

# Remove rows where target is missing, if any.
complete_target <- !is.na(y_raw)
X <- X[complete_target, ]
y_raw <- y_raw[complete_target]


# Making target variable as Binary target:
# Survived = 0
# Death, LET_IS > 0
y <- ifelse(y_raw == 0, "Survived", "Death")

# Complication is first, so caret treats Death as the positive/event class.
y <- factor(y, levels = c("Death", "Survived"))

cat("\nTarget distribution:\n")
print(table(y))
print(round(prop.table(table(y)) * 100, 2))


# -----------------------------
# 3. EXPLORATORY DATA ANALYSIS
# -----------------------------

# 3.1 Target class distribution
class_df <- as.data.frame(table(y))
colnames(class_df) <- c("Outcome", "Count")
class_df$Percent <- round(class_df$Count / sum(class_df$Count) * 100, 1)

p1 <- ggplot(class_df, aes(x = Outcome, y = Count, fill = Outcome)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_text(aes(label = paste0(Count, " (", Percent, "%)")), vjust = -0.3) +
  labs(title = "Target Class Distribution",
       x = "Outcome",
       y = "Number of Patients") +
  theme_minimal() +
  theme(legend.position = "none")

print(p1)

# 3.2 Age distribution by outcome

age_df <- data.frame(Age = X$Age, Outcome = y)

p2 <- ggplot(age_df, aes(x = Age, fill = Outcome)) +
  geom_histogram(bins = 30, colour = "white") +
  facet_wrap(~ Outcome, ncol = 1) +
  scale_x_continuous(breaks = seq(0, max(age_df$Age, na.rm = TRUE), by = 10)) +
  labs(title = "Age Distribution by Outcome",
       x = "Age",
       y = "Number of Patients") +
  theme_minimal() +
  theme(legend.position = "none")

print(p2)

# 3.3 Missing data plot
missing_df <- data.frame(
  Feature = names(X),
  MissingPercent = colMeans(is.na(X)) * 100
)

missing_df <- missing_df[order(-missing_df$MissingPercent), ]
top_missing <- head(missing_df, 20)

p3 <- ggplot(top_missing, aes(x = reorder(Feature, MissingPercent), y = MissingPercent)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Top 20 Features by Missing Percentage",
       x = "Feature",
       y = "Missing (%)") +
  theme_minimal()

print(p3)

# 3.4 Correlation 

# remove constant columns for plotting only.
X_for_corr <- X[, sapply(X, function(col) sd(col, na.rm = TRUE) > 0), drop = FALSE]

# choose the 15 most variable features.
feature_variance <- sapply(X_for_corr, var, na.rm = TRUE)
top_15_features <- names(sort(feature_variance, decreasing = TRUE))[1:15]
corr_data <- X_for_corr[, top_15_features, drop = FALSE]

# temporary median imputation only for this plot.
for (col in names(corr_data)) {
  med <- median(corr_data[[col]], na.rm = TRUE)
  if (is.na(med)) med <- 0
  corr_data[[col]][is.na(corr_data[[col]])] <- med
}

# create heatmap.
corr_matrix <- cor(corr_data)
corr_df <- reshape2::melt(corr_matrix)

p4 <- ggplot(corr_df, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white",
                       midpoint = 0, limits = c(-1, 1)) +
  labs(title = "Correlation Heatmap of 15 Selected Features",
       subtitle = "Median imputation used only for EDA visualisation",
       x = "", y = "", fill = "Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p4)


# -----------------------------
# 4. TRAIN-TEST SPLIT
# -----------------------------

train_index <- createDataPartition(y, p = 0.75, list = FALSE)

X_train <- X[train_index, , drop = FALSE]
X_test  <- X[-train_index, , drop = FALSE]
y_train <- y[train_index]
y_test  <- y[-train_index]

cat("\nTraining class distribution:\n")
print(table(y_train))

cat("\nTest class distribution:\n")
print(table(y_test))


# -----------------------------
# 5. PREPROCESSING 
# -----------------------------

# 5.1 Remove features with more than 60% missing values.
missing_percent_train <- colMeans(is.na(X_train)) * 100

removed_cols <- names(missing_percent_train[missing_percent_train > 60])
keep_cols <- names(missing_percent_train[missing_percent_train <= 60])

cat("\nRemoved columns with >60% missing values:\n")
print(removed_cols)

X_train <- X_train[, keep_cols, drop = FALSE]
X_test  <- X_test[, keep_cols, drop = FALSE]

cat("\nNumber of predictors after removing high-missing columns:\n")
print(ncol(X_train))



# 5.2 Identify variable types

# Count unique non-missing values in each column
unique_counts <- sapply(X_train, function(x) length(unique(na.omit(x))))

# Categorical columns
categorical_cols <- names(unique_counts[unique_counts <= 10])

# Continuous numeric columns
continuous_cols <- names(unique_counts[unique_counts > 10])

cat("Categorical columns:\n")
print(categorical_cols)

cat("Continuous numeric columns:\n")
print(continuous_cols)



# 5.3 Mode imputation for categorical variables

mode_value <- function(x) {
  x_no_na <- na.omit(x)
  ux <- unique(x_no_na)
  ux[which.max(tabulate(match(x_no_na, ux)))]
}

X_train_imp <- X_train
X_test_imp  <- X_test

for (col in categorical_cols) {
  mode_col <- mode_value(X_train[[col]])
  
  X_train_imp[[col]][is.na(X_train_imp[[col]])] <- mode_col
  X_test_imp[[col]][is.na(X_test_imp[[col]])]   <- mode_col
}


# 5.4 Median imputation for continuous variables

for (col in continuous_cols) {
  med_col <- median(X_train[[col]], na.rm = TRUE)
  
  X_train_imp[[col]][is.na(X_train_imp[[col]])] <- med_col
  X_test_imp[[col]][is.na(X_test_imp[[col]])]   <- med_col
}



# 5.5 Feature scaling
# Scaling is needed for ANN and PCA

scale_obj <- preProcess(X_train_imp, method = c("center", "scale"))

X_train_scaled <- predict(scale_obj, X_train_imp)
X_test_scaled  <- predict(scale_obj, X_test_imp)



# 5.6 Feature selection: near-zero variance
nzv_cols <- nearZeroVar(X_train_scaled)

if (length(nzv_cols) > 0) {
  X_train_final <- X_train_scaled[, -nzv_cols, drop = FALSE]
  X_test_final  <- X_test_scaled[, -nzv_cols, drop = FALSE]
} else {
  X_train_final <- X_train_scaled
  X_test_final  <- X_test_scaled
}

cat("Final number of predictors after preprocessing:\n")
print(ncol(X_train_final))


# -------------------------------------------------------------
# 6. PCA: DIMENSIONALITY REDUCTION
# -------------------------------------------------------------
# 6.1 Fit PCA on training data only

pca_model <- prcomp(
  X_train_final,
  center = FALSE,
  scale. = FALSE
)

# 6.2 Calculate variance explained
var_explained <- (pca_model$sdev)^2 / sum((pca_model$sdev)^2)
cum_var <- cumsum(var_explained)

pca_df <- data.frame(
  PC = 1:length(var_explained),
  Variance = var_explained,
  Cumulative = cum_var
)

# 6.3 Scree plot
p_scree <- ggplot(pca_df, aes(x = PC, y = Variance)) +
  geom_line() +
  geom_point() +
  labs(
    title = "Scree Plot",
    x = "Principal Component",
    y = "Variance Explained"
  ) +
  theme_minimal()

print(p_scree)

# 6.4 Cumulative variance plot
p_cumulative <- ggplot(pca_df, aes(x = PC, y = Cumulative)) +
  geom_line() +
  geom_point() +
  geom_hline(yintercept = 0.95, linetype = "dashed") +
  labs(
    title = "Cumulative Variance Explained",
    x = "Number of Principal Components",
    y = "Cumulative Variance"
  ) +
  theme_minimal()

print(p_cumulative)

# 6.5 Decide number of components for 95% variance
num_comp <- which(cum_var >= 0.95)[1]

cat("Number of PCA components retaining 95% variance:\n")
print(num_comp)

# 6.6 Transform training and test data using same PCA model
X_train_pca <- pca_model$x[, 1:num_comp, drop = FALSE]

X_test_pca <- predict(
  pca_model,
  newdata = X_test_final
)[, 1:num_comp, drop = FALSE]

cat("PCA training data dimensions:\n")
print(dim(X_train_pca))

cat("PCA test data dimensions:\n")
print(dim(X_test_pca))

# -------------------------------------------------------------
# 7. Class Label Definition
# -------------------------------------------------------------
# Death is the minority class.
# We handle imbalance using upsampling inside cross-validation.

y_train <- relevel(y_train, ref = "Death")
y_test  <- relevel(y_test, ref = "Death")

cat("\nTraining class distribution:\n")
print(table(y_train))


# -------------------------------------------------------------
# 8. MODEL TRAINING CONTROL
# -------------------------------------------------------------
# 5-fold cross-validation + upsampling.
# Upsampling is done only inside training folds.

ctrl <- trainControl(
  method = "cv",
  number = 5,
  sampling = "up",
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)


# -------------------------------------------------------------
# 9. TRAIN MODELS
# -------------------------------------------------------------
# Four models:
# 1. RF original
# 2. ANN original
# 3. RF PCA
# 4. ANN PCA


rf_original <- train(
  x = X_train_final,
  y = y_train,
  method = "rf",
  metric = "ROC",
  trControl = ctrl,
  ntree = 300,
  tuneLength = 5
)

ann_original <- train(
  x = X_train_final,
  y = y_train,
  method = "nnet",
  metric = "ROC",
  trControl = ctrl,
  tuneLength = 5,
  trace = FALSE,
  maxit = 300,
  MaxNWts = 20000
)

rf_pca <- train(
  x = X_train_pca,
  y = y_train,
  method = "rf",
  metric = "ROC",
  trControl = ctrl,
  ntree = 300,
  tuneLength = 5
)

ann_pca <- train(
  x = X_train_pca,
  y = y_train,
  method = "nnet",
  metric = "ROC",
  trControl = ctrl,
  tuneLength = 5,
  trace = FALSE,
  maxit = 300,
  MaxNWts = 20000
)

cat("\nRF Original Results:\n")
print(rf_original$results)
cat("\nBest RF Original Parameter:\n")
print(rf_original$bestTune)

cat("\nANN Original Results:\n")
print(ann_original$results)
cat("\nBest ANN Original Parameters:\n")
print(ann_original$bestTune)

cat("\nRF PCA Results:\n")
print(rf_pca$results)
cat("\nBest RF PCA Parameter:\n")
print(rf_pca$bestTune)

cat("\nANN PCA Results:\n")
print(ann_pca$results)
cat("\nBest ANN PCA Parameters:\n")
print(ann_pca$bestTune)


# -------------------------------------------------------------
# 10. TEST SET PREDICTIONS AND CONFUSION MATRICES
# -------------------------------------------------------------
# The final evaluation must be done on the unseen test set.
# Positive class = Death.

# RF original
rf_original_pred <- predict(rf_original, X_test_final)
rf_original_prob <- predict(rf_original, X_test_final, type = "prob")[, "Death"]
rf_original_cm <- confusionMatrix(rf_original_pred, y_test, positive = "Death")

# ANN original
ann_original_pred <- predict(ann_original, X_test_final)
ann_original_prob <- predict(ann_original, X_test_final, type = "prob")[, "Death"]
ann_original_cm <- confusionMatrix(ann_original_pred, y_test, positive = "Death")

# RF PCA
rf_pca_pred <- predict(rf_pca, X_test_pca)
rf_pca_prob <- predict(rf_pca, X_test_pca, type = "prob")[, "Death"]
rf_pca_cm <- confusionMatrix(rf_pca_pred, y_test, positive = "Death")

# ANN PCA
ann_pca_pred <- predict(ann_pca, X_test_pca)
ann_pca_prob <- predict(ann_pca, X_test_pca, type = "prob")[, "Death"]
ann_pca_cm <- confusionMatrix(ann_pca_pred, y_test, positive = "Death")

cat("\nConfusion Matrix: RF Original\n")
print(rf_original_cm)

cat("\nConfusion Matrix: ANN Original\n")
print(ann_original_cm)

cat("\nConfusion Matrix: RF PCA\n")
print(rf_pca_cm)

cat("\nConfusion Matrix: ANN PCA\n")
print(ann_pca_cm)

#-------------------------------------------------------------
# 11. PERFORMANCE METRICS TABLE
# -------------------------------------------------------------

calculate_metrics <- function(model_name, cm, prob_death) {
  roc_auc <- as.numeric(auc(roc(y_test, prob_death, levels = c("Survived", "Death"), quiet = TRUE)))
  precision <- as.numeric(cm$byClass["Precision"])
  sensitivity <- as.numeric(cm$byClass["Sensitivity"])
  f1 <- 2 * precision * sensitivity / (precision + sensitivity)
  
  data.frame(
    Model = model_name,
    Accuracy = as.numeric(cm$overall["Accuracy"]),
    Kappa = as.numeric(cm$overall["Kappa"]),
    Sensitivity_Death = sensitivity,
    Specificity_Survived = as.numeric(cm$byClass["Specificity"]),
    Precision_Death = precision,
    F1_Death = f1,
    ROC_AUC = roc_auc
  )
}

results <- rbind(
  calculate_metrics("RF_Original", rf_original_cm, rf_original_prob),
  calculate_metrics("ANN_Original", ann_original_cm, ann_original_prob),
  calculate_metrics("RF_PCA", rf_pca_cm, rf_pca_prob),
  calculate_metrics("ANN_PCA", ann_pca_cm, ann_pca_prob)
)

cat("\nFinal model performance table:\n")
results_rounded <- results
results_rounded[, -1] <- round(results_rounded[, -1], 4)
print(results_rounded)

# -------------------------------------------------------------
# 12. ROC CURVE PLOT
# -------------------------------------------------------------

roc_rf_original <- roc(y_test, rf_original_prob, levels = c("Survived", "Death"), quiet = TRUE)
roc_ann_original <- roc(y_test, ann_original_prob, levels = c("Survived", "Death"), quiet = TRUE)
roc_rf_pca <- roc(y_test, rf_pca_prob, levels = c("Survived", "Death"), quiet = TRUE)
roc_ann_pca <- roc(y_test, ann_pca_prob, levels = c("Survived", "Death"), quiet = TRUE)

plot(roc_rf_original, col = "blue", lwd = 2,
     main = "ROC Curves for RF and ANN Models")

lines(roc_ann_original, col = "red", lwd = 2)
lines(roc_rf_pca, col = "green", lwd = 2)
lines(roc_ann_pca, col = "purple", lwd = 2)

legend("bottomright",
       legend = c("RF Original", "ANN Original", "RF PCA", "ANN PCA"),
       col = c("blue", "red", "green", "purple"),
       lwd = 2)
# -------------------------------------------------------------
# SAVE FINAL MODEL- ANN Original AND PREPROCESSING OBJECTS FOR SHINY DEPLOYMENT
# -------------------------------------------------------------

mode_values <- list()
for (col in categorical_cols) {
  mode_values[[col]] <- mode_value(X_train[[col]])
}

median_values <- list()
for (col in continuous_cols) {
  median_values[[col]] <- median(X_train[[col]], na.rm = TRUE)
}

deployment_objects <- list(
  model = ann_original,
  model_name = "ANN Original",
  keep_cols = keep_cols,
  categorical_cols = categorical_cols,
  continuous_cols = continuous_cols,
  mode_values = mode_values,
  median_values = median_values,
  scale_obj = scale_obj,
  nzv_cols = nzv_cols,
  final_feature_names = colnames(X_train_final),
  positive_class = "Death",
  negative_class = "Survived",
  classification_threshold = 0.5
)

saveRDS(deployment_objects, "mi_ann_model.rds")
