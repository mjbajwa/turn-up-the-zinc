# Load Libraries ----------------------------------------------------------

library(tidyverse)
library(h2o)
library(zoo)

# Load Data ---------------------------------------------------------------

dictionary <- fromJSON(file = "./data/data_dictionary_v1.json")
data.all <- read_csv("./data/train_data/all_train.csv")

# Attempt to Predict Rougher Process for now ------------------------------

rel_columns <- names(data.all)[names(data.all) %>% str_detect("rougher")]
data_rougher_temp <- data.all %>% select(date, rel_columns) %>% na.locf()
input_cols <- names(data_rougher_temp)[names(data_rougher_temp) %>% str_detect("input|state")]
output_cols <- names(data_rougher_temp)[names(data_rougher_temp) %>% str_detect("recovery")] # rougher recovery only
data_rougher <- data_rougher_temp %>% select(input_cols, output_cols)

# cols_average <- FALSE
# if(cols_average){
#   data_rougher <- data_rougher %>%
#     na.locf() %>%
#     mutate(average_level.state = rowMeans(.[grep("level", names(.))]),
#            average_air.state = rowMeans(.[grep("level", names(.))]))
#   del_columns_2 <- names(data_rougher)[names(data_rougher) %>% str_detect("rougher.state")]
#   data_rougher <- data_rougher %>% 
#     select(-del_columns_2)
# }
# 
# data_rougher <- data_rougher %>% 
#   select(input_cols, output_cols)
# 
# # Use H2O -----------------------------------------------------------------
# 
# h2o.removeAll()
# h2o.init(nthreads = 6)
# 
# df <- as.h2o(data_rougher %>% filter(output_cols > 0))
# splits <- h2o.splitFrame(df, 0.8, seed = 1234)
# train  <- h2o.assign(splits[[1]], "train.hex") # 60%
# valid  <- h2o.assign(splits[[2]], "valid.hex") # 20%
# # test   <- h2o.assign(splits[[3]], "test.hex")  # 20%
# 
# response <- "rougher.output.recovery"
# predictors <- setdiff(names(df), response)
# predictors
# 
# model <- h2o.deeplearning(
#   model_id = "dl_model_first", 
#   training_frame = df, 
#   #nfolds = 5,
#   #fold_assignment = "Random",
#   #validation_frame = valid,   ## validation dataset: used for scoring and early stopping
#   x = predictors,
#   y = response,
#   #activation="Rectifier",  ## default
#   hidden=c(100, 100, 100),       ## default: 2 hidden layers with 200 neurons each
#   epochs=100,
#   nfolds=5,
#   fold_assignment="Modulo" # can be "AUTO", "Modulo", "Random" or "Stratified"
#   # verbose = TRUE## not enabled by default
# )
# 
# summary(model)
# 
# results <- m1@model$scoring_history %>% as.data.frame()
# ggplot(results, aes(x = epochs)) + 
#   geom_point(aes(y = training_rmse), color = "red2") + 
#   geom_point(aes(y = validation_rmse), color = "green3") + 
#   theme_bw()
# 
# 
# # NN Join Predictions to Rougher ---------------------------
# 
# predictions <- h2o.predict(model, df) %>% as.data.frame()
# 
# data_rougher <- data_rougher %>% 
#   mutate(rougher_recovery_predictions = ifelse(predictions$predict > 0, predictions$predict, 0))
# 
# train_idx <- train_idx %>% sort()
# test_idx <- (1:nrow(data_rougher))[-train_idx]
# 
# MASE.test <- function(){
#   
#   # Numerator --------------
#   
#   st.error <- data_rougher %>% 
#     select(rougher.output.recovery, rougher_recovery_predictions) %>% 
#     mutate(residuals = rougher.output.recovery - rougher_recovery_predictions) %>% 
#     pull(residuals) %>% 
#     abs() %>% 
#     sum()
#   
#   # Denominator
#   
#   idx <- 2:nrow(data_rougher)
#   naive.error <- data_rougher[idx, "rougher.output.recovery"] - data_rougher[idx-1,"rougher.output.recovery"]
#   naive.error <- sum(abs(naive.error))
#   
#   # Final value
#   T <- nrow(data_rougher)
#   MASE <- st.error/(T/(T-1)*naive.error)
#   return(MASE)
# }
# 
# MASE.test()

# Auto ML Philosophy -------------------------------------

h2o.removeAll()
h2o.init(nthreads = 7)

train_idx <- sample(1:nrow(data_rougher), 0.95*nrow(data_rougher)) # Take 75% of the data as training
df <- data_rougher %>% filter(output_cols > 0)
df.train <- data_rougher[train_idx, ] %>% as.h2o()
df.test <- data_rougher[-train_idx, ] %>% as.h2o()

response <- "rougher.output.recovery"
predictors <- setdiff(names(df), response)

# Run AutoML for 20 base models (limited to 1 hour max runtime by default)

aml.rougher <- h2o.automl(x = predictors, 
                          y = response,
                          training_frame = df.train,
                          max_models = 20,
                          seed = 1)

# View the AutoML Leaderboard

lb <- aml.rougher@leaderboard
print(lb, n = nrow(lb))  # Print all rows instead of default (6 rows)
aml.rougher@leader

# Predict with Auto-ML

predictions <- h2o.predict(aml.rougher, df %>% as.h2o()) %>% as.data.frame()

data_rougher <- data_rougher %>% 
  mutate(rougher_recovery_predictions = ifelse(predictions$predict > 0, predictions$predict, 0))

train_idx <- train_idx %>% sort()
test_idx <- (1:nrow(data_rougher))[-train_idx]

MASE.test <- function(){
  
  # Numerator --------------
  
  st.error <- data_rougher %>% 
    select(rougher.output.recovery, rougher_recovery_predictions) %>% 
    mutate(residuals = rougher.output.recovery - rougher_recovery_predictions) %>% 
    pull(residuals) %>% 
    abs() %>% 
    sum()
  
  # Denominator
  
  idx <- 2:nrow(data_rougher)
  naive.error <- data_rougher[idx, "rougher.output.recovery"] - data_rougher[idx-1,"rougher.output.recovery"]
  naive.error <- sum(abs(naive.error))
  
  # Final value
  T <- nrow(data_rougher)
  MASE <- st.error/(T/(T-1)*naive.error)
  return(MASE)
}

MASE.test()

# Sample Plots ------------------------------------------------------------

ggplot(data_rougher) + 
  geom_point(aes(x = rougher.output.recovery, y = rougher_recovery_predictions, 
                 color = abs(rougher.output.recovery - rougher_recovery_predictions)), size = 2) + 
  geom_abline(slope = 1, intercept = 0, size = 1.2, color = "green2", linetype = 2) + 
  scale_color_gradient(low = "green4", high = "red3", guide = FALSE) + 
  theme_bw()


## Save Model --------------------------------------------------------------
# 
# # save the model
# model_path <- h2o.saveModel(object = aml.rougher, path="./cache/", force=TRUE)
# 
# print(model_path)
# 
# # load the model
# 
# aml.rougher <- h2o.loadModel(model_path)

# Make Predictions --------------------------------------------------------

test_set <- read_csv("./data/test_data/all_test.csv")
test_data <- test_set %>% select(predictors)
predictions <- h2o.predict(aml.rougher, test_data %>% as.h2o()) %>% as.data.frame()
results <- ifelse(predictions$predict < 0, 0, ifelse(predictions$predict > 100, 99, predictions$predict))
results %>% hist(col = "red2")
test_set <- test_set %>% mutate(rougher.output.recovery = results) %>% select(date, rougher.output.recovery)
write_rds(test_set, "./submissions/feb-15/test_set_rougher_recovery.rds")
