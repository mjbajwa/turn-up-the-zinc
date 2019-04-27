# Load Libraries ----------------------------------------------------------

library(tidyverse)
library(xgboost)
library(caret)
library(rjson)
library(dygraphs)
library(xts)
library(keras)
library(GGally)
library(scales)
library(ggcorrplot)
library(h2o)

# Load Data ---------------------------------------------------------------

dictionary <- fromJSON(file = "./data/data_dictionary_v1.json")
data.all <- read_csv("./data/train_data/all_train.csv")
data.all %>% names
data_filt <- data.all %>% select(date, primary_cleaner.state.floatbank8_a_air)

# Clean Up Data For Rougher Process ------------------------------

rel_columns <- names(data.all)[names(data.all) %>% str_detect("rougher")]
data_rougher <- data.all %>% select(date, rel_columns) %>% na.omit()
input_cols <- names(data_rougher)[names(data_rougher) %>% str_detect("input")]
state_cols <- names(data_rougher)[names(data_rougher) %>% str_detect("state")]
output_cols <- names(data_rougher)[names(data_rougher) %>% str_detect("rougher.output.recovery")] # rougher recovery only

data_rougher <- data_rougher %>% 
  select(date, input_cols, state_cols, output_cols) #%>% 
  #na.locf() # Critical Point. 

cols_average <- FALSE

if(cols_average){
  data_rougher <- data_rougher %>%
    na.locf() %>%
    mutate(average_level.state = rowMeans(.[grep("level", names(.))]),
           average_air.state = rowMeans(.[grep("air", names(.))]))
  del_columns_2 <- names(data_rougher)[names(data_rougher) %>% str_detect("rougher.state")]
  data_rougher <- data_rougher %>% 
    select(-del_columns_2)
}

state_cols <- names(data_rougher)[names(data_rougher) %>% str_detect("state")] # update state_cols

lagged.df <- data_rougher

# Create Lag Variables for Input and State Columns ------------------------

# Convert to a function to test data can easily be converted

# Input stationary lags

lags <- seq(2)
lag_names <- paste("lag", formatC(lags, width = nchar(max(lags)), flag = "0"), sep = "_")
lag_functions <- setNames(paste("dplyr::lag(., ", lags, ")"), lag_names)

# State lags

lagged.df <- data_rougher %>% 
  select(date, input_cols, state_cols, output_cols) %>% 
  mutate_at(.vars = c(input_cols, state_cols),
            .funs = funs_(lag_functions)) %>% 
  na.omit() %>% 
  mutate(row_index = 1:n())

fix_lags <- FALSE

if(fix_lags)

  lagged.df <- lagged.df %>%
    select(date,
           -input_cols,
           paste0(input_cols, "_lag_", c(rep(1, length(input_cols)))),
           contains("state"),
           output_cols) #%>% 
  #filter(get(output_cols) < 50) 

lagged.df %>% names # just checking

# Check to see if it works.

# lagged.df <- lagged.df %>% select(contains("lag_1"), output_cols) %>% na.omit()

# # Create Simple H2O Model for Predictions ---------------------------------
# 
# library(h2o)
# 
# h2o.removeAll()
# h2o.init(nthreads = 6)
# 
# df <- as.h2o(lagged.df %>% select(-date))
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
#   hidden = c(100, 100),       ## default: 2 hidden layers with 200 neurons each
#   epochs = 100,
#   nfolds = 5,
#   fold_assignment = "Modulo" # can be "AUTO", "Modulo", "Random" or "Stratified"
#   # verbose = TRUE## not enabled by default
# )
# 
# summary(model)
# 
# model@model$variable_importances %>% as.data.frame() %>% head(10) %>%
#   ggplot() + 
#   geom_bar(aes(x = reorder(variable, -relative_importance), y = relative_importance, fill = relative_importance), 
#            stat = "identity", color = "black", alpha = 0.6) + 
#   coord_flip() + 
#   scale_fill_gradient(high = "green4", low = "red")
# 
# # Make some predictions
# 
# predictions <- h2o.predict(model, df) %>% as.data.frame()
# 
# data_rougher_h2o <- lagged.df %>% 
#   mutate(rougher_recovery_predictions_h2o = ifelse(predictions$predict > 0, predictions$predict, 0))

# Auto-ML H2O -------------------------------------------------------------

h2o.removeAll()
h2o.init(nthreads = 7)

# train_idx <- sample(1:nrow(lagged.df), 0.90*nrow(lagged.df)) # Take 75% of the data as training

data_density_correction <- TRUE

if(data_density_correction){
  
  rows.tails <- lagged.df %>% filter(get(output_cols) < 60) %>% pull(row_index)
  rows.tails <- c(rows.tails, lagged.df %>% filter(get(output_cols) > 95) %>% pull(row_index)) %>% sort()
  rows.medium <- lagged.df %>% filter(!(row_index %in% rows.tails)) %>% pull(row_index) %>% sort()
  
  x_tails <- 0.80
  x_medium <- 0.60
  
  train_idx <- c(sample(rows.tails, x_tails*length(rows.tails)), 
                 sample(rows.medium, x_medium*length(rows.medium))) %>% sort() %>% sample()
  
  tail_density <- x_tails*length(rows.tails)/length(train_idx)
  print(length(train_idx)/nrow(lagged.df))
  
}

df <- lagged.df %>% select(-c(date, row_index))
df.train <- df[train_idx, ] %>% as.h2o()
df.test <- df[-train_idx, ] %>% as.h2o()

response <- "rougher.output.recovery"
predictors <- setdiff(names(df), response) # This is used later ON - DO NOT CHANGE - AFFECTS DOWNSTREAM SIGNIFICANTLY

# Run AutoML for 20 base models (limited to 1 hour max runtime by default)

aml.rougher <- h2o.automl(x = predictors, 
                          y = response,
                          training_frame = df.train,
                          leaderboard_frame = df.test,
                          nfolds = 5,
                          max_models = 20,
                          stopping_metric = "MAE",
                          sort_metric = "MAE",
                          seed = 1)

# View the AutoML Leaderboard

lb <- aml.rougher@leaderboard
print(lb, n = nrow(lb))  # Print all rows instead of default (6 rows)
aml.rougher@leader

# Predict with Auto-ML

predictions <- h2o.predict(aml.rougher, df %>% as.h2o()) %>% as.data.frame()

# data_rougher_2 <- data_rougher %>% 
#   mutate(rougher_recovery_predictions = c(rep(NA, length(lags)), (ifelse(predictions$predict > 0, predictions$predict, 0))))
# 
# train_idx <- train_idx %>% sort()
# test_idx <- (1:nrow(data_rougher_2))[-train_idx]

# Sample Plots 

# ggplot(data_rougher_2) + 
#   geom_point(aes(x = rougher.output.recovery, y = rougher_recovery_predictions, 
#                  color = abs(rougher.output.recovery - rougher_recovery_predictions)), size = 2) + 
#   geom_abline(slope = 1, intercept = 0, size = 1.2, color = "green2", linetype = 2) + 
#   scale_color_gradient(low = "green4", high = "red3", guide = FALSE) + 
#   theme_bw()

# MASE.test(df = data_rougher_2 %>% na.omit(), col = "rougher_recovery_predictions")

# Make Predictions --------------------------------------------------------

test_set <- read_csv("./data/test_data/all_test.csv")
all_data <- bind_rows(data_rougher %>% 
                        select(date, input_cols, state_cols), 
                      test_set %>% 
                        select(date, input_cols, state_cols)) %>% arrange(date)

# Ensure the transformations below are the exact same as above (same variables)

all_data <- all_data %>% 
  select(date, input_cols, state_cols) %>% 
  mutate_at(.vars = c(input_cols, state_cols),
            .funs = funs_(lag_functions)) %>% 
  select(date, predictors)

test_data <- all_data %>% filter(date %in% test_set$date) %>% arrange(date)
predictions <- h2o.predict(aml.rougher, test_data %>% select(-date) %>% as.h2o()) %>% as.data.frame()
results <- ifelse(predictions$predict < 0, 0, ifelse(predictions$predict > 100, 99, predictions$predict))
results %>% hist(col = "red2")
test_set <- test_set %>% mutate(rougher.output.recovery = results) %>% select(date, rougher.output.recovery)
write_rds(test_set, "./submissions/feb-17/test_set_rougher_recovery.rds")

# # Fit a Keras Model -------------------------------------------------------
# 
# input_cols <- names(lagged.df)[names(lagged.df) %>% str_detect("input|state")]
# output_cols <- names(lagged.df)[names(lagged.df) %>% str_detect("rougher.output.recovery")]
# 
# # Simple Deep Feedforward Neural Network 
# 
# data_NN <- data.matrix(lagged.df %>% select(-date))
# train_idx <- sample(1:nrow(data_NN), 0.70*nrow(data_NN)) # Take 75% of the data as training
# #train_idx <- seq(1,lagged.df %>% filter(date < as.Date("2018-01-01")) %>% nrow())
# train_data <- data_NN[train_idx,]
# mean <- apply(train_data, 2, mean)
# std <- apply(train_data, 2, sd)
# data_NN <- scale(data_NN, center = mean, scale = std)
# 
# # data_NN[, "rougher.output.recovery"] %>% hist(col = "red2")
# # Define Train and Test
# 
# data_train <- data_NN[train_idx,]
# data_test <- data_NN[-train_idx,]
# 
# # Change Rougher Table
# 
# data_rougher_NN <- lagged.df %>% mutate(label = ifelse(row_number() %in% train_idx, "train", "test"))
# 
# # Create simple dense network 
# 
# model_2 <- keras_model_sequential() %>% 
#   layer_dense(units = input_cols %>% length(), activation = 'relu', input_shape = c(input_cols %>% length())) %>% 
#   layer_dropout(rate = 0.10) %>% 
#   layer_dense(units = 100, activation = 'relu') %>%
#   layer_dropout(rate = 0.10) %>%
#   layer_dense(units = 100, activation = 'relu') %>%
#   layer_dropout(rate = 0.10) %>%
#   layer_dense(units = output_cols %>% length())
# 
# model_2 %>% compile(
#   optimizer = optimizer_adam(lr = 0.001),
#   loss = "mse",
#   metrics = c('mae')
# )
# 
# callbacks <- list(callback_model_checkpoint(filepath = "./cache/weights_1.hdf5", save_best_only = TRUE))
# 
# history <- model_2 %>% fit(
#   x = data_train[,input_cols],
#   y = data_train[,output_cols],
#   validation_split = 0.10,
#   epochs = 300,
#   batch_size = 128,
#   callbacks = callbacks, 
#   shuffle = TRUE
# )
# 
# print(history$metrics$val_mean_absolute_error %>% min() %>% `*`(std[output_cols]))
# 
# predictions <- model_2 %>% predict(data_NN[,input_cols])*std[output_cols] + mean[output_cols]
# 
# data_rougher_NN <- lagged.df %>% 
#   mutate(rougher_recovery_predictions = ifelse(predictions < 0, 0, predictions))
# 
# train_idx <- train_idx %>% sort()
# 
# test_idx <- (1:nrow(lagged.df))[-train_idx]
# 
# MASE.test <- function(df = data_rougher_NN, col = "rougher_recovery_predictions", index = "All"){
#   
#   # Numerator --------------
#   
#   st.error <- df %>% 
#     select(rougher.output.recovery, col) %>% 
#     mutate(residuals = rougher.output.recovery - get(col)) %>% 
#     pull(residuals) %>% 
#     abs() %>% 
#     sum()
#   
#   # Denominator
#   
#   if(index == "ALL"){
#     idx <- 2:nrow(df) 
#   } else {
#     idx <- test_idx[-1]
#   }
#     naive.error <- df[idx, "rougher.output.recovery"] - df[idx-1,"rougher.output.recovery"]
#     naive.error <- sum(abs(naive.error))
#     
#     # Final value
#     T <- length(idx)
#     MASE <- st.error/(T/(T-1)*naive.error)
#     return(MASE)
# }
# 
# MASE.test(df = data_rougher_NN, col = "rougher_recovery_predictions")
# 
# data_rougher_NN <- data_rougher_NN %>% mutate(residuals = abs(rougher.output.recovery - rougher_recovery_predictions))
# 
# # Some Plots
# 
# ggplot(data_rougher_NN %>% mutate(year = lubridate::year(date)) %>% filter(year >= lubridate::year("2016-01-01"))) + 
#   geom_point(aes(x = date, y = rougher.output.recovery), color = "blue2") +
#   geom_line(aes(x = date, y = rougher.output.recovery), color = "blue2") +
#   geom_point(aes(x = date, y = rougher_recovery_predictions), color = "red2") +
#   geom_line(aes(x = date, y = rougher_recovery_predictions), color = "red2") + 
#   scale_x_datetime(labels=date_format ("%m-%y")) + 
#   theme_bw() + 
#   facet_wrap(year~., scales = "free")
# 
# ggplot(data_rougher_NN) + 
#   geom_histogram(aes(x = residuals, weight = residuals), binwidth = 25, color = "black", alpha = 0.6)
