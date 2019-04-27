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

# Add Derived Columns -----------------------------------------------------

# Derive Total Zinc Flow out of rougher process and secondary cleaning process at any given point in time.

data.all <- data.all %>% 
  mutate(final.output.concentrate_zn_flow = (final.output.recovery/100)*rougher.input.feed_rate*(rougher.input.feed_zn/100), 
         rougher.output.concentrate_zn_flow = (rougher.output.recovery/100)*rougher.input.feed_rate*(rougher.input.feed_zn/100))

# Define Closure -------------------------------------------------

lags_rougher <- 1:2
lags_primary <- 1
lags_secondary <- 1

lag_functions_rougher_ <- setNames(paste("dplyr::lag(., ", lags_rougher, ")"), 
                                   paste("lag", formatC(lags_rougher, width = nchar(max(lags_rougher)), flag = "0"), sep = "_"))

lag_functions_primary_ <- setNames(paste("dplyr::lag(., ", lags_primary, ")"), 
                                   paste("lag", formatC(lags_primary, width = nchar(max(lags_primary)), flag = "0"), sep = "_"))

lag_functions_secondary_ <- setNames(paste("dplyr::lag(., ", lags_secondary, ")"), 
                                     paste("lag", formatC(lags_secondary, width = nchar(max(lags_secondary)), flag = "0"), sep = "_"))

# Rougher Columns -----------------------------------------------------------

rougher_columns <- names(data.all)[names(data.all) %>% str_detect("rougher")]
rougher_input_state_columns <- rougher_columns[rougher_columns %>% str_detect("input|state")]

# Primary Variables -------------------------------------------------------

primary_columns <- names(data.all)[names(data.all) %>% str_detect("primary")]
primary_input_state_columns <- primary_columns[primary_columns %>% str_detect("input|state")]

# Secondary Variables -----------------------------------------------------

secondary_columns <- names(data.all)[names(data.all) %>% str_detect("secondary")]
secondary_input_state_columns <- secondary_columns[secondary_columns %>% str_detect("input|state")]

output_cols <- "final.output.concentrate_zn_flow" #"final.output.recovery"

# Create Lags and Filter Data ---------------------------------------------

apply_lags <- TRUE

if(apply_lags == TRUE) {
  
  lagged.df <- data.all %>%
    select(
      date,
      rougher_input_state_columns,
      primary_input_state_columns,
      secondary_input_state_columns,
      final.output.concentrate_zn,
      final.output.recovery,
      output_cols
    ) %>%
    mutate_at(
      .vars = c(rougher_input_state_columns),
      .funs = funs_(lag_functions_rougher_)
    ) %>%
    mutate_at(
      .vars = c(primary_input_state_columns),
      .funs = funs_(lag_functions_primary_)
    ) %>%
    mutate_at(
      .vars = c(secondary_input_state_columns),
      .funs = funs_(lag_functions_secondary_)
    ) %>%
    # filter(rougher.input.feed_zn > 0.5,
    #        rougher.input.feed_rate > 10, 
    #        final.output.concentrate_zn > 2) %>%  
    na.omit() %>%
    mutate(row_index = 1:n())
  
  fix_lags <- FALSE
  
  if (fix_lags)
  {
    lagged.df <- lagged.df %>%
      select(-c(rougher_input_state_columns,
                paste0(secondary_input_state_columns, "_lag_", c(rep(1, length(secondary_input_state_columns))))))
  }
  
} else {
  
  lagged.df <- data.all %>%
    select(
      date,
      rougher_input_state_columns,
      primary_input_state_columns,
      secondary_input_state_columns,
      final.output.concentrate_zn,
      output_cols
    ) %>%
    filter(rougher.input.feed_zn > 0.5,
      rougher.input.feed_rate > 10, 
      final.output.concentrate_zn > 20) %>%
    na.omit() %>%
    mutate(row_index = 1:n())
}

lagged.df$final.output.concentrate_zn <- NULL
# lagged.df$final.output.recovery %>% hist(col = "red2")

# Fit a Keras Model -------------------------------------------------------

input_cols <- names(lagged.df)[names(lagged.df) %>% str_detect("input|state")]
output_cols <- output_cols

# Split the data (lagged.df) to supply a slightly better proportion of low values to the model

data_density_correction <- FALSE

if(data_density_correction){
  
  rows.tails <- lagged.df %>% filter(final.output.recovery < 50) %>% pull(row_index)
  rows.tails <- c(rows.tails, lagged.df %>% filter(final.output.recovery > 80) %>% pull(row_index)) %>% sort()
  rows.medium <- lagged.df %>% filter(!(row_index %in% rows.tails)) %>% pull(row_index) %>% sort()
  
  x_tails <- 0.80
  x_medium <- 0.55
  
  train_idx <- c(sample(rows.tails, x_tails*length(rows.tails)), 
                 sample(rows.medium, x_medium*length(rows.medium))) %>% sort() %>% sample()
  
  tail_density <- x_tails*length(rows.tails)/length(train_idx)
  print(length(train_idx)/nrow(lagged.df))
  
}

# Define Matrices and stuff for

data_NN <- data.matrix(lagged.df %>% select(-c(date, row_index, final.output.recovery)))
train_idx <- sample(1:nrow(data_NN), 0.80*nrow(data_NN)) # Take 75% of the data as training
# train_idx <- seq(1,lagged.df %>% filter(date < as.Date("2018-01-01")) %>% nrow()) %>% sample()
train_data <- data_NN[train_idx,]
mean.nn <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
data_NN <- scale(data_NN, center = mean.nn, scale = std)

# data_NN[, "rougher.output.recovery"] %>% hist(col = "red2")
# Define Train and Test

data_train <- data_NN[train_idx,]
data_test <- data_NN[-train_idx,]

# Change Rougher Table

data_rougher_NN <- lagged.df %>% mutate(label = ifelse(row_number() %in% train_idx, "train", "test"))

# Create simple dense network

model_2 <- keras_model_sequential() %>%
  layer_dense(units = input_cols %>% length(), input_shape = c(input_cols %>% length())) %>%
  layer_batch_normalization() %>% 
  layer_activation(activation = "relu") %>% 
  layer_dropout(rate = 0.10) %>%
  layer_dense(units = 50) %>%
  layer_batch_normalization() %>% 
  layer_activation(activation = "relu") %>% 
  layer_dropout(rate = 0.15) %>%
  layer_dense(units = 50) %>%
  layer_batch_normalization() %>% 
  layer_activation(activation = "relu") %>% 
  layer_dropout(rate = 0.15) %>%
  layer_dense(units = output_cols %>% length())

model_2 %>% compile(
  optimizer = optimizer_adam(lr = 0.001),
  loss = "mse",
  metrics = c('mae')
)

callbacks <- list(callback_model_checkpoint(filepath = "./cache/weights_1.hdf5", save_best_only = TRUE))

history <- model_2 %>% fit(
  x = data_train[,input_cols],
  y = data_train[,output_cols],
  validation_split = 0.10,
  epochs = 50,
  batch_size = 128,
  callbacks = callbacks,
  shuffle = TRUE
)

print(history$metrics$val_mean_absolute_error %>% min() %>% `*`(std[output_cols]))

predictions <- model_2 %>% predict(data_NN[,input_cols])*std[output_cols] + mean.nn[output_cols]

data_final_NN <- lagged.df %>%
  mutate(final_recovery_predictions = predictions/(rougher.input.feed_rate*(rougher.input.feed_zn/100))*100) %>% 
  mutate(final_recovery_predictions = ifelse(final_recovery_predictions < 0, 0, ifelse(final_recovery_predictions > 100, 100, final_recovery_predictions))) %>% 
  mutate(residuals = get(output_cols) - final_recovery_predictions, 
         label = ifelse(row_number() %in% (train_idx %>% sort()), "train", "test"))

data_final_NN$final_recovery_predictions %>% hist(col = "red2")

# Analyze Residuals ---------------------------------------------------------------

ggplot(data_final_NN) + 
  #geom_histogram(aes(x = abs(residuals), weight = abs(residuals)), color = "black", fill = "red2", alpha = 0.6) + 
  geom_histogram(aes(x = abs(residuals), y = cumsum(..count..)/sum(..count..)*100, weight = abs(residuals)), 
                 color = "black", fill = "red2", alpha = 0.6)

test_residuals <- data_final_NN %>% filter(label == "test") %>% pull(residuals) %>% abs()
# sum(test_residuals[test_residuals > 25])/sum(test_residuals)*100

ggplot(data_final_NN %>% mutate(year = lubridate::year(date)) %>% filter(year >= lubridate::year("2016-01-01"))) +
  geom_point(aes(x = date, y = final.output.recovery), color = "blue2") +
  geom_line(aes(x = date, y = final.output.recovery), color = "blue2") +
  geom_point(aes(x = date, y = final_recovery_predictions), color = "red2") +
  geom_line(aes(x = date, y = final_recovery_predictions), color = "red2") +
  scale_x_datetime(labels=date_format ("%m-%y")) +
  theme_bw() +
  facet_wrap(.~year, scales = "free")

ggplot(data_final_NN) + 
  geom_point(aes(x = final.output.recovery, y = final_recovery_predictions, color = label))

# Make Predictions --------------------------------------------------------

all_inputs <- c(rougher_input_state_columns, primary_input_state_columns, secondary_input_state_columns)
test_set <- read_csv("./data/test_data/all_test.csv")
all_data <- bind_rows(data.all %>% 
                        select(date, all_inputs), 
                      test_set %>% 
                        select(date, all_inputs)) %>% arrange(date)

# Ensure the transformations below are the exact same as above (same variables)

predictors <- setdiff(colnames(data_NN), output_cols) 

if(apply_lags == TRUE){
  
  all_data <- all_data %>% 
    select(date, all_inputs) %>% 
    mutate_at(.vars = c(rougher_input_state_columns),
              .funs = funs_(lag_functions_rougher_)) %>% 
    mutate_at(.vars = c(primary_input_state_columns), 
              .funs = funs_(lag_functions_primary_)) %>% 
    mutate_at(.vars = c(secondary_input_state_columns), 
              .funs = funs_(lag_functions_secondary_)) %>% 
    select(date, predictors)
  
} else {
  
  all_data <- all_data %>% 
    select(date, predictors)
}

test_data <- all_data %>% filter(date %in% test_set$date) %>% arrange(date)
test_matrix <- data.matrix(test_data %>% select(-date))
test_matrix <- scale(test_matrix, center = mean.nn[!(names(mean.nn) %in% output_cols)], scale = std[!(names(mean.nn) %in% output_cols)])
test_data <- test_data %>% 
  mutate(predictions = predict(model_2, test_matrix)*std[output_cols] + mean.nn[output_cols]) %>% 
  mutate(predictions = predictions/(rougher.input.feed_rate*(rougher.input.feed_zn/100))*100) %>% 
  mutate(predictions = ifelse(predictions < 0, 0, ifelse(predictions > 100, 100, predictions))) %>% 
  mutate(predictions = ifelse(rougher.input.feed_zn < 0.05, 100, predictions))
results <- test_data$predictions
results %>% hist(col = "red2")
test_set <- test_set %>% mutate(final.output.recovery = results) %>% select(date, final.output.recovery)
write_rds(test_set, "./submissions/feb-19/test_set_final_recovery_der.rds")

