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

# Add Derived Columns -----------------------------------------------------

# Derive Total Zinc Flow out of rougher process and secondary cleaning process at any given point in time.

data.all <- data.all %>% 
  mutate(final.output.concentrate_zn_flow = (final.output.recovery/100)*rougher.input.feed_rate*(rougher.input.feed_zn/100), 
         rougher.output.concentrate_zn_flow = (rougher.output.recovery/100)*rougher.input.feed_rate*(rougher.input.feed_zn/100))

# Clean Up Data For Rougher Process ------------------------------

rel_columns <- names(data.all)[names(data.all) %>% str_detect("rougher")]
data_rougher <- data.all %>% select(date, rel_columns) %>% na.omit()
input_cols <- names(data_rougher)[names(data_rougher) %>% str_detect("input")]
state_cols <- names(data_rougher)[names(data_rougher) %>% str_detect("state")]
output_cols <- names(data_rougher)[names(data_rougher) %>% str_detect("rougher.output.recovery")] # rougher recovery only
output_cols <- "rougher.output.concentrate_zn_flow"

data_rougher <- data_rougher %>% 
  select(date, input_cols, state_cols, output_cols, rougher.output.recovery, rougher.output.concentrate_zn) #%>% 
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

lags <- seq(1)
lag_names <- paste("lag", formatC(lags, width = nchar(max(lags)), flag = "0"), sep = "_")
lag_functions <- setNames(paste("dplyr::lag(., ", lags, ")"), lag_names)

# State lags

lagged.df <- data_rougher %>% 
  select(date, input_cols, state_cols, output_cols, rougher.output.concentrate_zn, rougher.output.recovery) %>% 
  mutate_at(.vars = c(input_cols, state_cols),
            .funs = funs_(lag_functions)) %>% 
  filter(rougher.input.feed_zn > 0.05,
    rougher.input.feed_rate > 5, 
    rougher.output.concentrate_zn > 5) %>%  
  na.omit() %>% 
  mutate(row_index = 1:n())

lagged.df$rougher.output.concentrate_zn <- NULL

fix_lags <- TRUE

if(fix_lags){
  
  lagged.df <- lagged.df %>%
    select(date,
           input_cols,
           c(paste0(input_cols, "_lag_", c(rep(1, length(input_cols))))),
           state_cols,
           output_cols,
           rougher.output.recovery,
           row_index) #%>% 
  #filter(get(output_cols) < 50) 
  
  lagged.df %>% names # just checking
  
}

lagged.df$rougher.output.concentrate_zn_flow %>% hist(col = "red2")

# Fit a Keras Model -------------------------------------------------------

input_cols <- names(lagged.df)[names(lagged.df) %>% str_detect("input|state")]
output_cols <- output_cols

# Split the data (lagged.df) to supply a slightly better proportion of low values to the model

data_density_correction <- FALSE

if(data_density_correction){
  
  rows.tails <- lagged.df %>% filter(get(output_cols) < 60) %>% pull(row_index)
  rows.tails <- c(rows.tails, lagged.df %>% filter(get(output_cols) > 95) %>% pull(row_index)) %>% sort()
  rows.medium <- lagged.df %>% filter(!(row_index %in% rows.tails)) %>% pull(row_index) %>% sort()
  
  x_tails <- 0.90
  x_medium <- 0.40
  
  train_idx <- c(sample(rows.tails, x_tails*length(rows.tails)), 
                 sample(rows.medium, x_medium*length(rows.medium))) %>% sort() %>% sample()
  
  tail_density <- x_tails*length(rows.tails)/length(train_idx)
  print(length(train_idx)/nrow(lagged.df))
  
}

# Define Matrices and stuff for

data_NN <- data.matrix(lagged.df %>% select(-c(date, row_index, rougher.output.recovery)))
train_idx <- sample(1:nrow(data_NN), 0.80*nrow(data_NN)) # Take 75% of the data as training
# train_idx <- seq(1,lagged.df %>% filter(date < as.Date("2018-01-01")) %>% nrow()) %>% sample()
train_data <- data_NN[train_idx,]
mean.nn <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
data_NN <- scale(data_NN, center = mean.nn, scale = std)

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
  mutate(rougher_recovery_predictions = predictions/(rougher.input.feed_rate*(rougher.input.feed_zn/100))*100) %>% 
  mutate(rougher_recovery_predictions = ifelse(rougher_recovery_predictions < 0, 0, ifelse(rougher_recovery_predictions > 100, 100, rougher_recovery_predictions))) %>% 
  mutate(residuals = rougher.output.recovery - rougher_recovery_predictions, 
         label = ifelse(row_number() %in% (train_idx %>% sort()), "train", "test"))

data_final_NN$rougher_recovery_predictions %>% hist(col = "red2")

# Make Some Plots ---------------------------------------------------------

ggplot(data_final_NN) + 
  #geom_histogram(aes(x = abs(residuals), weight = abs(residuals)), color = "black", fill = "red2", alpha = 0.6) + 
  geom_histogram(aes(x = abs(residuals), y = cumsum(..count..)/sum(..count..)*100, weight = abs(residuals)), 
                 color = "black", fill = "red2", alpha = 0.6)

test_residuals <- data_final_NN %>% filter(label == "test") %>% pull(residuals) %>% abs()
# sum(test_residuals[test_residuals > 25])/sum(test_residuals)*100

ggplot(data_final_NN %>% mutate(year = lubridate::year(date)) %>% filter(year >= lubridate::year("2016-01-01"))) +
  geom_point(aes(x = date, y = rougher.output.recovery), color = "blue2") +
  geom_line(aes(x = date, y = rougher.output.recovery), color = "blue2") +
  geom_point(aes(x = date, y = rougher_recovery_predictions), color = "red2") +
  geom_line(aes(x = date, y = rougher_recovery_predictions), color = "red2") +
  scale_x_datetime(labels=date_format ("%m-%y")) +
  theme_bw() +
  facet_wrap(.~year, scales = "free")

ggplot(data_final_NN) + 
  geom_point(aes(x = rougher.output.recovery, y = rougher_recovery_predictions, color = label))

# Make Predictions --------------------------------------------------------

input_cols <- names(data_rougher)[names(data_rougher) %>% str_detect("input")]
state_cols <- names(data_rougher)[names(data_rougher) %>% str_detect("state")]

test_set <- read_csv("./data/test_data/all_test.csv")
all_data <- bind_rows(data_rougher %>% 
                        select(date, input_cols, state_cols), 
                      test_set %>% 
                        select(date, input_cols, state_cols)) %>% arrange(date)

# Ensure the transformations below are the exact same as above (same variables)

predictors <- setdiff(colnames(data_NN), output_cols) 

all_data <- all_data %>% 
  select(date, input_cols, state_cols) %>% 
  mutate_at(.vars = c(input_cols, state_cols),
            .funs = funs_(lag_functions)) %>% 
  select(date, predictors)

test_data <- all_data %>% filter(date %in% test_set$date) %>% arrange(date)
test_matrix <- data.matrix(test_data %>% select(-date))
test_matrix <- scale(test_matrix, center = mean.nn[!(names(mean.nn) %in% output_cols)], scale = std[!(names(mean.nn) %in% output_cols)])
test_data <- test_data %>% 
  mutate(predictions = predict(model_2, test_matrix)*std[output_cols] + mean.nn[output_cols]) %>% 
  mutate(predictions = predictions/(rougher.input.feed_rate*(rougher.input.feed_zn/100))*100) %>% 
  mutate(predictions = ifelse(predictions < 0, 0, ifelse(predictions > 100, 100, predictions))) #%>% 
  #mutate(predictions = ifelse(rougher.input.feed_zn < 0.05, 100, predictions))
#mutate(predictions = ifelse(rougher.input.feed_zn < 0.05, 100, predictions))
results <- test_data$predictions
results %>% hist(col = "red2")
test_set <- test_set %>% mutate(rougher.output.recovery = results) %>% select(date, rougher.output.recovery)
write_rds(test_set, "./submissions/feb-19/test_set_rougher_recovery_der.rds")

