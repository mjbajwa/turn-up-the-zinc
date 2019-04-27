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

# Load Data ---------------------------------------------------------------

dictionary <- fromJSON(file = "./data/data_dictionary_v1.json")
data.all <- read_csv("./data/train_data/all_train.csv")
data.all %>% names
data_filt <- data.all %>% select(date, primary_cleaner.state.floatbank8_a_air)

# Attempt to Predict Rougher Process for now ------------------------------

rel_columns <- names(data.all)[names(data.all) %>% str_detect("rougher")]
data_rougher <- data.all %>% select(date, rel_columns) %>% na.omit()
input_cols <- names(data_rougher)[names(data_rougher) %>% str_detect("input|state")]
output_cols <- names(data_rougher)[names(data_rougher) %>% str_detect("rougher.output.recovery")] # rougher recovery only
data_rougher <- data_rougher %>% 
  select(date, input_cols, output_cols) %>% 
  na.locf()

#saveRDS(data_rougher, "./cache/data_rougher.rds")
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
  
id <- FALSE
if(id){
  data_rougher %>%
    select(everything()) %>%  # replace to your needs
    summarise_all(funs(sum(is.na(.)))) %>% sum()
  
  ggplot(data_rougher, aes(x = date, y = rougher.state.floatbank10_e_level)) + 
    geom_point() + 
    theme_bw()
  
  ggplot(data_rougher, 
         aes(x = rougher.input.feed_rate, y = rougher.output.recovery)) + 
    geom_point() + 
    theme_bw()
  
  data_rougher$rougher.output.recovery %>% hist(col = "red2")
}

# cache the data_rougher table

# saveRDS(data_rougher, file = "./cache/data_rougher.rds")

input_cols <- names(data_rougher)[names(data_rougher) %>% str_detect("input|state")]
output_cols <- names(data_rougher)[names(data_rougher) %>% str_detect("rougher.output.recovery")]

# Simple Deep Feedforward Neural Network -------------------------------------------------

data_NN <- data.matrix(data_rougher %>% select(-c(date)))
train_idx <- sample(1:nrow(data_NN), 0.95*nrow(data_NN)) # Take 75% of the data as training
# train_idx <- seq(1,data_rougher %>% filter(date < as.Date("2018-01-01")) %>% nrow())
train_data <- data_NN[train_idx,]
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
data_NN <- scale(data_NN, center = mean, scale = std)

# data_NN[, "rougher.output.recovery"] %>% hist(col = "red2")
# Define Train and Test

data_train <- data_NN[train_idx,]
data_test <- data_NN[-train_idx,]

# Change Rougher Table

data_rougher <- data_rougher %>% mutate(label = ifelse(row_number() %in% train_idx, "train", "test"))

# Create simple dense network 

model <- keras_model_sequential() %>% 
  layer_dense(units = input_cols %>% length(), activation = 'relu', input_shape = c(input_cols %>% length())) %>% 
  layer_dropout(rate = 0.10) %>% 
  layer_dense(units = 50, activation = 'relu') %>%
  layer_dropout(rate = 0.10) %>%
  layer_dense(units = 50, activation = 'relu') %>%
  layer_dropout(rate = 0.05) %>%
  layer_dense(units = output_cols %>% length())

model %>% compile(
  optimizer = optimizer_adam(lr = 0.001),
  loss = "mse",
  metrics = c('mae')
)

callbacks <- list(callback_model_checkpoint(filepath = "./cache/weights_1.hdf5", save_best_only = TRUE))

history <- model %>% fit(
  x = data_train[,input_cols],
  y = data_train[,output_cols],
  validation_split = 0.10,
  epochs = 300,
  batch_size = 128,
  callbacks = callbacks
)

print(history$metrics$val_mean_absolute_error %>% min() %>% `*`(std[output_cols]))
retrain <- TRUE

# New Model ---------------------------------------------------------------
if(retrain){
  model_2 <- keras_model_sequential() %>% 
    layer_dense(units = input_cols %>% length(), activation = 'relu', input_shape = c(input_cols %>% length())) %>% 
    layer_dropout(rate = 0.10) %>% 
    layer_dense(units = 50, activation = 'relu') %>%
    layer_dropout(rate = 0.10) %>%
    layer_dense(units = 50, activation = 'relu') %>%
    layer_dropout(rate = 0.10) %>%
    layer_dense(units = output_cols %>% length())
  
  model_2 %>% compile(
    optimizer = optimizer_adam(lr = 0.001),
    loss = "mse",
    metrics = c('mae')
  )
  
  callbacks <- list(callback_model_checkpoint(filepath = "./cache/weights_2.hdf5", save_best_only = TRUE))
  
  history_2 <- model_2 %>% fit(
    x = data_NN[,input_cols],
    y = data_NN[,output_cols],
    validation_split = 0.20,
    epochs = 200,
    batch_size = 128, 
    callbacks = callbacks
  )
}
# Predict with the model

data_rougher <- data_rougher %>% 
  mutate(rougher_recovery_predictions = model_2 %>% predict(data_NN[,input_cols])*std[output_cols] + mean[output_cols]) %>% 
  mutate(rougher_recovery_predictions = ifelse(rougher_recovery_predictions < 0, 0, rougher_recovery_predictions))

plot_desire <- TRUE

if(plot_desire){
ggplot(data_rougher %>% mutate(year = lubridate::year(date)) %>% filter(year >= lubridate::year("2018-01-01"))) + 
    geom_point(aes(x = date, y = rougher.output.recovery), color = "blue2") +
    geom_line(aes(x = date, y = rougher.output.recovery), color = "blue2") +
    geom_point(aes(x = date, y = rougher_recovery_predictions), color = "red2") +
    geom_line(aes(x = date, y = rougher_recovery_predictions), color = "red2") + 
    scale_x_datetime(labels=date_format ("%m-%y")) + 
    theme_bw()
  
  ggplot(data_rougher %>% mutate(year = lubridate::year(date)) %>% filter(year < lubridate::year("2018-01-01"))) + 
    #geom_point(aes(x = date, y = rougher.output.recovery), color = "blue2") +
    geom_line(aes(x = date, y = rougher.output.recovery), color = "blue2") +
    #geom_point(aes(x = date, y = rougher_recovery_predictions), color = "red2") +
    geom_line(aes(x = date, y = rougher_recovery_predictions), color = "red2") 
  
  ggplot(data_rougher %>% mutate(year = lubridate::year(date))) + 
    geom_point(aes(x = rougher.output.recovery, y = rougher_recovery_predictions, 
                   color = abs(rougher.output.recovery - rougher_recovery_predictions)), size = 2) + 
    geom_abline(slope = 1, intercept = 0, size = 1.2, color = "green2", linetype = 2) + 
    facet_grid(.~label) +
    scale_color_gradient(low = "green4", high = "red3", guide = FALSE) + 
    theme_bw() 
}

# Compute MASE of Test Set ------------------------------------------------

train_idx <- train_idx %>% sort()
test_idx <- (1:nrow(data_rougher))[-train_idx]

MASE.test <- function(){
  
  # Numerator --------------
  
  st.error <- data_rougher[test_idx,] %>% 
    select(rougher.output.recovery, rougher_recovery_predictions) %>% 
    mutate(residuals = rougher.output.recovery - rougher_recovery_predictions) %>% 
    pull(residuals) %>% 
    abs() %>% 
    sum()
  
  # Denominator
  
  test_idx_2 <- test_idx[-1]
  naive.error <- data_rougher[test_idx_2, "rougher.output.recovery"] - data_rougher[test_idx_2 - 1, "rougher.output.recovery"]
  naive.error <- sum(abs(naive.error))
  
  # Final value
  T <- length(test_idx)
  MASE <- st.error/(T/(T-1)*naive.error)
  return(MASE)
}

MASE.test()

# Correlation of Inputs
# 
# m <- data_rougher %>% select(input_cols) %>% cor()
# ggcorrplot(m, hc.order = "TRUE", method = "circle", type = "upper", tl.cex = 6)

# Make Predictions --------------------------------------------------------

# Make Predictions on Final Test Set and Save -----------------------------

test_set <- read_csv("./data/test_data/all_test.csv")
input_cols <- names(data_rougher)[names(data_rougher) %>% str_detect("input|state")]
output_cols <- names(data_rougher)[names(data_rougher) %>% str_detect("rougher.output.recovery")] # rougher recovery only
test_set <- test_set %>% select(date, input_cols)

if(cols_average){
  test_set <- test_set %>%
    mutate(average_level.state = rowMeans(.[grep("level", names(.))]),
           average_air.state = rowMeans(.[grep("air", names(.))]))
  del_columns_2 <- names(test_set)[names(test_set) %>% str_detect("rougher.state")]
  test_set <- test_set %>% 
    select(-del_columns_2)
}

test_data <- test_set %>% select(input_cols)
test_matrix <- data.matrix(test_data)
test_matrix <- scale(test_matrix, center = mean[!(names(mean) %in% output_cols)], scale = std[!(names(mean) %in% output_cols)])
results <- predict(model_2, test_matrix)*std[output_cols] + mean[output_cols]
results <- ifelse(results < 0, 0, results)
results %>% hist(col = "red2")
# Combine test_set

test_set <- test_set %>% mutate(rougher.output.recovery = results) %>% select(date, rougher.output.recovery)
write_rds(test_set, "./submissions/feb-15/test_set_rougher_recovery_NN.rds")
