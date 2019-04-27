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

output_cols <- "final.output.recovery"

# Create Lagged DF --------------------------------------------------------

# Create Lags and Filter Data ---------------------------------------------

lagged.df <- data.all %>% 
  select(date, 
         rougher_input_state_columns, 
         primary_input_state_columns, 
         secondary_input_state_columns, 
         output_cols) %>% 
  na.locf() %>% 
  mutate(row_index = 1:n())


# Create biased training set ----------------------------------------------

# Fit a Keras Model -------------------------------------------------------

input_cols <- names(lagged.df)[names(lagged.df) %>% str_detect("input|state")]
output_cols <- output_cols

# Split the data (lagged.df) to supply a slightly better proportion of low values to the model

data_density_correction <- FALSE

if(data_density_correction){
  
  rows.tails <- lagged.df %>% filter(final.output.recovery < 50) %>% pull(row_index)
  rows.tails <- c(rows.tails, lagged.df %>% filter(final.output.recovery > 80) %>% pull(row_index)) %>% sort()
  rows.medium <- lagged.df %>% filter(!(row_index %in% rows.tails)) %>% pull(row_index) %>% sort()
  
  x_tails <- 0.85
  x_medium <- 0.55
  
  train_idx <- c(sample(rows.tails, x_tails*length(rows.tails)), 
                 sample(rows.medium, x_medium*length(rows.medium))) %>% sort() %>% sample()
  
  tail_density <- x_tails*length(rows.tails)/length(train_idx)
  print(length(train_idx)/nrow(lagged.df))
  
}

# Feed Data to Model ------------------------------------------------------

data_NN <- data.matrix(lagged.df %>% select(-c(date, row_index)))
# train_idx <- sample(1:nrow(data_NN), 0.70*nrow(data_NN)) # Take 75% of the data as training
train_idx <- seq(1,lagged.df %>% filter(date < as.Date("2018-01-01")) %>% nrow()) %>% sample()
train_data <- data_NN[train_idx,]
mean.nn <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
data_NN <- scale(data_NN, center = mean.nn, scale = std)

data <- data_NN

# Define Inputs for Convolution and Recurrent Model ----------------------------------------------------

source("./R/sequential_models/10-generator.R")

lookback <- 3
step <- 1
delay <- 1
batch_size <- 64
train_min <- 1
train_max <- floor(nrow(data)*0.70)
val_min <- train_max + 1
val_max <- floor(nrow(data)*0.80)
test_min <- val_max + 1
test_max <- nrow(data)

train_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = train_min,
  max_index = train_max,
  shuffle = TRUE,
  step = step, 
  batch_size = batch_size
)

val_gen = generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = val_min,
  max_index = val_max,
  step = step,
  batch_size = batch_size
)

test_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = test_min,
  max_index = test_max,
  step = step,
  batch_size = batch_size
)

# How many steps to draw from val_gen in order to see the entire validation set
val_steps <- (val_max - val_min - lookback) / batch_size

# How many steps to draw from test_gen in order to see the entire test set
test_steps <- (test_max - test_min - lookback) / batch_size -2

# Create a quick plot to see split sections -------------------------------

temp_pred <- lagged.df %>% filter(date < as.Date(paste0(2018,"-01-01")))
temp_pred <- data.table::data.table(temp_pred)

ggplot(temp_pred, 
       aes(x = date, y = final.output.recovery)) + 
  geom_step(size = 1) + 
  theme_bw() +
  annotate("rect", xmin = temp_pred[1, date], xmax = temp_pred[train_max, date], ymin=0, ymax=Inf, alpha=0.1, fill="green") +
  annotate("rect", xmin = temp_pred[val_min, date], xmax = temp_pred[val_max, date], ymin=0, ymax=Inf, alpha=0.1, fill="blue") +
  annotate("rect", xmin = temp_pred[test_min,date], xmax = temp_pred[test_max, date], ymin=0, ymax=Inf, alpha=0.1, fill="red") +
  annotate("segment", x = temp_pred$date[1], xend = temp_pred$date[1+lookback], y = 0, yend = 0, size = 2, color = "black", linetype = 1) +
  annotate("text", x = mean.POSIXct(temp_pred$date[1:lookback]), y = 3, label = "Lookback Length") +
  labs(title = paste0("Training, Validation and Test Split"), x = "Timestamp", y = "Recovery (%)")

# Naive Method ------------------------------------------------------------

evaluate_naive_method <- function() {
  batch_maes <- c()
  for (step in 1:(test_steps-2)) {
    c(samples, targets) %<-% test_gen()
    preds <- samples[,dim(samples)[[2]],2]
    mae <- mean(abs(preds - targets))
    batch_maes <- c(batch_maes, mae)
  }
  print(mean(batch_maes))
}

a <- evaluate_naive_method()
a*(std[output_cols])

# Dense Model

# model <- keras_model_sequential() %>% 
#   layer_flatten(input_shape = c((lookback + 1)/ step, dim(data)[[-1]] - 1)) %>% 
#   layer_dense(units = 64, activation = "relu") %>% 
#   layer_dense(units = 128, activation = "relu") %>% 
#   layer_dense(units = 64, activation = "relu") %>% 
#   layer_dense(units = 1)

# Recurrent model (AR)

# model <- keras_model_sequential() %>%
#   layer_gru(units = 10,
#             dropout = 0.1,
#             recurrent_dropout = 0.5,
#             return_sequences = TRUE,
#             input_shape = list(NULL, dim(data)[-1])) %>%
#   layer_gru(units = 20, activation = "relu",
#             dropout = 0.1,
#             recurrent_dropout = 0.5) %>%
#   layer_dense(units = 1)

model <- keras_model_sequential() %>%
  layer_gru(units = 5,
            dropout = 0.1,
            recurrent_dropout = 0.5,
            return_sequences = TRUE,
            input_shape = list(NULL, dim(data)[-1])) %>%
  layer_gru(units = 10, activation = "relu",
            dropout = 0.1,
            recurrent_dropout = 0.5) %>%
  layer_dense(units = 1)

# Convolutional Model

# model <- keras_model_sequential() %>% 
#   layer_conv_1d(filters = 4, kernel_size = 4, activation = "relu",
#                 input_shape = list(NULL, dim(data)[[-1]])) %>% 
#   layer_max_pooling_1d(pool_size = 1) %>% 
#   layer_conv_1d(filters = 4, kernel_size = 5, activation = "relu") %>% 
#   layer_max_pooling_1d(pool_size = 1) %>% 
#   layer_dense(units = 1)

# Convoluational + Recurrent Model

# model <- keras_model_sequential() %>% 
#   layer_conv_1d(filters = 32, kernel_size = 5, activation = "relu",
#                 input_shape = list(NULL, dim(data)[[-1]])) %>% 
#   layer_max_pooling_1d(pool_size = 3) %>% 
#   layer_conv_1d(filters = 32, kernel_size = 5, activation = "relu") %>% 
#   layer_gru(units = 32, dropout = 0.1, recurrent_dropout = 0.5) %>% 
#   layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_adam(),
  loss = "mse",
  metrics = c('mae')
)

callbacks <- list(callback_model_checkpoint(filepath = "./cache/weights.hdf5", save_best_only = TRUE))

history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 200,
  epochs = 20,
  validation_data = val_gen,
  validation_steps = val_steps, 
  callbacks = callbacks
)

print(history$metrics$val_loss %>% min() %>% `*`(std[output_cols]))

# Make Predictions --------------------------------------------------------

# Predictions WITH DROPOUT (Using Yarin Gal's Methods) --------------------

pred_gen = generator(
  data = data,
  lookback = lookback,
  delay = delay,
  min_index = 1,
  max_index = NULL,
  step = step,
  batch_size = batch_size,
  pred = TRUE
)

N_samples <- 1
# predictions_all <- data.table(matrix(0,nrow = test_max - lookback - delay, ncol = N_samples))
predictions <- model %>% predict_generator(pred_gen, steps = (test_max - delay - lookback)/(batch_size+1) + 1, verbose = T)

data_rougher_2 <- data_rougher %>% 
  filter(date < as.Date(paste0(date_filter,"-01-01"))) %>% 
  mutate(rougher_recovery_predictions = c(rep(NA, lookback + delay), predictions*std[output_cols] + mean[output_cols])) %>% 
  mutate(rougher_recovery_predictions = pmax(rougher_recovery_predictions, 0))

plot_final <- ggplot(data_rougher_2) + 
  geom_point(aes(x = date, y = rougher.output.recovery), color = "blue2") +
  geom_line(aes(x = date, y = rougher.output.recovery), color = "blue2") +
  geom_point(aes(x = date, y = rougher_recovery_predictions), color = "red2") +
  geom_line(aes(x = date, y = rougher_recovery_predictions), color = "red2") + 
  scale_x_datetime(labels=date_format ("%m-%y")) +
  theme_bw() + 
  annotate("rect", xmin = temp_pred[1, date], xmax = temp_pred[train_max, date], ymin=0, ymax=Inf, alpha=0.15, fill="green") +
  annotate("rect", xmin = temp_pred[val_min, date], xmax = temp_pred[val_max, date], ymin=0, ymax=Inf, alpha=0.15, fill="orange") +
  annotate("rect", xmin = temp_pred[test_min,date], xmax = temp_pred[test_max, date], ymin=0, ymax=Inf, alpha=0.15, fill="red") 

plot_final
# ggplot(data_rougher_2 %>% filter(date > as.Date("2018-01-01"))) + 
#   #geom_point(aes(x = date, y = rougher.output.recovery), color = "blue2") +
#   geom_line(aes(x = date, y = rougher.output.recovery), color = "blue2") +
#   #geom_point(aes(x = date, y = rougher_recovery_predictions), color = "red2") +
#   geom_line(aes(x = date, y = rougher_recovery_predictions), color = "red2") 

data_rougher_2 <- data_rougher_2 %>%
  mutate(label = c(
    rep("Train", train_max),
    rep("Val", val_max - train_max),
    rep("Test", test_max - val_max)
  ), 
  naive_prediction = lag(rougher.output.recovery, 1)
  )

ggplot(data_rougher_2) + 
  geom_line(aes(x = date, y = rougher.output.recovery), color = "blue2") +
  geom_line(aes(x = date, y = naive_prediction), color = "red2", alpha = 0.5) + 
  scale_x_datetime(labels=date_format ("%m-%y")) +
  theme_bw() + 
  annotate("rect", xmin = temp_pred[1, date], xmax = temp_pred[train_max, date], ymin=0, ymax=Inf, alpha=0.15, fill="green") +
  annotate("rect", xmin = temp_pred[val_min, date], xmax = temp_pred[val_max, date], ymin=0, ymax=Inf, alpha=0.15, fill="orange") +
  annotate("rect", xmin = temp_pred[test_min,date], xmax = temp_pred[test_max, date], ymin=0, ymax=Inf, alpha=0.15, fill="red") 


ggplot(data_rougher_2 %>% mutate(year = lubridate::year(date))) + 
  geom_point(aes(x = rougher.output.recovery, y = rougher_recovery_predictions, 
                 color = abs(rougher.output.recovery - rougher_recovery_predictions)), size = 2) + 
  geom_abline(slope = 1, intercept = 0, size = 1.2, color = "green2", linetype = 2) + 
  facet_grid(.~label) +
  scale_color_gradient(low = "green4", high = "red3", guide = FALSE) + 
  theme_bw()

ggplot(data_rougher_2) + 
  geom_point(aes(x = rougher.output.recovery, y = naive_prediction)) +
  geom_abline(slope = 1, intercept = 0, size = 1.2, color = "green2", linetype = 2)

ggplot(data_rougher_2) + 
  geom_point(aes(x = rougher.input.feed_rate, y = rougher.output.recovery))

# Understand where the gganimate package can help -------------------------

ggplot(data_rougher_2) + 
  geom_point(aes(x = rougher.input.feed_rate, 
                 y = rougher.output.recovery, 
                 color = rougher.output.recovery)) + 
  transition_time(date)
