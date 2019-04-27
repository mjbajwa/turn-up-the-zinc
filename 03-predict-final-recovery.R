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

# Predict the whole thing

input_cols <- names(data.all)[names(data.all) %>% str_detect("input|state")]
output_cols <- names(data.all)[names(data.all) %>% str_detect("final.output.recovery")]
data.final <- data.all %>% select(date, input_cols, output_cols) %>% na.locf()

# data_rougher <- data_rougher %>% 
#   na.locf() %>% 
#   mutate(average_level = rowMeans(.[grep("level", names(.))]),
#          average_air = rowMeans(.[grep("level", names(.))]))
# del_columns_2 <- names(data.all)[names(data) %>% str_detect("rougher.state")]
# data_rougher <- data_rougher %>% 
#   select(-del_columns_2) %>% 
#   filter(final.output.recovery > 40)

# Simple Deep Feedforward Neural Network -------------------------------------------------

data_NN <- data.matrix(data.final %>% select(-c(date)))
train_idx <- sample(1:nrow(data_NN), 0.95*nrow(data_NN)) # Take 75% of the data as training
# train_idx <- seq(1,data.final %>% filter(date < as.Date("2018-01-01")) %>% nrow())
train_data <- data_NN[train_idx,]
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
data_NN <- scale(data_NN, center = mean, scale = std)

# Define Train and Test

data_train <- data_NN[train_idx,]
data_test <- data_NN[-train_idx,]

# Change Rougher Table

data.final <- data.final %>% mutate(label = ifelse(row_number() %in% train_idx, "train", "test"))

# Create Model ------------------------------------------------------------

model <- keras_model_sequential() %>% 
  layer_dense(units = input_cols %>% length(), activation = 'relu', input_shape = c(input_cols %>% length())) %>% 
  layer_dropout(rate = 0.10) %>% 
  layer_dense(units = 50, activation = 'relu') %>%
  layer_dropout(rate = 0.10) %>%
  layer_dense(units = 50, activation = 'relu') %>%
  layer_dropout(rate = 0.10) %>%
  layer_dense(units = output_cols %>% length())

model %>% compile(
  optimizer = optimizer_adam(lr = 0.001),
  loss = "mse",
  metrics = c('mae')
)

callbacks <- list(callback_model_checkpoint(filepath = "./cache/weights.hdf5", save_best_only = TRUE))

history <- model %>% fit(
  x = data_train[,input_cols],
  y = data_train[,output_cols],
  validation_split = 0.10,
  epochs = 500,
  batch_size = 128, 
  callbacks = callbacks
)

print(history$metrics$val_mean_absolute_error %>% min() %>% `*`(std[output_cols]))

model$load_weights <- "./cache/weights.hdf5"

# Predict with the model

data.final <- data.final %>% 
  mutate(total_recovery_predictions = model %>% predict(data_NN[,input_cols])*std[output_cols] + mean[output_cols]) %>% 
  mutate(total_recovery_predictions = ifelse(total_recovery_predictions < 0, 0, ifelse(total_recovery_predictions > 100, 95, total_recovery_predictions)))

plot_desire <- FALSE

if(plot_desire){
  ggplot(data.final %>% mutate(year = lubridate::year(date)) %>% filter(year >= lubridate::year("2018-01-01"))) + 
    geom_point(aes(x = date, y = final.output.recovery), color = "blue2") +
    geom_line(aes(x = date, y = final.output.recovery), color = "blue2") +
    geom_point(aes(x = date, y = total_recovery_predictions), color = "red2") +
    geom_line(aes(x = date, y = total_recovery_predictions), color = "red2") + 
    scale_x_datetime(labels=date_format ("%m-%y")) + 
    theme_bw()
  
  ggplot(data.final %>% mutate(year = lubridate::year(date)) %>% filter(year < lubridate::year("2018-01-01"))) + 
    #geom_point(aes(x = date, y = final.output.recovery), color = "blue2") +
    geom_line(aes(x = date, y = final.output.recovery), color = "blue2") +
    #geom_point(aes(x = date, y = total_recovery_predictions), color = "red2") +
    geom_line(aes(x = date, y = total_recovery_predictions), color = "red2") 
  
  ggplot(data.final %>% mutate(year = lubridate::year(date))) + 
    geom_point(aes(x = final.output.recovery, y = total_recovery_predictions, 
                   color = abs(final.output.recovery - total_recovery_predictions)), size = 2) + 
    geom_abline(slope = 1, intercept = 0, size = 1.2, color = "green2", linetype = 2) + 
    facet_grid(.~label) +
    scale_color_gradient(low = "green4", high = "red3", guide = FALSE) + 
    theme_bw() 
}

# Compute MASE of Test Set ------------------------------------------------

train_idx <- train_idx %>% sort()
test_idx <- (1:nrow(data.final))[-train_idx]

MASE.test <- function(){
  
  # Numerator --------------
  
  st.error <- data.final[test_idx,] %>% 
    select(final.output.recovery, total_recovery_predictions) %>% 
    mutate(residuals = final.output.recovery - total_recovery_predictions) %>% 
    pull(residuals) %>% 
    abs() %>% 
    sum()
  
  # Denominator
  
  test_idx_2 <- test_idx[-1]
  naive.error <- data.final[test_idx_2, "final.output.recovery"] - data.final[test_idx_2 - 1, "final.output.recovery"]
  naive.error <- sum(abs(naive.error))
  
  # Final value
  T <- length(test_idx)
  MASE <- st.error/(T/(T-1)*naive.error)
  return(MASE)
}

MASE.test()

data.final <- data.final %>% mutate(res = abs(final.output.recovery - total_recovery_predictions))

ggplot(data.final) + 
  geom_histogram(aes(x = res, y = ..count.., fill = mean(final.output.recovery))) + 
  scale_fill_gradient(low = "red2", high = "green3")

ggplot(data.final) + 
  geom_point(aes(x = final.output.recovery, y = res, color = res)) + 
  scale_color_gradient(high = "red2", low = "green3")

# Make Predictions on Final Test Set and Save -----------------------------

test_set <- read_csv("./data/test_data/all_test.csv")
test_data <- test_set %>% select(input_cols)
test_matrix <- data.matrix(test_data)
test_matrix <- scale(test_matrix, center = mean[!(names(mean) %in% output_cols)], scale = std[!(names(mean) %in% output_cols)])
results <- predict(model, test_matrix)*std[output_cols] + mean[output_cols]
results <- ifelse(results < 0, 0, ifelse(results > 100, 100, results))
results %>% hist(col = "red2")
# Combine test_set

test_set <- test_set %>% mutate(final.output.recovery = results) %>% select(date, final.output.recovery)
write_rds(test_set, "./submissions/feb-15/test_set_final_recovery_NN.rds")
