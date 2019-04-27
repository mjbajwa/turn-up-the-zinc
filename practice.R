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
data <- read_csv("./data/train_data/all_train.csv")
data %>% names
data_filt <- data %>% select(date, primary_cleaner.state.floatbank8_a_air)
data.ts <- xts(x = data_filt, order.by = data_filt$date)

dygraph(data.ts)

# Attempt to Predict Rougher Process for now ------------------------------

rel_columns <- names(data)[names(data) %>% str_detect("rougher")]
data_rougher <- data %>% select(date, rel_columns) %>% na.locf()
saveRDS(data_rougher, "./cache/data_rougher.rds")

del_columns <- names(data)[names(data) %>% str_detect("calculation")]
data_rougher <- data_rougher %>% select(-del_columns)

data_rougher %>%
  select(everything()) %>%  # replace to your needs
  summarise_all(funs(sum(is.na(.)))) %>% sum()

ggplot(data_rougher, aes(x = date, y = rougher.state.floatbank10_e_level)) + 
  geom_point() + 
  theme_bw()

data_rougher$rougher.output.recovery %>% hist(col = "red2")

# Simple Deep Feedforward Neural Network

data_NN <- data.matrix(data_rougher %>% select(-c(date)))
train_idx <- sample(1:nrow(data_NN), 0.75*nrow(data_NN)) # Take 75% of the data as training
train_data <- data_NN[train_idx,]
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
data_NN <- scale(data_NN, center = mean, scale = std)

# Define Train and Test

data_train <- data_NN[train_idx,]
data_test <- data_NN[-train_idx,]

# Change Rougher Table

data_rougher <- data_rougher %>% mutate(label = ifelse(row_number() %in% train_idx, "train", "test"))

# Create simple dense network 

input_cols <- names(data_rougher)[names(data_rougher) %>% str_detect("input|state")]
output_cols <- names(data_rougher)[names(data_rougher) %>% str_detect("recovery")] # rougher recovery only

model <- keras_model_sequential() %>% 
  layer_dense(units = input_cols %>% length(), activation = 'relu', input_shape = c(input_cols %>% length())) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_dense(units = 50, activation = 'relu') %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 22, activation = 'relu') %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = output_cols %>% length())

model %>% compile(
  optimizer = optimizer_adam(lr = 0.01),
  loss = "mse",
  metrics = c('mae')
)

history <- model %>% fit(
  x = data_train[,input_cols],
  y = data_train[,output_cols],
  validation_split = 0.20,
  epochs = 100,
  batch_size = 128
)

print(history$metrics$val_mean_absolute_error %>% min() %>% `*`(std[output_cols]))

# Predict with the model

data_rougher_2 <- data_rougher %>% 
  mutate(rougher_recovery_predictions = model %>% predict(data_NN[,input_cols])*std[output_cols] + mean[output_cols])

ggplot(data_rougher_2 %>% mutate(year = lubridate::year(date)) %>% filter(year < lubridate::year("2017-01-01"))) + 
  geom_point(aes(x = date, y = rougher.output.recovery), color = "blue2") +
  geom_line(aes(x = date, y = rougher.output.recovery), color = "blue2") +
  geom_point(aes(x = date, y = rougher_recovery_predictions), color = "red2") +
  geom_line(aes(x = date, y = rougher_recovery_predictions), color = "red2") + 
  scale_x_datetime(labels=date_format ("%m-%y")) + 
  facet_grid(year~., scales = "free") + 
  theme_bw()

ggplot(data_rougher_2 %>% mutate(year = lubridate::year(date)) %>% filter(year < lubridate::year("2018-01-01"))) + 
  #geom_point(aes(x = date, y = rougher.output.recovery), color = "blue2") +
  geom_line(aes(x = date, y = rougher.output.recovery), color = "blue2") +
  #geom_point(aes(x = date, y = rougher_recovery_predictions), color = "red2") +
  geom_line(aes(x = date, y = rougher_recovery_predictions), color = "red2") 

ggplot(data_rougher_2 %>% mutate(year = lubridate::year(date))) + 
  geom_point(aes(x = rougher.output.recovery, y = rougher_recovery_predictions, 
                 color = abs(rougher.output.recovery - rougher_recovery_predictions)), size = 2) + 
  geom_abline(slope = 1, intercept = 0, size = 1.2, color = "green2", linetype = 2) + 
  facet_grid(.~label) +
  scale_color_gradient(low = "green4", high = "red3", guide = FALSE) + 
  theme_bw() 


# Other Random Plots ------------------------------------------------------

m <- data_rougher %>% select(input_cols) %>% cor()
ggcorrplot(m, hc.order = "TRUE", method = "circle", type = "upper", tl.cex = 6)

ggplot(data_rougher_2) + 
  geom_point(aes(x = rougher.input.feed_rate, y = rougher_recovery_predictions))
