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
  select(date, input_cols, state_cols, output_cols, rougher.output.concentrate_zn) #%>% 
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
  select(date, input_cols, state_cols, output_cols, rougher.output.concentrate_zn) %>% 
  mutate_at(.vars = c(input_cols, state_cols),
            .funs = funs_(lag_functions)) %>% 
  filter(
    rougher.input.feed_zn > 0.05,
    rougher.input.feed_rate > 1, 
    rougher.output.recovery > 40) %>%  
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
           row_index) #%>% 
  #filter(get(output_cols) < 50) 
  
  lagged.df %>% names # just checking
  
}

lagged.df$rougher.output.recovery %>% hist(col = "red2")

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

# train_idx <- seq(1,lagged.df %>% filter(date < as.Date("2018-01-01")) %>% nrow()) %>% sample()

# Change Rougher Table

train_idx <- sample(1:nrow(lagged.df), 0.85*nrow(lagged.df)) 
data_rougher_NN <- lagged.df %>% mutate(label = ifelse(row_number() %in% train_idx, "train", "test"))

# Set up Auto-ML ----------------------------------------------------------

h2o.removeAll()
h2o.init(nthreads = 7)

# train_idx <- sample(1:nrow(lagged.df), 0.90*nrow(lagged.df)) # Take 75% of the data as training

data_density_correction <- FALSE

if(data_density_correction){
  
  rows.tails <- lagged.df %>% filter(final.output.recovery < 50) %>% pull(row_index)
  rows.tails <- c(rows.tails, lagged.df %>% filter(final.output.recovery > 80) %>% pull(row_index)) %>% sort()
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

aml.final <- h2o.automl(x = predictors, 
                        y = response,
                        training_frame = df.train,
                        leaderboard_frame = df.test,
                        nfolds = 5,
                        max_models = 20,
                        stopping_metric = "MAE",
                        sort_metric = "MAE",
                        exclude_algos = c("DeepLearning"),
                        seed = 1)

# View the AutoML Leaderboard

lb <- aml.final@leaderboard
print(lb, n = nrow(lb))  # Print all rows instead of default (6 rows)
aml.final@leader

predictions <- h2o.predict(aml.final, df %>% as.h2o()) %>% as.data.frame()
total_recovery_predictions <- predictions$predict

# data.final <- data.final %>% 
#   mutate(total_recovery_predictions = ifelse(total_recovery_predictions < 0, 0, ifelse(total_recovery_predictions > 100, 95, total_recovery_predictions)))

# Make Predictions --------------------------------------------------------

input_cols <- names(data_rougher)[names(data_rougher) %>% str_detect("input")]
state_cols <- names(data_rougher)[names(data_rougher) %>% str_detect("state")]

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
predictions <- h2o.predict(aml.final@leader, test_data %>% select(-date) %>% as.h2o()) %>% as.data.frame()
test_data <- test_data %>% 
  mutate(predictions = ifelse(predictions$predict < 0, 0, ifelse(predictions$predict > 100, 99, predictions$predict))) 
results <- test_data$predictions
results %>% hist(col = "red2")
test_set <- test_set %>% mutate(rougher.output.recovery = results) %>% select(date, rougher.output.recovery)
write_rds(test_set, "./submissions/feb-20/test_set_rougher_recovery.rds")
