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
    filter(rougher.input.feed_zn > 0.5,
           rougher.input.feed_rate > 10, 
           final.output.concentrate_zn > 2) %>%  
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
      rougher.input.feed_rate > 5, 
      final.output.concentrate_zn > 20) %>%
    na.omit() %>%
    mutate(row_index = 1:n())
}

lagged.df$final.output.concentrate_zn <- NULL
lagged.df$final.output.recovery %>% hist(col = "red2")

# Fit a Keras Model -------------------------------------------------------

input_cols <- names(lagged.df)[names(lagged.df) %>% str_detect("input|state")]
output_cols <- output_cols

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

train_idx <- sample(1:nrow(lagged.df), 0.85*nrow(lagged.df)) 

df <- lagged.df %>% select(-c(date, row_index))
df.train <- df[train_idx, ] %>% as.h2o()
df.test <- df[-train_idx, ] %>% as.h2o()

response <- "final.output.recovery"
predictors <- setdiff(names(df), response) # This is used later ON - DO NOT CHANGE - AFFECTS DOWNSTREAM SIGNIFICANTLY

# Run AutoML for 20 base models (limited to 1 hour max runtime by default)

aml.final <- h2o.automl(x = predictors, 
                        y = response,
                        training_frame = df.train,
                        leaderboard_frame = df.test,
                        nfolds = 5,
                        max_models = 10,
                        stopping_metric = "MAE",
                        sort_metric = "MAE",
                        exclude_algos = c("DeepLearning"),
                        seed = 1)

# View the AutoML Leaderboard

lb <- aml.final@leaderboard
print(lb, n = nrow(lb))  # Print all rows instead of default (6 rows)
aml.final@leader

predictions <- h2o.predict(aml.final@leader, df %>% as.h2o()) %>% as.data.frame()
total_recovery_predictions <- predictions$predict

# data.final <- data.final %>% 
#   mutate(total_recovery_predictions = ifelse(total_recovery_predictions < 0, 0, ifelse(total_recovery_predictions > 100, 95, total_recovery_predictions)))

# Make Predictions --------------------------------------------------------

all_inputs <- c(rougher_input_state_columns, primary_input_state_columns, secondary_input_state_columns)

test_set <- read_csv("./data/test_data/all_test.csv")
all_data <- bind_rows(data.all %>% 
                        select(date, all_inputs), 
                      test_set %>% 
                        select(date, all_inputs)) %>% arrange(date)

# Ensure the transformations below are the exact same as above (same variables)

all_data <- all_data %>% 
  select(date, all_inputs) %>% 
  mutate_at(.vars = c(rougher_input_state_columns),
            .funs = funs_(lag_functions_rougher_)) %>% 
  mutate_at(.vars = c(primary_input_state_columns), 
            .funs = funs_(lag_functions_primary_)) %>% 
  mutate_at(.vars = c(secondary_input_state_columns), 
            .funs = funs_(lag_functions_secondary_)) %>% 
  select(date, predictors)

test_data <- all_data %>% filter(date %in% test_set$date) %>% arrange(date)
predictions <- h2o.predict(aml.final@leader, test_data %>% select(-date) %>% as.h2o()) %>% as.data.frame()
test_data <- test_data %>% 
  mutate(predictions = predictions$predict) %>% 
  mutate(predictions = ifelse(predictions < 0, 0, ifelse(predictions > 100, 100, predictions))) %>% 
  mutate(predictions = ifelse(rougher.input.feed_zn < 0.05, 100, predictions))
results <- test_data$predictions
results %>% hist(col = "red2")
test_set <- test_set %>% mutate(final.output.recovery = results) %>% select(date, final.output.recovery)
write_rds(test_set, "./submissions/feb-20/test_set_final_recovery.rds")
