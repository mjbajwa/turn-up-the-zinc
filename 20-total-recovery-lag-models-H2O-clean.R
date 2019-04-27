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

lags_rougher <- 1:3
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
rougher_input_state_columns_orig <- rougher_columns[rougher_columns %>% str_detect("input|state")]

# Primary Variables -------------------------------------------------------

primary_columns <- names(data.all)[names(data.all) %>% str_detect("primary")]
primary_input_state_columns_orig <- primary_columns[primary_columns %>% str_detect("input|state")]
#primary_input_state_columns <- primary_input_state_columns[!(primary_input_state_columns %>% str_detect("level"))]

# Secondary Variables -----------------------------------------------------

secondary_columns <- names(data.all)[names(data.all) %>% str_detect("secondary")]
secondary_input_state_columns_orig <- secondary_columns[secondary_columns %>% str_detect("input|state")]
#secondary_input_state_columns <- secondary_input_state_columns[!(secondary_input_state_columns %>% str_detect("level"))]

output_cols <- "final.output.recovery"

# Averaging Action  -------------------------------------------------------

primary_vars_to_remove <- data.all %>% select(contains("primary_cleaner.state.floatbank8")) %>% select(matches("level|air")) %>% names()

lagged.df <- data.all %>%
  
  select(
    date,
    rougher_input_state_columns_orig,
    primary_input_state_columns_orig,
    secondary_input_state_columns_orig,
    final.output.concentrate_zn,
    output_cols
  ) %>% 
  
  # Create Rougher Level Average
  
  mutate(
    rougher_average_level.state.b_f = rowMeans(.[grep("rougher.state.floatbank10_b_level|
                                                      rougher.state.floatbank10_d_level|
                                                      rougher.state.floatbank10_e_level|
                                                      rougher.state.floatbank10_f_level", names(.))]),
    rougher_average_air.state.b_f = rowMeans(.[grep("rougher.state.floatbank10_b_air|
                                                    rougher.state.floatbank10_d_air|
                                                    rougher.state.floatbank10_e_air|
                                                    rougher.state.floatbank10_f_air", names(.))])
    ) %>% 
  
  select(
    -c(rougher.state.floatbank10_b_level,
       rougher.state.floatbank10_d_level,
       rougher.state.floatbank10_e_level,
       rougher.state.floatbank10_f_level, 
       rougher.state.floatbank10_b_air,
       rougher.state.floatbank10_d_air,
       rougher.state.floatbank10_e_air,
       rougher.state.floatbank10_f_air)
  ) %>% 
  
  # Create Primary Level Average
  
  mutate(
    primary_average_level.state = rowMeans(data.all %>% select(contains("primary_cleaner.state.floatbank8")) %>% select(contains("level"))),
    primary_average_air.state = rowMeans(data.all %>% select(contains("primary_cleaner.state.floatbank8")) %>% select(contains("air")))
  ) %>% 
  
  select(-primary_vars_to_remove)

# Redefine columns for lag purposes

rougher_input_state_columns <- lagged.df %>% select(contains("rougher")) %>% names
primary_input_state_columns <- lagged.df %>% select(contains("primary")) %>% names
secondary_input_state_columns <- lagged.df %>% select(contains("secondary")) %>% names

# Create Lags and Filter Data ---------------------------------------------

apply_lags <- TRUE

if(apply_lags) {
  
  lagged.df <- lagged.df %>%
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
    filter(rougher.input.feed_zn > 1,
           rougher.input.feed_rate > 10, 
           final.output.concentrate_zn > 20, 
           final.output.recovery > 40, 
           final.output.recovery < 90) %>%
    na.omit() %>%
    mutate(row_index = 1:n())
  
  fix_lags <- FALSE
  
  if (fix_lags)
  {
    lagged.df <- lagged.df %>%
      select(-(paste0(secondary_input_state_columns, "_lag_", c(rep(1, length(secondary_input_state_columns))))))
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
    filter(rougher.input.feed_zn > 2,
           rougher.input.feed_rate > 200, 
           final.output.concentrate_zn > 20, 
           final.output.recovery > 50, 
           final.output.recovery < 80) %>%
    na.omit() %>%
    mutate(row_index = 1:n())
}

# Plot Key Variables ------------------------------------------------------

#lagged.df.plot <- lagged.df %>% select(rougher.input.feed_rate, rougher.input.feed_zn, final.output.recovery, final.output.concentrate_zn)
#GGally::ggpairs(lagged.df.plot)

# Remove artificial variables

lagged.df$final.output.concentrate_zn <- NULL
lagged.df$final.output.recovery %>% hist(col = "red2")

# Fit a Keras Model -------------------------------------------------------

input_cols <- names(lagged.df)[names(lagged.df) %>% str_detect("input|state")]
output_cols <- output_cols

# Set up Auto-ML ----------------------------------------------------------

h2o.removeAll()
h2o.init(nthreads = 7)

# train_idx <- sample(1:nrow(lagged.df), 0.90*nrow(lagged.df)) # Take 75% of the data as training

train_idx <- sample(1:nrow(lagged.df), 0.90*nrow(lagged.df)) 

df <- lagged.df %>% select(-c(date, row_index))
df.train <- df[train_idx, ] %>% as.h2o()
df.test <- df[-train_idx, ] %>% as.h2o()

response <- output_cols
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
                        seed = 100)

# View the AutoML Leaderboard

lb <- aml.final@leaderboard
print(lb, n = nrow(lb))  # Print all rows instead of default (6 rows)
aml.final@leader

predictions <- h2o.predict(aml.final@leader, df %>% as.h2o()) %>% as.data.frame()
total_recovery_predictions <- predictions$predict

# data.final <- data.final %>% 
#   mutate(total_recovery_predictions = ifelse(total_recovery_predictions < 0, 0, ifelse(total_recovery_predictions > 100, 95, total_recovery_predictions)))

# Make Predictions --------------------------------------------------------

all_inputs <- c(rougher_input_state_columns_orig, primary_input_state_columns_orig, secondary_input_state_columns_orig)
test_set <- read_csv("./data/test_data/all_test.csv")
all_data <- bind_rows(data.all %>% 
                        select(date, all_inputs), 
                      test_set %>% 
                        select(date, all_inputs)) %>% arrange(date)

# Ensure the transformations below are the exact same as above (same variables)

predictors <- setdiff(colnames(df), output_cols) 

# Apply Lags to the Data

all_data_new <- all_data %>% 
  
  select(date, all_inputs) %>% 
  
  # Make Changes
  
  # Create Rougher Level Average
  
  mutate(
    rougher_average_level.state.b_f = rowMeans(.[grep("rougher.state.floatbank10_b_level|
                                                      rougher.state.floatbank10_d_level|
                                                      rougher.state.floatbank10_e_level|
                                                      rougher.state.floatbank10_f_level", names(.))]),
    rougher_average_air.state.b_f = rowMeans(.[grep("rougher.state.floatbank10_b_air|
                                                    rougher.state.floatbank10_d_air|
                                                    rougher.state.floatbank10_e_air|
                                                    rougher.state.floatbank10_f_air", names(.))])
    ) %>% 
  
  select(
    -c(rougher.state.floatbank10_b_level,
       rougher.state.floatbank10_d_level,
       rougher.state.floatbank10_e_level,
       rougher.state.floatbank10_f_level, 
       rougher.state.floatbank10_b_air,
       rougher.state.floatbank10_d_air,
       rougher.state.floatbank10_e_air,
       rougher.state.floatbank10_f_air)
  ) %>% 
  
  # Create Primary Level Average
  
  mutate(
    primary_average_level.state = rowMeans(all_data %>% select(contains("primary_cleaner.state.floatbank8")) %>% select(contains("level"))),
    primary_average_air.state = rowMeans(all_data %>% select(contains("primary_cleaner.state.floatbank8")) %>% select(contains("air")))
  ) %>% 
  
  select(-primary_vars_to_remove) %>% 
  
  # Apply Lags
  
  mutate_at(.vars = c(rougher_input_state_columns),
            .funs = funs_(lag_functions_rougher_)) %>% 
  mutate_at(.vars = c(primary_input_state_columns), 
            .funs = funs_(lag_functions_primary_)) %>% 
  mutate_at(.vars = c(secondary_input_state_columns), 
            .funs = funs_(lag_functions_secondary_)) %>% 
  select(date, predictors)

test_data <- all_data_new %>% filter(date %in% test_set$date) %>% arrange(date)
predictions <- h2o.predict(aml.final@leader, test_data %>% select(-date) %>% as.h2o()) %>% as.data.frame()
test_data <- test_data %>% 
  mutate(predictions = predictions$predict) %>% 
  mutate(predictions = ifelse(predictions < 0, 0, ifelse(predictions > 100, 100, predictions))) #%>% 
  #mutate(predictions = ifelse(rougher.input.feed_zn < 0.02, 100, predictions))
results <- test_data$predictions
results %>% hist(col = "red2")
test_set <- test_set %>% mutate(final.output.recovery = results) %>% select(date, final.output.recovery)
write_rds(test_set, "./submissions/feb-25/test_set_final_recovery.rds")
