# Load Libraries ----------------------------------------------------------

library(tidyverse)
library(h2o)
library(zoo)
library(rjson)

# Load Data ---------------------------------------------------------------

dictionary <- fromJSON(file = "./data/data_dictionary_v1.json")
data.all <- read_csv("./data/train_data/all_train.csv")
data.all %>% names

# Predict the whole thing

input_cols <- names(data.all)[names(data.all) %>% str_detect("input|state")]
output_cols <- names(data.all)[names(data.all) %>% str_detect("final.output.recovery")]
data.final <- data.all %>% select(input_cols, output_cols) %>% na.locf()

# H2O Models --------------------------------------------------------------

h2o.removeAll()
h2o.init(nthreads = 6)

train_idx <- sample(1:nrow(data.final), 0.95*nrow(data.final)) # Take 75% of the data as training
df <- data.final %>% filter(output_cols > 0)
df.train <- data.final[train_idx, ] %>% as.h2o()
df.test <- data.final[-train_idx, ] %>% as.h2o()

response <- "final.output.recovery"
predictors <- setdiff(names(df), response)

# Run AutoML for 20 base models (limited to 1 hour max runtime by default)

aml.final <- h2o.automl(x = predictors, 
                        y = response,
                        training_frame = df.train,
                        max_models = 20,
                        seed = 1)

# View the AutoML Leaderboard

lb <- aml.final@leaderboard
print(lb, n = nrow(lb))  # Print all rows instead of default (6 rows)
aml.final@leader

# Preliminary Predictions -------------------------------------------------

predictions <- h2o.predict(aml.final, df %>% as.h2o()) %>% as.data.frame()
total_recovery_predictions <- predictions$predict

data.final <- data.final %>% 
  mutate(total_recovery_predictions = ifelse(total_recovery_predictions < 0, 0, ifelse(total_recovery_predictions > 100, 95, total_recovery_predictions)))

MASE.test <- function(){
  
  # Numerator --------------
  
  st.error <- data.final %>% 
    select(final.output.recovery, total_recovery_predictions) %>% 
    mutate(residuals = final.output.recovery - total_recovery_predictions) %>% 
    pull(residuals) %>% 
    abs() %>% 
    sum()
  
  # Denominator
  
  idx <- 2:nrow(data.final)
  naive.error <- data.final[idx, "final.output.recovery"] - data.final[idx-1,"final.output.recovery"]
  naive.error <- sum(abs(naive.error))
  
  # Final value
  T <- nrow(data.final)
  MASE <- st.error/(T/(T-1)*naive.error)
  return(MASE)
}

MASE.test()

# Some Plots --------------------------------------------------------------

plot_desire <- TRUE

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
  
  ggplot(data.final) + 
    geom_point(aes(x = final.output.recovery, y = total_recovery_predictions, 
                   color = abs(final.output.recovery - total_recovery_predictions)), size = 2) + 
    geom_abline(slope = 1, intercept = 0, size = 1.2, color = "green2", linetype = 2) + 
    scale_color_gradient(low = "green4", high = "red3", guide = FALSE) + 
    theme_bw() 
}

# Save Model --------------------------------------------------------------

# save the model
# model_path <- h2o.saveModel(object = aml.final, path="./cache/", force=TRUE)
# 
# print(model_path)
# 
# # load the model
# 
# aml.final <- h2o.loadModel(model_path)

# Make Predictions --------------------------------------------------------

# Make Predictions on Final Test Set and Save -----------------------------

test_set <- read_csv("./data/test_data/all_test.csv")
test_data <- test_set %>% select(predictors)
predictions <- h2o.predict(aml.final, test_data %>% as.h2o()) %>% as.data.frame()
results <- ifelse(predictions$predict < 0, 0, ifelse(predictions$predict > 100, 90, predictions$predict))
results %>% hist(col = "red2")
# Combine test_set
test_set <- test_set %>% mutate(final.output.recovery = results) %>% select(date, final.output.recovery)
write_rds(test_set, "./submissions/feb-16/test_set_final_recovery.rds")
