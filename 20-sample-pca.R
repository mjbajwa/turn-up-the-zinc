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

# Define Filter Function --------------------------------------------------

filter_data_manual <- function(df, primary = FALSE, secondary = FALSE){
  
  # Rougher Filters
  
  df <- filter(df,
               rougher.input.feed_fe > 5,                  
               rougher.input.feed_pb > 0.5,                   
               rougher.input.feed_rate > 100,                 
               rougher.input.feed_size > 10, rougher.input.feed_size < 100,                         
               rougher.input.feed_sol > 20,                  
               rougher.input.feed_zn > 4,                 
               rougher.input.floatbank10_copper_sulfate > 1, rougher.input.floatbank10_copper_sulfate < 22,
               rougher.input.floatbank10_xanthate > 1, rougher.input.floatbank10_xanthate < 10,        
               rougher.input.floatbank11_copper_sulfate > 1, rougher.input.floatbank11_copper_sulfate < 21,
               rougher.input.floatbank11_xanthate > 1, 
               primary_cleaner.input.feed_size > 1)
  
  # Primary Filters
  
  if(primary == TRUE){
    df <- filter(df, 
                 primary_cleaner.input.copper_sulfate > 50, 
                 primary_cleaner.input.depressant < 20,        
                 primary_cleaner.input.feed_size < 10, primary_cleaner.input.feed_size > 5,        
                 primary_cleaner.input.xanthate < 4,          
                 primary_cleaner.state.floatbank8_a_air > 1000,  primary_cleaner.state.floatbank8_a_air < 2000,
                 #primary_cleaner.state.floatbank8_a_level > -800, primary_cleaner.state.floatbank8_a_level < -400,
                 primary_cleaner.state.floatbank8_b_air > 1000, primary_cleaner.state.floatbank8_a_air < 1800,  
                 #primary_cleaner.state.floatbank8_b_level > -700, primary_cleaner.state.floatbank8_b_level < -300,
                 primary_cleaner.state.floatbank8_c_air > 1100,  
                 #primary_cleaner.state.floatbank8_c_level > -600, primary_cleaner.state.floatbank8_c_level < -400,
                 primary_cleaner.state.floatbank8_d_air > 1000
                 #primary_cleaner.state.floatbank8_d_level > -600, primary_cleaner.state.floatbank8_d_level > -300
    )
  }
  
  # Secondary Filters
  
  if(secondary == TRUE){
    df <- filter(df,
                 secondary_cleaner.state.floatbank2_a_air > 20, secondary_cleaner.state.floatbank2_a_air < 40,
                 secondary_cleaner.state.floatbank2_b_air > 10, secondary_cleaner.state.floatbank2_b_air < 40,
                 secondary_cleaner.state.floatbank3_a_air > 20,  secondary_cleaner.state.floatbank3_a_air < 40,
                 secondary_cleaner.state.floatbank3_b_air > 15, secondary_cleaner.state.floatbank3_b_air < 30,   
                 secondary_cleaner.state.floatbank4_a_air > 5, secondary_cleaner.state.floatbank4_a_air < 30,  
                 secondary_cleaner.state.floatbank4_b_air > 5,   
                 secondary_cleaner.state.floatbank5_a_air > 5, secondary_cleaner.state.floatbank5_a_air < 35,  
                 secondary_cleaner.state.floatbank5_b_air > 5, secondary_cleaner.state.floatbank5_a_air < 25, 
                 secondary_cleaner.state.floatbank6_a_air > 5
    )
  }
  
  return(df)
}

# Load Data ---------------------------------------------------------------

dictionary <- fromJSON(file = "./data/data_dictionary_v1.json")
data.all <- read_csv("./data/train_data/all_train.csv")
data.all %>% names

# Input transforms

# data.all <- data.all %>%
#   mutate(rougher.input.xanthate_average = (rougher.input.floatbank11_xanthate + rougher.input.floatbank10_xanthate)/2,
#          rougher.input.copper_sulfate_average = (rougher.input.floatbank10_copper_sulfate + rougher.input.floatbank11_copper_sulfate)/2) %>%
#   select(-c(rougher.input.floatbank11_xanthate,
#             rougher.input.floatbank10_xanthate,
#             rougher.input.floatbank10_copper_sulfate,
#             rougher.input.floatbank11_copper_sulfate))

# Define Closure -------------------------------------------------

lags_rougher <- seq(1,6,1)
lags_primary <- seq(1,2,1)
lags_secondary <- seq(1,2,1)

lag_functions_rougher_ <- setNames(paste("dplyr::lag(., ", lags_rougher, ")"), 
                                   paste("lag", formatC(lags_rougher, width = nchar(max(lags_rougher)), flag = "0"), sep = "_"))

lag_functions_primary_ <- setNames(paste("dplyr::lag(., ", lags_primary, ")"), 
                                   paste("lag", formatC(lags_primary, width = nchar(max(lags_primary)), flag = "0"), sep = "_"))

lag_functions_secondary_ <- setNames(paste("dplyr::lag(., ", lags_secondary, ")"), 
                                     paste("lag", formatC(lags_secondary, width = nchar(max(lags_secondary)), flag = "0"), sep = "_"))

# Rougher Columns -----------------------------------------------------------

rougher_columns <- names(data.all)[names(data.all) %>% str_detect("primary")]

data_rougher <- data.all %>% select(date, rougher_columns) %>% select(date, contains("air")) %>% na.omit()
pcomp <- princomp(data_rougher %>% select(-date) %>% as.matrix())
pcomp$sdev[[1]]/sum(pcomp$sdev)
pcomp$sdev[[2]]/sum(pcomp$sdev)
pcomp$sdev[[3]]/sum(pcomp$sdev)

scores_df <- data_rougher %>% bind_cols(pcomp$scores %>% data.frame)
data_new <- data.all %>% left_join(scores_df, by = "date")

ggplot(data_new %>% filter(final.output.recovery > 35)) + 
  geom_point(aes(x = Comp.1, y = Comp.2, color = final.output.recovery))
