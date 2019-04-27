# Create a filteration function to improve the associated filters with the rougher/primary and secondary data.

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

# Analyze Filtration ---------------------------------------------------------

data.train <- read_csv("./data/train_data/all_train.csv") %>% mutate(label = "train")
data.test <- read_csv("./data/test_data/all_test.csv") %>% mutate(label = "test")
data.train <- data.train %>% select(names(data.test), rougher.output.recovery, final.output.recovery)

# Rougher

data.rougher <- data.test %>% select(date, contains("rougher")) %>% melt(id.vars = c("date"))
ggplot(data.rougher) + 
  geom_histogram(aes(x = value, fill = variable), color = "black") + 
  facet_wrap(variable~., scales = "free") + 
  theme(legend.position = "bottom")

data.train.filt <- data.train %>% 
  filter(
    rougher.input.feed_fe > 5,                  
    rougher.input.feed_pb > 0.5,                   
    rougher.input.feed_rate > 100,                 
    rougher.input.feed_size > 25, rougher.input.feed_size < 100,                         
    rougher.input.feed_sol > 20,                  
    rougher.input.feed_zn > 2.5,                 
    rougher.input.floatbank10_copper_sulfate > 2, rougher.input.floatbank10_copper_sulfate < 22,
    rougher.input.floatbank10_xanthate > 2, rougher.input.floatbank10_xanthate < 10,        
    rougher.input.floatbank11_copper_sulfate > 1, rougher.input.floatbank11_copper_sulfate < 21,
    rougher.input.floatbank11_xanthate > 1
    )

data.rougher <- data.train.filt %>% select(date, contains("rougher")) %>% melt(id.vars = c("date", "rougher.output.recovery"))
ggplot(data.rougher) + 
  geom_histogram(aes(x = value, fill = variable), color = "black") + 
  facet_wrap(variable~., scales = "free") + 
  theme(legend.position = "bottom")

ggplot(data.rougher) + 
  geom_point(aes(x = value, y = rougher.output.recovery, fill = variable), pch = 21, color = "black") + 
  facet_wrap(variable~., scales = "free") + 
  theme(legend.position = "none")

# rougher.state.floatbank10_a_air         
# rougher.state.floatbank10_a_level       
# rougher.state.floatbank10_b_air         
# rougher.state.floatbank10_b_level       
# rougher.state.floatbank10_c_air         
# rougher.state.floatbank10_c_level       
# rougher.state.floatbank10_d_air         
# rougher.state.floatbank10_d_level       
# rougher.state.floatbank10_e_air         
# rougher.state.floatbank10_e_level       
# rougher.state.floatbank10_f_air         
# rougher.state.floatbank10_f_level  

# Primary Filters ---------------------------------------------------------

data.primary <- data.train %>% select(date, contains("primary")) %>% melt(id.vars = c("date"))
ggplot(data.primary) + 
  geom_histogram(aes(x = value, fill = variable), color = "black") + 
  facet_wrap(variable~., scales = "free") + 
  theme(legend.position = "bottom") + 
  ggtitle("PRIMARY VARIABLES")

data.primary <- data.train %>% 
  filter(
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

ggplot(data.primary %>% select(date, contains("primary")) %>% melt(id.vars = c("date"))) + 
  geom_histogram(aes(x = value, fill = variable), color = "black") + 
  facet_wrap(variable~., scales = "free") + 
  theme(legend.position = "bottom") + 
  ggtitle("PRIMARY VARIABLES")

# Secondary Filters -------------------------------------------------------

data.sec <- data.train %>% select(date, contains("secondary")) %>% melt(id.vars = c("date"))

ggplot(data.sec) + 
  geom_histogram(aes(x = value, fill = variable), color = "black") + 
  facet_wrap(variable~., scales = "free") + 
  theme(legend.position = "bottom") + 
  ggtitle("Secondary VARIABLES")

data.sec <- data.train %>% 
  filter(
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

data.sec <- data.sec %>% select(date, contains("secondary")) %>% melt(id.vars = c("date"))

ggplot(data.sec) + 
  geom_histogram(aes(x = value, fill = variable), color = "black") + 
  facet_wrap(variable~., scales = "free") + 
  theme(legend.position = "bottom") + 
  ggtitle("Secondary VARIABLES")

# FINAL FUNCTION ----------------------------------------------------------

# CANNOT current account for averaged variables

filter_data_manual <- function(df, primary = FALSE, secondary = FALSE){
  
  # Rougher Filters
  
  df <- filter(df,
               rougher.input.feed_fe > 5,                  
               rougher.input.feed_pb > 0.5,                   
               rougher.input.feed_rate > 100,                 
               rougher.input.feed_size > 25, rougher.input.feed_size < 100,                         
               rougher.input.feed_sol > 20,                  
               rougher.input.feed_zn > 2.5,                 
               rougher.input.floatbank10_copper_sulfate > 2, rougher.input.floatbank10_copper_sulfate < 22,
               rougher.input.floatbank10_xanthate > 2, rougher.input.floatbank10_xanthate < 10,        
               rougher.input.floatbank11_copper_sulfate > 1, rougher.input.floatbank11_copper_sulfate < 21,
               rougher.input.floatbank11_xanthate > 1)
  
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
