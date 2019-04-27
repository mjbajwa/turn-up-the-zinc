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
library(reshape2)

# Load Data ---------------------------------------------------------------

dictionary <- fromJSON(file = "./data/data_dictionary_v1.json")
data.all <- read_csv("./data/train_data/all_train.csv")
data.all %>% names

# Rougher Process ---------------------------------------------------------

rel_columns <- names(data.all)[names(data.all) %>% str_detect("rougher")]
data_rougher <- data.all %>% select(date, rel_columns, primary_cleaner.input.feed_size) %>% na.locf()
data_rougher %>% names()
input_cols <- names(data_rougher)[names(data_rougher) %>% str_detect("input")]
state_cols <- names(data_rougher)[names(data_rougher) %>% str_detect("state")]
output_cols <- names(data_rougher)[names(data_rougher) %>% str_detect("rougher.output.recovery")]

# Preliminary Plots for inputs

plot_density <- function(columns){
  
  ggplot(data_rougher %>% select(date, columns) %>% melt(id.vars = "date")) + 
    geom_histogram(aes(x = value, y = ..density.., fill = variable), alpha = 0.5, color = "black") + 
    facet_wrap(variable~., scales = "free") + 
    theme_bw() + 
    theme(legend.position = "none") 

}

plot_density(input_cols)
plot_density(state_cols)
plot_density(output_cols)

# Plot Downstream Feedsize and Recovery of Rougher

ggplot(data_rougher %>% select(date, output_cols, primary_cleaner.input.feed_size) %>% melt(id.vars = "date")) + 
  #geom_histogram(aes(x = value, y = ..density.., fill = variable), alpha = 0.5, color = "black") + 
  geom_line(aes(x = date, y = value, color = variable), size = 0.9) + 
  geom_point(aes(x = date, y = value, color = variable), size = 2) +
  facet_wrap(variable~., scales = "free") + 
  theme_bw() + 
  theme(legend.position = "none") 

ggplot(data_rougher %>% select(date, input_cols, output_cols, primary_cleaner.input.feed_size)) + 
  #geom_histogram(aes(x = value, y = ..density.., fill = variable), alpha = 0.5, color = "black") + 
  geom_point(aes(x = rougher.input.feed_rate*rougher.input.feed_zn/100, 
                 y = get(output_cols), 
                 color = get(output_cols)), size = 2) +
  theme_bw() + 
  theme(legend.position = "bottom") + 
  scale_color_gradient(low = "red", high = "green4") + 
  labs(title = "Zinc Input to Rougher (no Lag)", x = "Zinc Input Flow Rate", y = output_cols)

ggplot(data_rougher %>% select(date, input_cols, output_cols, primary_cleaner.input.feed_size)) + 
  #geom_histogram(aes(x = value, y = ..density.., fill = variable), alpha = 0.5, color = "black") + 
  geom_point(aes(x = rougher.input.feed_rate*rougher.input.feed_zn/100, 
                 y = get(output_cols), 
                 color = get(output_cols)), size = 2) +
  theme_bw() + 
  theme(legend.position = "bottom") + 
  scale_color_gradient(low = "red", high = "green4") + 
  labs(title = "Zinc Input to Rougher (no Lag)", x = "Zinc Input Flow Rate", y = output_cols)

# Lagged Plots ------------------------------------------------------------

lags <- seq(5)
lag_names <- paste("lag", formatC(lags, width = nchar(max(lags)), flag = "0"), 
                   sep = "_")

lag_functions <- setNames(paste("dplyr::lag(., ", lags, ")"), lag_names)

lagged.df <- data_rougher %>% 
  select(input_cols, output_cols) %>% 
  mutate_at(.vars = input_cols,
            .funs = funs_(lag_functions))

lagged.df %>% select(contains("rougher.input.feed_fe")) # Check to see if it works.

ggplot(lagged.df) + 
  geom_point(aes(x = rougher.input.feed_rate_lag_2*rougher.input.feed_zn_lag_2/100, 
                 y = get(output_cols), 
                 color = get(output_cols)), size = 2) +
  theme_bw() + 
  theme(legend.position = "bottom") + 
  scale_color_gradient(low = "red", high = "green4") + 
  labs(title = "Zinc Input to Rougher (with Lag)", x = "Zinc Input Flow Rate", y = output_cols)

# Train vs. Test Visualizations -------------------------------------------

data.train <- read_csv("./data/train_data/all_train.csv") %>% mutate(label = "train")
data.test <- read_csv("./data/test_data/all_test.csv") %>% mutate(label = "test")
data.comp <- bind_rows(data.train %>% select(names(data.test)), data.test)
names_filter <- names(data.test)[!(names(data.test) %in% c("date", "label"))]

# Distribution Plots ------------------------------------------------------

ggplot(data.comp %>% select(date, label, contains("rougher")) %>% melt(id.vars = c("date", "label"))) + 
  geom_histogram(aes(x = value, y = ..density.., fill = label, alpha = label), color = "black") + 
  geom_density(aes(x = value, color = label)) + 
  scale_alpha_manual(values = c(0.6, 0.9)) + 
  facet_wrap(variable~., scales = "free")

ggplot(data.comp %>% select(date, label, contains("primary")) %>% melt(id.vars = c("date", "label"))) + 
  geom_histogram(aes(x = value, y = ..count.., fill = label, alpha = label), color = "black") + 
  scale_alpha_manual(values = c(0.6, 0.9)) + 
  facet_wrap(variable~., scales = "free")

ggplot(data.comp %>% select(date, label, contains("secondary")) %>% melt(id.vars = c("date", "label"))) + 
  geom_histogram(aes(x = value, y = ..count.., fill = label, alpha = label), color = "black") + 
  scale_alpha_manual(values = c(0.6, 0.9)) + 
  facet_wrap(variable~., scales = "free")

# Boxplots ----------------------------------------------------------------

normalize <- function(x){
  return((x- mean(x))/(max(x)-min(x)))
}

data.rougher <- data.comp %>%
  select(label, contains("rougher")) %>%
  na.omit() %>%
  group_by(label) %>%
  mutate_all(funs(normalize))

# data.rougher <- data.comp %>% select(label, contains("rougher"))

ggplot(data.rougher %>% melt(id.vars = c("label")) %>% arrange(desc(label))) + 
  geom_boxplot(aes(x = variable, y = value, fill = variable)) + 
  theme(legend.position = "none", axis.text.x = element_text(angle = 90, hjust = 1)) + 
  facet_wrap(label~.) 

data.primary <- data.comp %>% 
  select(label, contains("primary")) %>% 
  na.omit() %>% 
  group_by(label) %>% 
  mutate_all(funs(normalize))

ggplot(data.primary %>% melt(id.vars = c("label")) %>% arrange(desc(label))) + 
  geom_boxplot(aes(x = variable, y = value, fill = variable)) + 
  theme(legend.position = "none", axis.text.x = element_text(angle = 90, hjust = 1)) + 
  facet_wrap(label~.) 

data.secondary <- data.comp %>% 
  select(label, contains("secondary")) %>% 
  na.omit() %>% 
  group_by(label) %>% 
  mutate_all(funs(normalize))

ggplot(data.primary %>% melt(id.vars = c("label")) %>% arrange(desc(label))) + 
  geom_boxplot(aes(x = variable, y = value, fill = variable)) + 
  theme(legend.position = "none", axis.text.x = element_text(angle = 90, hjust = 1)) + 
  facet_wrap(label~.) 

# Analyze Data for Covariate Shifts Over Time -----------------------------

data.train <- read_csv("./data/train_data/all_train.csv") %>% mutate(year = lubridate::year(date))

# Rougher Covariate Shift

data.rougher <- data.train %>% 
  select(year, contains("rougher")) %>% 
  na.omit() %>% 
  mutate_all(funs(normalize))

ggplot(data.rougher %>% select(rougher.input.feed_rate, year, rougher.input.feed_zn, rougher.output.recovery) %>% melt(id.vars = c("year"))) + 
  geom_boxplot(aes(x = variable, y = value, fill = variable)) + 
  theme(legend.position = "none", axis.text.x = element_text(angle = 90, hjust = 1)) + 
  facet_wrap(year~.) 

# Final Output Recovery Shift

data.final <- data.train %>% 
  select(year, contains("primary"), contains("secondary"), final.output.recovery, rougher.output.recovery) %>% 
  na.omit() #%>% 
  
# data.final <- data.final %>% mutate_at(.vars = names(data.final)[!(names(data.final) == "year")], .funs = funs(normalize))

ggplot(data.final %>% select(year, rougher.output.recovery, final.output.recovery) %>% melt(id.vars = c("year"))) + 
  geom_boxplot(aes(x = variable, y = value, fill = variable)) + 
  theme(legend.position = "none", axis.text.x = element_text(angle = 90, hjust = 1)) + 
  facet_wrap(year~.) 

# CORRELATION BETWEEN LEVELS IN TEST DATA ----------------------------------------------------------------------------

data.train <- read_csv("./data/train_data/all_train.csv") %>% mutate(label = "train")
data.test <- read_csv("./data/test_data/all_test.csv") %>% mutate(label = "test")

# Rougher Levels

corr.train <- data.train %>% select(contains("level")) %>% select(contains("rougher")) %>% na.omit() %>% cor() %>% round(1)
ggcorrplot(corr.train, hc.order = TRUE, type = "lower", lab = TRUE)

corr.test <- data.test %>% select(contains("level")) %>% select(contains("rougher")) %>% na.omit() %>% cor() %>% round(1)
ggcorrplot(corr.test, hc.order = TRUE, type = "lower", lab = TRUE)

corr.train <- data.train %>% select(contains("air")) %>% select(contains("rougher")) %>% na.omit() %>% cor() %>% round(1)
ggcorrplot(corr.train, hc.order = TRUE, type = "lower", lab = TRUE)

corr.test <- data.test %>% select(contains("air")) %>% select(contains("rougher")) %>% na.omit() %>% cor() %>% round(1)
ggcorrplot(corr.test, hc.order = TRUE, type = "lower", lab = TRUE)

# Primary Levels and Air

corr.train <- data.train %>% select(contains("level")) %>% select(contains("primary")) %>% na.omit() %>% cor() %>% round(1)
ggcorrplot(corr.train, hc.order = TRUE, type = "lower", lab = TRUE)

corr.test <- data.test %>% select(contains("level")) %>% select(contains("primary")) %>% na.omit() %>% cor() %>% round(1)
ggcorrplot(corr.test, hc.order = TRUE, type = "lower", lab = TRUE)

corr.train <- data.train %>% select(contains("air")) %>% select(contains("primary")) %>% na.omit() %>% cor() %>% round(1)
ggcorrplot(corr.train, hc.order = TRUE, type = "lower", lab = TRUE)

corr.test <- data.test %>% select(contains("air")) %>% select(contains("primary")) %>% na.omit() %>% cor() %>% round(1)
ggcorrplot(corr.test, hc.order = TRUE, type = "lower", lab = TRUE)

# Secondary Levels and Air

corr.train <- data.train %>% select(contains("level")) %>% select(contains("secondary")) %>% na.omit() %>% cor() %>% round(1)
ggcorrplot(corr.train, hc.order = TRUE, type = "lower", lab = TRUE)

corr.test <- data.test %>% select(contains("level")) %>% select(contains("secondary")) %>% na.omit() %>% cor() %>% round(1)
ggcorrplot(corr.test, hc.order = TRUE, type = "lower", lab = TRUE)

corr.train <- data.train %>% select(contains("air")) %>% select(contains("secondary")) %>% na.omit() %>% cor() %>% round(1)
ggcorrplot(corr.train, hc.order = TRUE, type = "lower", lab = TRUE)

corr.test <- data.test %>% select(contains("air")) %>% select(contains("secondary")) %>% na.omit() %>% cor() %>% round(1)
ggcorrplot(corr.test, hc.order = TRUE, type = "lower", lab = TRUE)

# Covariance of Key Variables ---------------------------------------------

ggplot(data.comp %>% filter(rougher.input.feed_rate > 200)) + 
  geom_point(aes(x = rougher.input.floatbank11_xanthate, y = rougher.input.floatbank10_xanthate, color = label))

ggplot(data.comp %>% filter(rougher.input.feed_rate > 200)) + 
  geom_point(aes(x = rougher.input.floatbank10_copper_sulfate, y = rougher.input.floatbank11_copper_sulfate, color = label))

ggpairs(data.comp %>% select(rougher.input.floatbank11_xanthate, rougher.input.floatbank10_xanthate, 
                             rougher.input.floatbank10_copper_sulfate, rougher.input.floatbank11_copper_sulfate))


# Data Filtration ---------------------------------------------------------

data.train <- read_csv("./data/train_data/all_train.csv") %>% mutate(label = "train")
data.test <- read_csv("./data/test_data/all_test.csv") %>% mutate(label = "test")

data.train <- data.train %>% select(names(data.test), rougher.output.recovery, final.output.recovery)

# Rougher

data.rougher <- data.train %>% select(date, contains("rougher")) %>% melt(id.vars = c("date", "rougher.output.recovery"))
ggplot(data.rougher) + 
  geom_histogram(aes(x = value, fill = variable), color = "black") + 
  facet_wrap(variable~., scales = "free") + 
  theme(legend.position = "bottom")
