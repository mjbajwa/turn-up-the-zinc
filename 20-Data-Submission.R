# Submit Data for Joining
library(tidyverse)
library(reshape2)

#final.recovery <- read_rds("./submissions/feb-19/test_set_final_recovery.rds")
#rougher.recovery <- read_rds("./submissions/feb-19/test_set_rougher_recovery.rds")

final.recovery <- read_rds("./submissions/mar-1/test_set_final_recovery.rds")
rougher.recovery <- read_rds("./submissions/mar-1/test_set_rougher_recovery.rds")

final.data <- final.recovery %>% left_join(rougher.recovery) %>% mutate(index = 1:n())

ggplot(final.data) + 
  geom_point(aes(x = index, y = final.output.recovery), color = "orange2")+
  geom_point(aes(x = index, y = rougher.output.recovery), color = "blue3") + 
  geom_vline(xintercept = final.data %>% filter(date > as.Date("2017-01-01")) %>% pull(index) %>% min(), size = 2) + 
  theme_bw() + 
  theme(legend.position = "bottom")
  
write_csv(final.data %>% select(-index), "./submissions/mar-1/submission_zinc_small_model.csv")

# Sample Edit -------------------------------------------------------------

# final.data <- final.data %>% mutate(final.output.recovery = ifelse(final.output.recovery < 40, 40, 0))

# # Submit Data for Joining
# library(tidyverse)
# library(reshape2)
# 
# #final.recovery <- read_rds("./submissions/feb-19/test_set_final_recovery.rds")
# #rougher.recovery <- read_rds("./submissions/feb-19/test_set_rougher_recovery.rds")
# 
# final.recovery <- read_rds("./submissions/mar-1/test_set_final_recovery_for_anton.rds")
# rougher.recovery <- read_rds("./submissions/mar-1/test_set_rougher_recovery_for_anton.rds")
# 
# final.data <- final.recovery %>% left_join(rougher.recovery) %>% mutate(index = 1:n())
# 
# write_csv(final.data %>% select(-index), "./submissions/mar-1/all_predictions_for_anton.csv")

