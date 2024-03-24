library(tidyverse)
library(tidymodels)
tidymodels_prefer()

train_df <- read_csv("train.csv")
test_df <- read_csv("test.csv")
glimpse(train_df)

x_feats <- c('X_Minimum', 'X_Maximum','Y_Minimum', 'Y_Maximum', 'Pixels_Areas', 'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity', 'Minimum_of_Luminosity', 'Maximum_of_Luminosity', 'Length_of_Conveyer', 'TypeOfSteel_A300', 'TypeOfSteel_A400', 'Steel_Plate_Thickness', 'Edges_Index', 'Empty_Index', 'Square_Index', 'Outside_X_Index', 'Edges_X_Index', 'Edges_Y_Index', 'Outside_Global_Index', 'LogOfAreas', 'Log_X_Index', 'Log_Y_Index', 'Orientation_Index', 'Luminosity_Index', 'SigmoidOfAreas')
Y_feats <- c('Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults')

x_df <- train_df %>% select(all_of(x_feats))
Y_df <- train_df %>% select(all_of(Y_feats))

lm_model <-
  linear_reg() %>%
  set_engine("lm")

lm_fit <- lm_model %>%
  fit_xy(x = x_df, y = Y_df)

lm_fit %>% extract_fit_engine()

p <- predict(lm_fit, new_data = train_df)
names(p) <- Y_feats
# set max per row to 1
p %>%
  rowwise() %>%
  mutate(max_prob = max(cols(Pastry:Other_Faults)))
