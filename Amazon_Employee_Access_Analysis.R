library(vroom)
library(tidyverse)
library(tidymodels)

amazon_train <- vroom("train.csv")
amazon_test <- vroom("test.csv")

amazon_train <- amazon_train|>
  mutate(ACTION = factor(ACTION)) # I turned the response into a Factor because it is binary (0/1)

#### EDA ####
amazon_train |>
  ggplot(aes(x = ACTION)) +
  geom_bar() +
  labs(
    x = "Access Decision",
    y = "Count",
    title = "Distribution of Employee Access Decisions"
  )

#### logistic regression ####
# defining my recipe for logistic regression.
my_recipe <- recipe(ACTION ~ ., data = amazon_train) |>
  step_mutate_at(all_predictors(), fn = ~factor(.)) |>
  step_other(all_nominal_predictors(), threshold = 0.001) |>
  step_dummy(all_nominal_predictors())

preped_recipe <- prep(my_recipe)

baked_amazon <- preped_recipe|>
  bake(new_data = NULL)


# defining my ML model and workflow

my_log_model <- logistic_reg()|>
  set_engine("glm")

my_workflow <- workflow()|>
  add_recipe(my_recipe)|>
  add_model(my_log_model)|>
  fit(data = amazon_train)

log_predictions <- predict(my_workflow, 
                           new_data = amazon_test,
                           type = "prob")

Log_submission <- log_predictions |>
  bind_cols(amazon_test) |>
  select(id, .pred_1) |>
  rename(ACTION = .pred_1)
  

vroom_write(x = Log_submission, file = "./LogPreds.csv", delim = ",")


#### Random Forest ####

my_forest_recipe <- recipe(ACTION ~ ., data = amazon_train) |>
  step_mutate_at(all_predictors(), fn = ~factor(.)) |>
  step_other(all_nominal_predictors(), threshold = 0.001) # I do not encode here because Trees handle categoracle data well.

# defining my Random Forest model and workflow
my_forest_mod <- rand_forest(mtry = tune(),
                             min_n = tune(),
                             trees = 500)|>
  set_engine("ranger")|>
  set_mode("classification")

forest_workflow <- workflow()|>
  add_recipe(my_forest_recipe)|>
  add_model(my_forest_mod)

# setting up a grid for CV
mygrid <- grid_regular(mtry(range = c(1, 9)),
                       min_n(range = c(2,10)),
                       levels = 5)
# spiting up the data for CV
folds <- vfold_cv(amazon_train, v = 5, repeats = 1)

# running K-fold CV
CV_results <- forest_workflow|>
  tune_grid(resamples = folds, 
            grid = mygrid,
            metrics = metric_set(accuracy, roc_auc))

# find best tuning parameters 
bestTune <- CV_results|>
  select_best(metric = "roc_auc") # I wanted roc_auc because we want the probabilities 
bestTune

# Defining the final workflow with the "best tunned" Random Forest
final_wf<- forest_workflow|>
  finalize_workflow(bestTune)|>
  fit(data = amazon_train)

# predictions using Random Forest
forest_predict <- final_wf|>
  predict(new_data = amazon_test,
          type = "prob")

# Feture engineering the predictions for Kaggle submission
forest_submission <- forest_predict |>
  bind_cols(amazon_test) |>
  select(id, .pred_1) |>
  rename(ACTION = .pred_1)


vroom_write(x = forest_submission, file = "./ForestPreds.csv", delim = ",")



