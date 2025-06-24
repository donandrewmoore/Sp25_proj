# Load necessary libraries
library(tidyverse)
library(afex)
library(stargazer)

# Load datasets
human_survey_data <- read_csv('survey_analysis_agg.csv')
cnn_data <- read_csv('cnn_result_agg.csv')

# Rename columns
human_survey_data <- human_survey_data %>%
  rename(
    participant_id = row_index,
    confidence = confidence_mean,
    accuracy = accuracy_mean
  )

cnn_data <- cnn_data %>%
  rename(
    participant_id = row_index,
    confidence = confidence_mean,
    accuracy = accuracy_mean
  )

# Label agent type
human_survey_data <- human_survey_data %>%
  mutate(agent_type = "Human")

cnn_data <- cnn_data %>%
  mutate(agent_type = "CNN")

# Adjust participant IDs for CNNs to make them unique
cnn_data <- cnn_data %>%
  mutate(participant_id = participant_id + 210)

# Function to run mixed ANOVA
run_mixed_anova <- function(dataset_choice, human_survey_data, cnn_data) {
  
  # Filter by dataset
  if (dataset_choice != "all") {
    human_survey_data <- human_survey_data %>%
      filter(dataset == dataset_choice)
    
    cnn_data <- cnn_data %>%
      filter(dataset == dataset_choice)
  }
  
  # Label agent type and adjust CNN participant IDs again for uniqueness
  human_survey_data <- human_survey_data %>%
    mutate(agent_type = "Human")
  
  cnn_data <- cnn_data %>%
    mutate(agent_type = "CNN",
           participant_id = participant_id + 1000)
  
  # Combine datasets
  combined_data <- bind_rows(human_survey_data, cnn_data)
  
  # Convert to long format
  long_df <- combined_data %>%
    pivot_longer(
      cols = c(confidence, accuracy),
      names_to = "measure",
      values_to = "score"
    ) %>%
    mutate(
      agent_type = factor(agent_type),
      measure = factor(measure, levels = c("confidence", "accuracy")),
      participant_id = factor(participant_id)
    )
  
  # Run ANOVA
  anova_result <- aov_ez(
    id = "participant_id",
    dv = "score",
    data = long_df,
    within = "measure",
    between = "agent_type",
    type = 3
  )
  
  # Extract and return table
  anova_table <- as.data.frame(anova_result$anova_table)
  anova_table$Effect <- rownames(anova_table)
  
  return(anova_table)
}

# Run ANOVAs for each dataset
anova_cars <- run_mixed_anova("cars", human_survey_data, cnn_data)
anova_flowers <- run_mixed_anova("flowers", human_survey_data, cnn_data)
anova_dmc <- run_mixed_anova("dmc", human_survey_data, cnn_data)
anova_all <- run_mixed_anova("all", human_survey_data, cnn_data)

# Print results using stargazer
stargazer(anova_cars, type = "latex", summary = FALSE, title = "ANOVA - Cars Dataset",
          out = "anova_cars.tex")

stargazer(anova_flowers, type = "latex", summary = FALSE, title = "ANOVA - Flowers Dataset",
          out = "anova_flowers.tex")

stargazer(anova_dmc, type = "latex", summary = FALSE, title = "ANOVA - DMC Dataset",
          out = "anova_dmc.tex")

stargazer(anova_all, type = "latex", summary = FALSE, title = "ANOVA - All Datasets",
          out = "anova_all.tex")
