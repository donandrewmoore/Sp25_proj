import numpy as np
import pandas as pd

"""
This script cleans and processes survey data exported from QuestionPro, producing a long-format CSV suitable for further analysis.

How row_index relates to Response ID:
- The raw data file contains a 'Response ID' column for each respondent.
- During cleaning, some rows are filtered out (e.g., incomplete responses, failed attention checks).
- The 'row_index' field in the output is a zero-based index corresponding to the order of respondents after all filtering steps.
- To map a 'row_index' in the cleaned data back to the original 'Response ID', use the following approach:
    1. Apply the same filtering steps to the raw data as in this script.
    2. The nth row in the filtered DataFrame (where n = row_index) will have the corresponding 'Response ID'.

General overview:
- Reads the raw survey CSV, applies quality filters, and extracts relevant columns for each task (flowers, cars, dmc).
- Reshapes the data to long format, applies softmax to logits, merges with ground truth labels, and computes confidence/accuracy.
- Outputs the cleaned, long-format data as 'survey_analysis.csv'.
- Update the DICT_FILE path to point to your dictionary file as needed.
"""

# Input and output file paths (relative)
INPUT_FILE = "QuestionPro-SR-RawData-04-28-2025.csv"  # or update to your input file
OUTPUT_FILE = "survey_analysis.csv" # or update to your desired output file name
# Placeholder for the dictionary file path; update this as needed
DICT_FILE = "NEO_data/human_data_dict.csv"  # <-- UPDATE THIS PATH TO YOUR DICTIONARY FILE

# 1. Read and clean the survey data
survey_data = pd.read_csv(INPUT_FILE, skiprows=3)
x = 293 - 73  # Adjust as needed for your data
survey_data = survey_data.tail(x)
survey_data = survey_data[survey_data["Please complete the reCAPTCHA verification to continue this survey."] == "PASS"]
survey_data = survey_data.loc[:, survey_data.columns.str.len() <= 30]
survey_data = survey_data[survey_data['Response Status'] == 'Completed']

# 2. Attention check: keep only rows that pass
survey_attention = survey_data.filter(regex="How's", axis=1)
keep_indices = []
for index, row in survey_attention.iterrows():
    if any(row.str.contains('pass', case=False, na=False)):
        keep_indices.append(index)
survey_data = survey_data.loc[keep_indices]
survey_data.reset_index(drop=True, inplace=True)

# 3. Extract relevant columns for each task
def extract_and_rename(df, regex, names, prefix):
    sub = df.filter(regex=regex, axis=1)
    sub = sub.iloc[:, 10:]
    cols = []
    for i in range(1, 51):
        cols.extend([f"{name}_{i}" for name in names])
    sub.columns = cols
    return sub

flower_names = ['Buttercup', 'Coltsfoot', 'Dandelion', 'Daffodil', 'Sunflower']
car_names = ['Audi', 'BMW', 'Chevrolet', 'Dodge', 'Ford']
dmc_names = ['Dog', 'Chicken', 'Muffin']

survey_flower = extract_and_rename(survey_data, 'Buttercup|Daffodil|Sunflower|Coltsfoot|Dandelion', flower_names, 'flower')
survey_cars = extract_and_rename(survey_data, 'Audi|BMW|Chevrolet|Dodge|Ford', car_names, 'car')
survey_dmc = extract_and_rename(survey_data, 'Dog|Chicken|Muffin', dmc_names, 'dmc')

# 4. Reshape wide survey data into long format and apply softmax
def reshape_survey_data(df, option_type):
    df = df.reset_index().rename(columns={'index': 'row_index'})
    df_long = df.melt(id_vars='row_index', var_name=f'{option_type}_full', value_name='logit')
    df_long[[f'{option_type}_option', 'question_group']] = df_long[f'{option_type}_full'].str.rsplit('_', n=1, expand=True)
    df_long['question_group'] = df_long['question_group'].astype(int)
    df_long = df_long.drop(columns=[f'{option_type}_full'])
    all_nan_groups = (
        df_long.groupby(['question_group', 'row_index'])['logit']
        .transform(lambda x: x.isna().all())
    )
    df_long = df_long[~all_nan_groups]
    if df_long.empty:
        print(f"No valid data found for {option_type}.")
        return pd.DataFrame(columns=['row_index', 'confidence', f'{option_type}_option', 'question_group'])
    df_long['logit'] = pd.to_numeric(df_long['logit'], errors='coerce')
    def softmax(x):
        x = np.array(x-0.5)
        x = x[~np.isnan(x)]
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)
    def apply_softmax(group):
        logits = group['logit'].values
        probs = softmax(logits)
        group = group.loc[~group['logit'].isna()].copy()
        group['probs'] = probs
        group['confidence'] = probs.max()
        return group
    df_long_with_confidence = df_long.groupby(['question_group', 'row_index'], group_keys=False).apply(apply_softmax)
    cols_order = ['row_index', 'question_group', f'{option_type}_option', 'probs', 'confidence']
    df_long_with_confidence = df_long_with_confidence[cols_order]
    df_long_with_confidence = df_long_with_confidence.sort_values(by=['row_index', 'question_group', f'{option_type}_option'])
    df_long_with_confidence.reset_index(drop=True, inplace=True)
    return df_long_with_confidence

survey_flower_long = reshape_survey_data(survey_flower, 'flower')
survey_cars_long = reshape_survey_data(survey_cars, 'car')
survey_dmc_long = reshape_survey_data(survey_dmc, 'dmc')

# 5. Read and process the dictionary file (update path as needed)
dict_df = pd.read_csv(DICT_FILE)
dict_df = dict_df[dict_df['assignment'] == 'test']
dict_df = dict_df.rename(columns={'image': 'question_group'})
dict_df['question_group'] = dict_df['question_group'].str.replace('.jpg', '', regex=False)
dict_df_flower = dict_df[dict_df['dataset'] == 'oxflowers'][['question_group', 'true_class']]
dict_df_cars = dict_df[dict_df['dataset'] == 'stanford_cars'][['question_group', 'true_class']]
dict_df_dmc = dict_df[dict_df['dataset'] == 'dog_chicken'][['question_group', 'true_class']]

# 6. Merge and stack
survey_flower_long = survey_flower_long.rename(columns={'flower_option': 'option'})
survey_cars_long = survey_cars_long.rename(columns={'car_option': 'option'})
survey_dmc_long = survey_dmc_long.rename(columns={'dmc_option': 'option'})
for df in [survey_flower_long, survey_cars_long, survey_dmc_long, dict_df_flower, dict_df_cars, dict_df_dmc]:
    df['question_group'] = df['question_group'].astype(int)
survey_flower_long = survey_flower_long.merge(dict_df_flower, on='question_group', how='left')
survey_cars_long = survey_cars_long.merge(dict_df_cars, on='question_group', how='left')
survey_dmc_long = survey_dmc_long.merge(dict_df_dmc, on='question_group', how='left')
survey_long = pd.concat([survey_flower_long, survey_cars_long, survey_dmc_long], ignore_index=True)

# 7. Compute predicted class and accuracy
def get_predict_class(group):
    group['predict_index'] = group['confidence'] == group['probs']
    group['accuracy_indicator'] = np.where(
        group['predict_index'],
        (group['option'] == group['true_class']).astype(int),
        np.nan
    )
    group['accuracy'] = group['accuracy_indicator'].mean()
    return group

survey_analysis = survey_long.groupby(['row_index', 'question_group'], group_keys=False).apply(get_predict_class)
survey_analysis = survey_analysis.merge(dict_df[['true_class','dataset']], on='true_class', how='left')
row_stats = survey_analysis.groupby('row_index', as_index=False).agg(
    confidence_mean=('confidence', 'mean'),
    accuracy_mean=('accuracy', 'mean')
)
survey_analysis = survey_analysis.merge(row_stats, on='row_index', how='left')
survey_analysis = survey_analysis.drop(columns=['predict_index', 'accuracy_indicator'])
survey_analysis.drop_duplicates(inplace=True)
survey_analysis = survey_analysis.sort_values(by=['row_index', 'question_group'])
survey_analysis.reset_index(drop=True, inplace=True)

# 8. Save the cleaned and processed data
survey_analysis.to_csv(OUTPUT_FILE, index=False)

print(f"Survey data cleaning complete. Output saved to {OUTPUT_FILE}.") 