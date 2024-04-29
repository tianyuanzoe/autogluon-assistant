basic_intro_prompt = "The following sections contain descriptive information about the datascience competition:"
basic_system_prompt = "You are an expert assistant that parses information about datascience competitions."
columns_in_train_not_test_template = (
    "The following columns are in the train dataset and not test dataset. The label column is likely one of these:\n{}"
)
task_files_template = "# Available Files\n{}"
data_description_template = "# Data Description\n{}"
evaluation_description_template = "# Evaluation Description\n{}"
eval_metric_prompt = """
Based on the information provided in {evaluation_description}, identify the correct evaluation metric to be used from among these KEYS:
{metrics} 

The descriptions of these metrics are:
{metric_descriptions} 
respectively. 
If the exact metric is not in the list provided, then choose the metric that you think best approximates the one in the task description.
Only respond with the exact names of the metrics mentioned in KEYS. Do not respond with the metric descriptions.
"""
format_instructions_template = "{}"
infer_test_id_column_template = "The Id column for the sample submission is {output_id_column} but a column with that name is not found in the test data. Which column from the following list of test data columns is most likely to be the Id column:\n{test_columns}\nIf no reasonable Id column is present, for example if all the columns appear to be similarly named feature columns, response with the value NO_ID_COLUMN_IDENTIFIED"
parse_fields_template = "Based on the above information, what are the correct values for the following fields: {}"
zip_file_prompt = "If there are zip (e.g. .zip or .gz) versions of files and non-zipped versions of the files, choose the non-zip version. For example, return 'train.csv' rather than 'train.csv.zip'."
