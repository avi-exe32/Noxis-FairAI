import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing

# 1. & 2. Load the dataset using pandas
try:
    df = pd.read_csv('test_biased_loans.csv')
except FileNotFoundError:
    print("Error: 'dataset.csv' not found. Please ensure the file exists in the working directory.")
    exit(1)

# 3. Drop missing values in the key columns
# Key columns are the sensitive attribute ('gender') and the target label ('loan_approved')
df = df.dropna(subset=['gender', 'loan_approved'])

# Reset index to ensure alignment when we add the weights back later
df = df.reset_index(drop=True)

# Define the privileged and unprivileged groups for the Reweighing algorithm
# Context specifies: gender = 1 is privileged, gender = 0 is unprivileged
privileged_groups = [{'gender': 1}]
unprivileged_groups = [{'gender': 0}]

# 4. Convert the dataframe into an AIF360 BinaryLabelDataset
# Context specifies: favorable outcome value is 1
aif360_dataset = BinaryLabelDataset(
    df=df,
    label_names=['loan_approved'],
    protected_attribute_names=['gender'],
    favorable_label=1.0,
    unfavorable_label=0.0
)

# 5. Apply the Reweighing algorithm
# This will calculate weights for each instance to mitigate the Disparate Impact (currently 0.58)
reweighing_algo = Reweighing(
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)

# Fit the algorithm to the dataset and transform it to generate the weights
aif360_dataset_transformed = reweighing_algo.fit_transform(aif360_dataset)

# 6. Extract the newly calculated instance weights
new_instance_weights = aif360_dataset_transformed.instance_weights

# 7. Add these weights as a new column 'fair_weights' to the pandas dataframe
df['fair_weights'] = new_instance_weights

# 8. Save the mitigated dataframe to a new file named 'mitigated_dataset.csv'
df.to_csv('mitigated_dataset.csv', index=False)

print("Bias mitigation complete. Reweighed data saved to 'mitigated_dataset.csv'.")