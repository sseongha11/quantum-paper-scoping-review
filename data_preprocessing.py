import pandas as pd

# Load the CSV files
engineering_village_df = pd.read_csv('Engineering_Village_detailed_7-5-2024_225818185.csv', on_bad_lines='skip')
scopus_df = pd.read_csv('backup/scopus.csv')

# Define a mapping from Engineering Village columns to Scopus columns
column_mapping = {
    'Author': 'Authors',
    'Author affiliation': 'Affiliations',
    'Uncontrolled terms': 'Author Keywords',
    'Publisher/Repository': 'Publisher',
    'Language': 'Language of Original Document',
    'Controlled/Subject terms': 'Index Keywords',
    'Open Access type(s)': 'Open Access',
    'Data Provider': 'Source',
    'Publication year': 'Year',
    'Source': 'Source title',
}

# Rename the columns in Engineering Village dataframe
engineering_village_df.rename(columns=column_mapping, inplace=True)


# Function to reformat authors' names from Engineering Village format to Scopus format
def reformat_authors(authors):
    if pd.isna(authors):
        return authors
    authors_list = authors.split('; ')
    formatted_authors = []
    for author in authors_list:
        if ' (' in author:
            name, affiliation = author.split(' (')
            formatted_name = name.strip() + " " + affiliation.replace(')', '').strip()
        else:
            formatted_name = author.strip()
        formatted_authors.append(formatted_name)
    return '; '.join(formatted_authors)


# Apply the reformatting function to the Authors column in Engineering Village dataframe
engineering_village_df['Authors'] = engineering_village_df['Authors'].apply(reformat_authors)

# Ensure unique column names by appending suffixes where necessary
engineering_village_df = engineering_village_df.loc[:, ~engineering_village_df.columns.duplicated()]

# Select only the columns from Engineering Village that are present in Scopus
common_columns = [col for col in scopus_df.columns if col in engineering_village_df.columns]
engineering_village_df = engineering_village_df[common_columns]

# Add missing columns to the Engineering Village dataframe to match the Scopus dataframe's structure
for col in scopus_df.columns:
    if col not in engineering_village_df.columns:
        engineering_village_df[col] = pd.NA

# Ensure the column order matches Scopus
engineering_village_df = engineering_village_df[scopus_df.columns]

# Combine the dataframes
initial_scopus_count = len(scopus_df)
initial_ev_count = len(engineering_village_df)

# Drop columns with all NA values from both DataFrames
scopus_df_cleaned = scopus_df.dropna(axis=1, how='all')
engineering_village_df_cleaned = engineering_village_df.dropna(axis=1, how='all')

# Concatenate the cleaned DataFrames
merged_df = pd.concat([scopus_df_cleaned, engineering_village_df_cleaned], ignore_index=True)

# Continue with the rest of your code for dropping duplicates and saving the DataFrame
merged_df.drop_duplicates(subset=['DOI'], inplace=True)
final_count = len(merged_df)

# Save the merged dataframe to a new CSV file
output_path = 'outputs/merged_output.csv'
merged_df.to_csv(output_path, index=False)

# Output the counts and the path to the saved file
print(f"Initial Scopus count: {initial_scopus_count}")
print(f"Initial Engineering Village count: {initial_ev_count}")
print(f"Final merged count: {final_count}")
print(f"Output path: {output_path}")
