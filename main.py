import pandas as pd

def load_data(file_path):
    # Load the dataset from an Excel file
    df = pd.read_excel(file_path)
    return df

def follow_flowchart(row):
    try:
        if row['FOXA1'] > 1320.1436:
            if row['ZP2'] > 41.4533:
                if row['CTRC'] > 0.3677:
                    return True
                else:
                    if row['C1orf110'] > 0.4803:
                        return True
                    else:
                        if row['FAM120AOS'] > 1321.6393:
                            return True
                        else:
                            return False
            else:
                if row['TCP10L2'] > 0.2695:
                    if row['GABRR2'] > 5.1181:
                        if row['GPR88'] > 17.5683:
                            return True
                        else:
                            return False
                    else:
                        return False
                else:
                    return False
        else:
            return False
    except KeyError as e:
        print(f"KeyError: {e}")
        return False
    except ValueError as e:
        print(f"ValueError: {e}")
        return False

def analyze_flowchart(df):
    df['TNBC_Flowchart'] = df.apply(follow_flowchart, axis=1)
    return df

def filter_tnbc_datasets(df):
    # Filter rows where TNBCYN is 'TNBC' or empty, and TNBC_Flowchart is True
    tnbc_df = df[(df['TNBCYN'] == 'TNBC') | (df['TNBCYN'].isna()) | (df['TNBC_Flowchart'] == True)]
    return tnbc_df

def calculate_normal_means(df):
    # Extract the last five rows as normal data
    normal_data = df.tail(5)
    
    # Convert relevant columns to numeric types
    normal_data = normal_data.apply(pd.to_numeric, errors='coerce')
    
    # Calculate the mean values for the normal data to set as reference
    normal_means = normal_data.mean(numeric_only=True)
    return normal_means

def clean_data_for_comparison(df):
    # Remove the last five rows used for normal data
    df_cleaned = df.iloc[:-5]
    
    # Convert relevant columns to numeric types
    df_cleaned = df_cleaned.apply(pd.to_numeric, errors='coerce')
    return df_cleaned

def compare_with_normal(df, normal_means):
    # Function to determine if a person is suspected TNBC based on their data
    def is_suspected_tnbc(row, normal_means):
        for col in normal_means.index:
            if pd.notna(normal_means[col]) and row[col] > normal_means[col]:
                return True
        return False

    # Apply the function to classify each person in the dataset
    df['TNBC_Suspected'] = df.apply(lambda row: is_suspected_tnbc(row, normal_means), axis=1)
    return df

def main():
    # Load the combined dataset
    file_path = 'bbtnbc_final.xlsx'
    df = load_data(file_path)
    
    # Analyze using the flowchart
    df = analyze_flowchart(df)
    
    # Filter TNBC datasets
    tnbc_df = filter_tnbc_datasets(df)
    
    # Calculate normal reference values from the dataset
    normal_means = calculate_normal_means(df)
    
    # Clean the dataset for comparison
    df_cleaned = clean_data_for_comparison(df)
    
    # Compare the TNBC data with normal values to finalize TNBC suspicion
    final_classified = compare_with_normal(df_cleaned, normal_means)
    
    # Display the results
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(final_classified)

if __name__ == "__main__":
    main()