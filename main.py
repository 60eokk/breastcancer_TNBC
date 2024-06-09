import pandas as pd

def load_data(file_path):
    df = pd.read_excel(file_path)
    return df

def check_er_pr_her2(row):
    try:
        return (pd.isna(row['ER']) or row['ER'] < 1) and (pd.isna(row['PR']) or row['PR'] < 1) and (pd.isna(row['HER2']) or row['HER2'] < 1)
    except KeyError:
        return False

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

def filter_tnbc_datasets(df):
    # Apply ER, PR, HER2 check
    df['ER_PR_HER2_Check'] = df.apply(check_er_pr_her2, axis=1)
    # Apply flowchart logic
    df['TNBC_Suspected'] = df.apply(follow_flowchart, axis=1)
    # Filter rows where TNBCYN is 'TNBC' or empty, and ER_PR_HER2_Check and TNBC_Suspected are True
    tnbc_df = df[((df['TNBCYN'] == 'TNBC') | (df['TNBCYN'].isna())) & (df['ER_PR_HER2_Check'] == True) & (df['TNBC_Suspected'] == True)]
    return tnbc_df

def calculate_normal_means(df):
    normal_data = df.tail(5)
    normal_data = normal_data.apply(pd.to_numeric, errors='coerce')
    normal_means = normal_data.mean(numeric_only=True)
    return normal_means

def clean_data_for_comparison(df):
    df_cleaned = df.iloc[:-5]
    df_cleaned = df_cleaned.apply(pd.to_numeric, errors='coerce')
    return df_cleaned

def compare_with_normal(df, normal_means):
    def is_suspected_tnbc(row, normal_means):
        for col in normal_means.index:
            if pd.notna(normal_means[col]) and row[col] > normal_means[col]:
                return True
        return False

    df['TNBC_Suspected'] = df.apply(lambda row: is_suspected_tnbc(row, normal_means), axis=1)
    return df

def main():
    # Load the combined dataset
    file_path = 'bbtnbc_final.xlsx'
    df = load_data(file_path)
    
    # Filter TNBC datasets based on ER, PR, HER2 and flowchart logic
    tnbc_df = filter_tnbc_datasets(df)
    
    # Calculate normal reference values from the dataset
    normal_means = calculate_normal_means(df)
    
    # Clean the dataset for comparison
    df_cleaned = clean_data_for_comparison(df)
    
    # Compare the TNBC data with normal values to finalize TNBC suspicion
    final_classified = compare_with_normal(tnbc_df, normal_means)
    
    # Display the results
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(final_classified[['TNBC_Suspected']])

if __name__ == "__main__":
    main()