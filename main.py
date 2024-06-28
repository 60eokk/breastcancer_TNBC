import pandas as pd


def check_tnbc_suspective(patient_data):
    if patient_data['FOXA1'] <= 1320.1436:
        if patient_data['C6orf146'] <= 0:
            if patient_data['GNAO1'] > 173.0652:
                return 'NON-TNBC'
            else:
                if patient_data['GAGE13'] <= 0:
                    if patient_data['SMCP'] > 0.3651:
                        return 'NON-TNBC'
                    else:
                        if patient_data['GALP'] > 0.3808:
                            return 'NON-TNBC'
                        else:
                            if patient_data['CDK17'] <= 804.2082:
                                if patient_data['AQR'] <= 455.7455:
                                    return 'NON-TNBC'
                                else:
                                    if patient_data['OR10H2'] <= 0:
                                        if patient_data['C13orf34'] <= 126.3544:
                                            return 'NON-TNBC'
                                        else:
                                            if patient_data['HIST1H4L'] > 0.815:
                                                return 'NON-TNBC'
                                            else:
                                                if patient_data['CHST8'] > 293.0429:
                                                    return 'TNBC'
                                                else:
                                                    if patient_data['FABPS'] <= 85.6336:
                                                        if patient_data['HPR'] > 0.2361:
                                                            return 'TNBC'
                                                        else:
                                                            return 'NON-TNBC'
                                                    else:
                                                        if patient_data['SRDSA1'] <= 1866.936:
                                                            return 'NON-TNBC'
                                                        else:
                                                            if patient_data['391714'] > 0:
                                                                return 'TNBC'
                                                            else:
                                                                return 'NON-TNBC'
                                    else:
                                        return 'NON-TNBC'
                                    
                            else:
                                return 'NON-TNBC'
                
                else:
                    return 'TNBC'
        else:
            return 'TNBC'
    else:
        if patient_data['ZP2'] > 41.4533:
            if patient_data['CTRC'] <= 0.3677:
                if patient_data['C1orf110'] <= 0.4803:
                    return 'NON-TNBC'
                else:
                    return 'TNBC'
            else:
                return 'TNBC'
                if patient_data['TCP10L2'] > 0.2695:
                    return 'Non-TNBC'
                else:
                    return 'TNBC'
        else:
            if patient_data['TCP10L2'] > 0.2695:
                if patient_data['FAM120AOS'] > 1121.6393:
                    return 'NON-TNBC'
                else:
                    return 'TNBC'
            else:
                if patient_data['GABR2'] <= 5.1181:
                    return 'NON-TNBC'
                else:
                    if patient_data['GPR88'] > 17.5683:
                        return 'TNBC'
                    else:
                        return 'NON-TNBC'
                    

# Example usage of the TNBC suspicion function
sample_patient_data = new_data.iloc[0]
tnbc_suspective_result = check_tnbc_suspective(sample_patient_data)
print("TNBC Suspective Result for Sample Patient:\n", tnbc_suspective_result)


def load_data(file_path):
    # Load the dataset from an Excel file
    df = pd.read_excel(file_path)
    return df

def analyze_bbtnbc1(df):
    # Print column names for debugging
    print("Column names:", df.columns.tolist())

    # Check if ER, PR, and HER2 values are all less than 1
    df = df.drop(0)  # Drop the header row used for column names
    df = df.reset_index(drop=True)
    df = df.apply(pd.to_numeric, errors='coerce')

    tnbc_suspected = df[(df['ER'] < 1) & (df['PR'] < 1) & (df['HER2'] < 1)]

    return tnbc_suspected

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
    df['TNBC_Suspected'] = df.apply(follow_flowchart, axis=1)
    return df
def filter_tnbc_datasets(df):
    # Filter rows where TNBCYN is 'TNBC' or empty
    tnbc_df = df[(df['TNBCYN'] == 'TNBC') | (df['TNBCYN'].isna())]
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
    # Load the first dataset
    df1 = load_data('bbtnbc1.xlsx')
    # Load the combined dataset
    file_path = 'bbtnbc_final.xlsx'
    df = load_data(file_path)

    # Filter TNBC datasets
    tnbc_df = filter_tnbc_datasets(df)

    # Calculate normal reference values from the dataset
    normal_means = calculate_normal_means(df)

    # Clean the dataset for comparison
    df_cleaned = clean_data_for_comparison(df)

    # Analyze bbtnbc1.xlsx to find initial TNBC suspects
    tnbc_suspected = analyze_bbtnbc1(df1)
    # Compare the TNBC data with normal values to finalize TNBC suspicion
    final_classified = compare_with_normal(df_cleaned, normal_means)

    if not tnbc_suspected.empty:
        # Analyze from FOXA1 to GPR88 according to the flowchart
        tnbc_flowchart = analyze_flowchart(tnbc_suspected)

        # Filter only the TNBC datasets
        tnbc_only = tnbc_flowchart[tnbc_flowchart['TNBC_Suspected'] == True]

        # Load the second dataset
        df2 = load_data('bbtnbc2.xlsx')

        # Calculate normal reference values from the second dataset
        normal_means = calculate_normal_means(df2)

        # Clean the second dataset for comparison
        df2_cleaned = clean_data_for_comparison(df2)

        # Compare the TNBC data with normal values to finalize TNBC suspicion
        final_classified = compare_with_normal(df2_cleaned, normal_means)

        # Display the results
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        print(final_classified)
    else:
        print("No initial TNBC suspects found based on ER, PR, and HER2 values.")
    # Display the results
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(final_classified)

if __name__ == "__main__":
    main()