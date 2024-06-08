import pandas as pd

# Load and inspect the uploaded Excel files
file_path1 = 'bbtnbc1.xlsx'
file_path2 = 'bbtnbc2.xlsx'

def load_data(file_path):
    # Load the dataset from an Excel file
    df = pd.read_excel(file_path)
    return df

def analyze_bbtnbc1(df):
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
                    if row['C1orf110'] > 0.4803:
                        return row['TCP10L2'] > 0.2695 and row['GABRR2'] > 5.1181 and row['GPR88'] > 17.5683
                    return row['C1orf110'] <= 0.4803 and row['FAM120AOS'] > 1121.6399 and row['GPR88'] > 17.5683
                return row['CTRC'] <= 0.3677 and row['C1orf110'] > 0.4803 and row['FAM120AOS'] > 1121.6399 and row['GPR88'] > 17.5683
            return row['ZP2'] <= 41.4533 and row['TCP10L2'] > 0.2695 and row['GABRR2'] > 5.1181 and row['GPR88'] > 17.5683
        if row['FOXA1'] <= 1320.1436:
            if row['C6orf146'] > 173.0652:
                return row['GAGE13'] > 0 and row['GPR88'] > 17.5683
            if row['C6orf146'] <= 173.0652:
                return row['SMCP'] > 0.3651 and row['CDK17'] > 804.2082 and row['AQR'] > 455.7455
    except KeyError as e:
        print(f"KeyError: {e}")
        return False
    except ValueError as e:
        print(f"ValueError: {e}")
        return False
    return False

def analyze_flowchart(df):
    # Apply the flowchart logic to classify TNBC
    df['TNBC_Suspected'] = df.apply(follow_flowchart, axis=1)
    return df

def calculate_normal_means(df):
    # Extract the last five rows as normal data
    normal_data = df.tail(5)
    
    # Convert relevant columns to numeric types
    normal_data = normal_data.apply(pd.to_numeric, errors='coerce')
    
    # Calculate the mean values for the normal data to set as reference
    normal_means = normal_data.mean(numeric_only=True)
    return normal_means

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
    df1 = load_data(file_path1)
    
    # Analyze bbtnbc1.xlsx to find initial TNBC suspects
    tnbc_suspected = analyze_bbtnbc1(df1)
    
    if not tnbc_suspected.empty:
        # Analyze from FOXA1 to GPR88 according to the flowchart
        tnbc_flowchart = analyze_flowchart(tnbc_suspected)
        
        # Load the second dataset
        df2 = load_data(file_path2)
        
        # Calculate normal reference values from the second dataset
        normal_means = calculate_normal_means(df2)
        
        # Clean the second dataset for comparison
        df2_cleaned = df2.iloc[:-5]
        df2_cleaned = df2_cleaned.apply(pd.to_numeric, errors='coerce')
        
        # Compare the data with normal values to finalize TNBC suspicion
        final_classified = compare_with_normal(df2_cleaned, normal_means)
        
        # Display the results
        print(final_classified.head())
    else:
        print("No initial TNBC suspects found based on ER, PR, and HER2 values.")

if __name__ == "__main__":
    main()