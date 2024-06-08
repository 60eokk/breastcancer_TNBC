import pandas as pd

def load_data(file_path):
    # Load the dataset from an Excel file
    df = pd.read_excel(file_path)
    return df

def clean_data(df):
    # Rename columns using the first row
    df.columns = df.iloc[0]
    df = df[1:]

    # Resetting the index after renaming
    df.reset_index(drop=True, inplace=True)
    
    # Convert relevant columns to numeric types
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

def follow_flowchart(row):
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
    return False

def classify_tnbc(df):
    # Apply the classification function to each row in the dataset
    df['TNBC_Suspected'] = df.apply(follow_flowchart, axis=1)
    return df

def main(file_path):
    # Load the data
    df = load_data(file_path)
    
    # Clean the data
    df_cleaned = clean_data(df)
    
    # Classify each individual as TNBC suspected or not
    classified_df = classify_tnbc(df_cleaned)
    
    # Return the final classified dataframe
    return classified_df

# Example usage:
file_path = 'bbtnbc2.xlsx'
classified_df = main(file_path)

# Display the results
import ace_tools as tools; tools.display_dataframe_to_user(name="Classified TNBC Results", dataframe=classified_df)
print(classified_df.head())