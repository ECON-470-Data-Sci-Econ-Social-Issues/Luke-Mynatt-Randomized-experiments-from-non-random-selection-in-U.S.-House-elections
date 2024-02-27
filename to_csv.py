import pandas as pd

def convert_txt_to_csv(txt_data_file, csv_output_file):
    # Define column names based on setup file information
    column_names = [
        "ICPSR STUDY NUMBER", "ICPSR VERSION NUMBER", "ICPSR PART NUMBER",
        "ASTERISK CNL-ONLY DATA", "CAND NUMBER", "ICPSR STATE CODE", 
        "YEAR OF ELECTION", "OFFICE CODE", "CONGRESSIONAL DIST NO", 
        "ELECTION TYPE CODE", "MONTH OF ELECTION", "CANDIDATE'S VOTE", 
        "ICPSR PARTY CODE", "CANDIDATE'S NAME", "CONGRESS TO WHICH ELECTD", 
        "TOTAL VOTE CAST IN ELEC", "NUMBER OF CANDIDATES", "HIGHEST PERCENT", 
        "2ND HIGHEST PERCENT", "MARGIN OF VICTORY", "CANDIDATE'S PERCENT", 
        "CANDIDATE'SP PERCENT FROM VICTOR", "ELECTION OUTCOME"
    ]

    # Define column widths based on setup file information
    colspecs = [
        (0, 4), (4, 5), (5, 6), (6, 7), (7, 9), (9, 11), (11, 14), 
        (14, 15), (15, 18), (18, 19), (19, 21), (21, 29), (29, 33), 
        (33, 79), (79, 82), (82, 90), (90, 92), (92, 96), (96, 100), 
        (100, 104), (104, 108), (108, 112), (112, 113)
    ]

    # Read the TXT file into a pandas DataFrame using fixed-width column format
    df = pd.read_fwf(txt_data_file, colspecs=colspecs, names=column_names, header=None, encoding='latin1')

    # Save the DataFrame to a CSV file
    df.to_csv(csv_output_file, index=False)

# Example usage
txt_data_file = './07757-0001-Data.txt'  # Update with actual path to your TXT data file
csv_output_file = './output_file.csv'  # Desired path for the output CSV file

convert_txt_to_csv(txt_data_file, csv_output_file)
