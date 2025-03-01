import pandas as pd

def clean_and_compute_dividends(input_file, output_file):

    df = pd.read_csv(input_file)
    
    # Remove rows that have no date
    df = df.dropna(subset=["date"])
    
    # Convert 'date' to a datetime object. Rows with an invalid date become NaT and are dropped.
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    # Extract the year from the date to enable annual aggregation.
    df['Year'] = df['date'].dt.year

    # Convert the DIVAMT column to numeric. Any non-numeric values (like "C") are set to NaN and then replaced with 0.
    df['DIVAMT'] = pd.to_numeric(df['DIVAMT'], errors='coerce').fillna(0)
    df['RET'] = pd.to_numeric(df['RET'], errors='coerce').fillna(0)

    # Group the data by PERMNO (company identifier) and Year and sum the dividend amounts.
    annual_div = df.groupby(['PERMNO', 'Year'])['DIVAMT'].sum().reset_index()
    annual_div = annual_div.rename(columns={'DIVAMT': 'AnnualDiv'})
    
    # Merge the annual dividend sum back into the main dataframe.
    df = df.merge(annual_div, on=['PERMNO', 'Year'], how='left')
    
    # Remove any company–year where the total annual dividend is zero.
    df = df[df['AnnualDiv'] != 0]
    
    # Count the number of records (months) available for each PERMNO/Year group.
    month_counts = df.groupby(['PERMNO', 'Year']).size().reset_index(name='MonthCount')
    df = df.merge(month_counts, on=['PERMNO', 'Year'], how='left')
    
    # Compute the equal monthly dividend yield by dividing the annual dividend by the number of months present.
    df['MonthlyDiv'] = df['AnnualDiv'] / df['MonthCount']
    
    # Replace the monthly dividend value (DIVAMT) with the computed monthly yield.
    df['DIVAMT'] = df['MonthlyDiv']
    
    # Drop columns used only for calculation (AnnualDiv, MonthCount, MonthlyDiv)
    df_clean = df.drop(columns=['AnnualDiv', 'MonthCount', 'MonthlyDiv'])
    
    # Reset index if desired
    df_clean = df_clean.reset_index(drop=True)
    
    # Save the cleaned data to a CSV file for further processing
    df_clean.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")

if __name__ == '__main__':
    input_file = './eapxqdfipgfwn9cv.csv'      
    output_file = 'cleaned_data.csv'
    clean_and_compute_dividends(input_file, output_file)
