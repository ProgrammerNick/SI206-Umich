import json
import pandas as pd

# Function to load the data from the JSON file
def load_expenses(file_path):
    with open(file_path, 'r') as file:
        expenses_data = json.load(file)
    return pd.DataFrame(expenses_data)

# Function to get the total expenses
def total_expenses(df):
    return df['Amount'].sum()

# Function to get the category-wise breakdown of expenses
def category_breakdown(df):
    return df.groupby('Category')['Amount'].sum().sort_values(ascending=False)

# Function to get the total amount spent in a specific payment method
def payment_method_breakdown(df, payment_method):
    return df[df['Payment Method'] == payment_method]['Amount'].sum()

# Function to get the monthly spending
def monthly_spending(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M')
    return df.groupby('Month')['Amount'].sum()

# Function to get the most frequent category
def most_frequent_category(df):
    return df['Category'].mode()[0]

# Main function to interact with the user
def expense_tracker():
    file_path = input("Enter the path to your JSON expense data file: ")
    df = load_expenses(file_path)
    
    while True:
        print("\nAvailable Queries:")
        print("1. Total Expenses")
        print("2. Category Breakdown")
        print("3. Payment Method Breakdown")
        print("4. Monthly Spending")
        print("5. Most Frequent Category")
        print("6. Exit")
        
        query = input("Enter your query number: ")
        
        if query == '1':
            print(f"Total Expenses: ${total_expenses(df):,.2f}")
        
        elif query == '2':
            category_data = category_breakdown(df)
            print("\nCategory Breakdown:")
            print(category_data)
        
        elif query == '3':
            payment_method = input("Enter payment method (e.g., 'Credit Card', 'Cash'): ")
            print(f"Total spent with {payment_method}: ${payment_method_breakdown(df, payment_method):,.2f}")
        
        elif query == '4':
            monthly_data = monthly_spending(df)
            print("\nMonthly Spending:")
            print(monthly_data)
        
        elif query == '5':
            print(f"The most frequent category of spending is: {most_frequent_category(df)}")
        
        elif query == '6':
            print("Exiting the tracker. Goodbye!")
            break
        else:
            print("Invalid choice, please try again.")

# To run the tracker
if __name__ == '__main__':
    expense_tracker()
