import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to load and clean the data from the JSON file
def load_and_clean_expenses(file_path):
    with open(file_path, 'r') as file:
        expenses_data = json.load(file)
    
    df = pd.DataFrame(expenses_data)

    # 1. Remove irrelevant columns (e.g., extra columns)
    df = df[['Date', 'Category', 'Amount', 'Description', 'Payment Method']]

    # 2. Standardize Category names by correcting typos
    category_corrections = {
        'Fod': 'Food',
        'Transprt': 'Transport',
        'Clothng': 'Clothing'
    }
    df['Category'] = df['Category'].replace(category_corrections)

    # 3. Handle missing data (e.g., missing Amount, Date)
    df = df.dropna(subset=['Amount', 'Date'])  # Remove rows with missing critical data
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')  # Ensure Amount is numeric
    df = df.dropna(subset=['Amount'])  # Remove rows with invalid Amount after conversion

    # 4. Convert 'Date' to a consistent format (ensure it’s datetime)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])  # Remove rows with invalid dates

    # 5. Remove duplicate entries based on Description and Amount
    df = df.drop_duplicates(subset=['Description', 'Amount'])

    # 6. Clean up Payment Method consistency (e.g., lowercase and standardization)
    df['Payment Method'] = df['Payment Method'].str.strip().str.title()

    return df

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
    df['Month'] = df['Date'].dt.to_period('M')
    return df.groupby('Month')['Amount'].sum()

# Function to get the category spending trends
def category_trends(df, category):
    df['Month'] = df['Date'].dt.to_period('M')
    category_data = df[df['Category'] == category].groupby('Month')['Amount'].sum()
    return category_data

# Function to get top N expensive items
def top_expenses(df, n=5):
    return df.nlargest(n, 'Amount')[['Date', 'Category', 'Amount', 'Description']]

# Function to get budget vs actual expenses for a category
def budget_vs_actual(df, category, budget):
    actual = df[df['Category'] == category]['Amount'].sum()
    return {"Budget": budget, "Actual": actual, "Difference": budget - actual}

# Function to generate a bar chart for category breakdown
def plot_category_breakdown(df):
    category_data = category_breakdown(df)
    category_data.plot(kind='bar', title="Category Breakdown", color='skyblue', figsize=(10, 6))
    plt.ylabel('Amount')
    plt.show()

# Function to generate a line chart for monthly spending
def plot_monthly_spending(df):
    monthly_data = monthly_spending(df)
    monthly_data.plot(kind='line', title="Monthly Spending Trend", color='green', marker='o', figsize=(10, 6))
    plt.ylabel('Amount')
    plt.show()

# Main function to interact with the user
def expense_tracker():
    file_path = input("Enter the path to your JSON expense data file: ")
    df = load_and_clean_expenses(file_path)
    
    while True:
        print("\nAvailable Queries:")
        print("1. Total Expenses")
        print("2. Category Breakdown")
        print("3. Payment Method Breakdown")
        print("4. Monthly Spending")
        print("5. Category Spending Trends")
        print("6. Top N Expenses")
        print("7. Budget vs Actual Spending")
        print("8. Visualize Category Breakdown (Bar Chart)")
        print("9. Visualize Monthly Spending Trend (Line Chart)")
        print("10. Exit")
        
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
            category = input("Enter category to track trends (e.g., 'Food', 'Transport'): ")
            category_trends_data = category_trends(df, category)
            print(f"\nTrends for {category}:")
            print(category_trends_data)
        
        elif query == '6':
            n = int(input("Enter the number of top expenses to show: "))
            top_expenses_data = top_expenses(df, n)
            print(f"\nTop {n} Expenses:")
            print(top_expenses_data)
        
        elif query == '7':
            category = input("Enter the category for budget vs actual (e.g., 'Food', 'Transport'): ")
            budget = float(input("Enter your budget for this category: "))
            budget_data = budget_vs_actual(df, category, budget)
            print(f"\nBudget vs Actual for {category}:")
            print(budget_data)
        
        elif query == '8':
            plot_category_breakdown(df)
        
        elif query == '9':
            plot_monthly_spending(df)
        
        elif query == '10':
            print("Exiting the tracker. Goodbye!")
            break
        else:
            print("Invalid choice, please try again.")

# To run the tracker
if __name__ == '__main__':
    expense_tracker()
