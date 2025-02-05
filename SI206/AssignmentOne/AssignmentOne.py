import json
import pandas as pd

def load_expenses(file_path):
    """Load expenses from a JSON file into a Pandas DataFrame."""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # Convert the JSON format into a DataFrame
        df = pd.DataFrame(data)
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return pd.DataFrame(columns=["Date", "Category", "Amount", "Description"])

def save_expenses(file_path, df):
    """Save the DataFrame back to the JSON file."""
    # Convert the DataFrame back into the original JSON format
    data = {
        "Date": df["Date"].tolist(),
        "Category": df["Category"].tolist(),
        "Amount": df["Amount"].tolist(),
        "Description": df["Description"].tolist()
    }

    # Save the data as JSON
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def add_expense(df):
    """Add a new expense to the DataFrame."""
    date = input("Enter date (YYYY-MM-DD): ")
    category = input("Enter category: ")

    # Validate the amount input
    while True:
        try:
            amount_input = input("Enter amount: ")
            amount = float(amount_input.replace('$', '').strip())
            break
        except ValueError:
            print("Invalid amount. Please enter a valid number, optionally prefixed with a $ sign.")

    description = input("Enter description: ")

    new_expense = {"Date": date, "Category": category, "Amount": amount, "Description": description}
    df = pd.concat([df, pd.DataFrame([new_expense])], ignore_index=True)
    return df

def view_summary(df):
    """View summary statistics of expenses."""
    if df.empty:
        print("No expenses to summarize.")
        return

    print("\nExpense Summary:")
    print("Total Amount Spent: $", df["Amount"].sum())
    print("\nCategory Breakdown:")
    print(df.groupby("Category")["Amount"].sum())
    print("\nMonthly Breakdown:")
    df["Month"] = pd.to_datetime(df["Date"]).dt.to_period("M")
    print(df.groupby("Month")["Amount"].sum())
    df.drop(columns=["Month"], inplace=True)

def filter_expenses(df):
    """Filter expenses based on user input."""
    category = input("Enter category to filter by (leave blank for all): ")
    if category:
        filtered = df[df["Category"] == category]
    else:
        filtered = df

    print("\nFiltered Expenses:")
    print(filtered)

def main():
    file_path = "expenses_1.json"  # Use expenses_1.json
    expenses = load_expenses(file_path)

    while True:
        print("\nExpense Tracker Menu:")
        print("1. Add Expense")
        print("2. View Summary")
        print("3. Filter Expenses")
        print("4. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            expenses = add_expense(expenses)
            save_expenses(file_path, expenses)  # Save back to expenses_1.json
        elif choice == "2":
            view_summary(expenses)
        elif choice == "3":
            filter_expenses(expenses)
        elif choice == "4":
            print("Exiting Expense Tracker. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
