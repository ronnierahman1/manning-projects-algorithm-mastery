from budget.data_handler import get_total_budget, get_categories

def main():
    """
    Main entry point for Milestone 1 of SmartBudgetPlanner.
    
    This version of the application focuses on:
      1. Prompting the user for their total monthly income.
      2. Collecting spending categories along with the minimum required expense for each.
      3. Displaying the collected input in a readable format.
    
    Steps:
      - Call get_total_budget() to retrieve and validate the total monthly income.
      - Call get_categories() to collect the spending categories.
      - Print the total income and list each category with its minimum expense.
    """
    print("=== SmartBudgetPlanner - Milestone 1 ===")
    
    # Retrieve the user's total monthly income.
    total_income = get_total_budget()
    
    # Retrieve the spending categories and their minimum expenses.
    categories = get_categories()
    
    # Display the collected input.
    print("\nCollected Input:")
    print(f"Total Monthly Income: ${total_income:,.2f}")
    print("Spending Categories:")
    for cat in categories:
        print(f" - {cat['Category']}: ${cat['Minimum Expense']:.2f}")

if __name__ == "__main__":
    main()
