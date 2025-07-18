# Import necessary functions from the project's modules.
# - get_total_budget and get_categories handle basic user inputs (Milestone 1).
# - smart_allocation implements the tiered budgeting logic.
# - optimize_budget implements the dynamic programming (priority-based) budgeting.
# - plot_budget_breakdown visualizes the final budget breakdown.
from budget.data_handler import get_total_budget, get_categories
from budget.tiered_planner import smart_allocation
from budget.dp_planner import optimize_budget
from budget.utils import plot_budget_breakdown
import pandas as pd

def display_budget_breakdown(rows, total_income):
    """
    Create a textual summary of the budget allocation and print it to the console.
    
    Steps:
      1. Convert the list of allocation dictionaries into a Pandas DataFrame.
      2. Format the "Amount" column to display as currency.
      3. Sum up the raw numerical amounts (from "AmountRaw") to verify the totals.
      4. Print the budget breakdown in a readable table.
      5. Compare the total allocated with the total income to confirm that the budget is balanced.
    """
    # Write your code here.
    pass

def main():
    """
    Main entry point of the SmartBudgetPlanner application (Milestone 2 - Before Version).
    
    This version extends Milestone 1 by adding two budgeting strategies:
    
      Strategy 1: Smart Tiered Budgeting (Emergency fund-aware)
        - Prompts the user for emergency fund target and current emergency fund balance.
        - Uses smart_allocation() to split the remaining funds into "Emergency Fund" and "Miscellaneous".
      
      Strategy 2: Priority-Based Budgeting (Dynamic Programming)
        - Prompts the user for an optional priority for each spending category.
        - Asks for a savings goal percentage from the leftover funds.
        - Uses optimize_budget() to optimally allocate the remaining funds (after securing savings) across categories.
      
    Steps:
      1. Display a menu for selecting the budgeting strategy.
      2. Retrieve total monthly income and spending categories (Milestone 1 functionality).
      3. Calculate the sum of essential expenses.
      4. Based on the selected strategy, prompt for additional parameters and compute the allocation:
         - For Tiered Budgeting, prompt for emergency fund target and current emergency balance.
         - For Priority-Based Budgeting, prompt for category priorities and a savings goal percentage.
      5. Build a list (or table) of rows representing the full budget breakdown.
      6. Display the breakdown as text and plot it as a bar chart.
    """
    print("=== SmartBudgetPlanner ===")
    print("Choose your budgeting strategy:")
    print("1. Smart Tiered Budgeting (Emergency fund-aware)")
    print("2. Priority-Based Budgeting (Dynamic Programming)")

    # Step 1: Get the user's choice.
    choice = input("Enter 1 or 2: ").strip()

    # Step 2: Retrieve total monthly income and spending categories.
    total_income = get_total_budget()
    categories = get_categories()
    # Calculate the sum of all essential expenses.
    essential_total = sum(cat["Minimum Expense"] for cat in categories)

    if choice == "1":
        # Strategy 1: Smart Tiered Budgeting.
        # Step 3a: Prompt for emergency fund target.
        # Write your code here.
        emergency_target = None  # Write your code here.
        
        # Step 4a: Prompt for current emergency fund balance.
        # Write your code here.
        current_emergency = None  # Write your code here.
        
        # Step 5a: Compute the allocation using smart_allocation().
        allocation = smart_allocation(total_income, essential_total, emergency_target, current_emergency)
        
        # Step 6a: Build the budget breakdown rows.
        # Start with essential expenses.
        # Write your code here.
        rows = None  # Write your code here.
        
        # Then add the allocation entries (e.g., Emergency Fund and Miscellaneous).
        # Write your code here.
        # (You may use a loop to append each key-value pair from the allocation.)
        pass  # Write your code here.
        
        # Step 7a: Display and plot the final budget breakdown.
        display_budget_breakdown(rows, total_income)
        plot_budget_breakdown(rows, title="Smart Tiered Budget Allocation")

    elif choice == "2":
        # Strategy 2: Priority-Based Budgeting.
        # Step 3b: Prompt for an optional priority for each category.
        print("\nFor each category, enter an optional priority (default is 1.0).")
        for cat in categories:
            # Write your code here to prompt for priority and update each category.
            pass  # Write your code here.
        
        # Step 4b: Ask for the percentage of leftover funds to save.
        # Write your code here.
        savings_percent = None  # Write your code here.
        
        # Step 5b: Calculate the leftover funds after essential expenses.
        # Write your code here.
        leftover = None  # Write your code here.
        
        # Step 6b: Compute the savings amount based on the savings percentage.
        # Write your code here.
        savings_amount = None  # Write your code here.
        
        # Step 7b: Compute the funds available for dynamic programming allocation.
        # Write your code here.
        budget_for_dp = None  # Write your code here.
        
        # Optionally, display the computed leftover, savings, and available funds.
        # Write your code here.
        pass  # Write your code here.
        
        # Step 8b: Use optimize_budget() to compute the allocation on the available funds.
        from budget.dp_planner import optimize_budget
        result = optimize_budget(essential_total + budget_for_dp, categories)
        
        # Step 9b: Build the budget breakdown rows from the optimization results.
        # Write your code here.
        rows = None  # Write your code here.
        
        # Append a row for Savings.
        # Write your code here.
        pass  # Write your code here.
        
        # Step 10b: Display and plot the final budget breakdown.
        display_budget_breakdown(rows, total_income)
        plot_budget_breakdown(rows, title="Priority-Based Budget Allocation")

    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
