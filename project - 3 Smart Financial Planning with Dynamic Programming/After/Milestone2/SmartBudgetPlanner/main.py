# Import necessary functions from the project's modules.
# - get_total_budget and get_categories (Milestone 1 functions) handle user inputs.
# - smart_allocation implements the tiered budgeting logic (Milestone 2).
# - optimize_budget implements the dynamic programming (priority-based) budgeting (Milestone 2).
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
    Main entry point of the SmartBudgetPlanner application after Milestone 2.
    
    This version extends the basic input collection (Milestone 1) by adding two budgeting strategies:
    
      Strategy 1: Smart Tiered Budgeting (Emergency fund-aware)
        - Prompts the user for their emergency fund target and current emergency fund balance.
        - Uses smart_allocation() to split the remaining funds into Emergency Fund and Miscellaneous based on fixed percentages.
      
      Strategy 2: Priority-Based Budgeting (Dynamic Programming)
        - Prompts the user for an optional priority for each spending category.
        - Asks for a savings goal percentage from the leftover funds.
        - Uses optimize_budget() to optimally allocate the remaining funds (after securing savings) across categories.
      
    The program then displays the full budget breakdown in a text table and plots a bar chart of the allocations.
    
    Steps:
      1. Display a menu for selecting the budgeting strategy.
      2. Retrieve total monthly income and spending categories (Milestone 1).
      3. Compute the sum of essential expenses.
      4. Depending on the chosen strategy, prompt for additional parameters and compute allocations:
          - For Tiered Budgeting: prompt for emergency fund target and current emergency balance.
          - For Priority-Based Budgeting: prompt for category priorities and savings goal percentage.
      5. Build a list of rows (dictionaries) representing the full budget breakdown.
      6. Display the breakdown and plot the allocations.
    """
    print("=== SmartBudgetPlanner ===")
    print("Choose your budgeting strategy:")
    print("1. Smart Tiered Budgeting (Emergency fund-aware)")
    print("2. Priority-Based Budgeting (Dynamic Programming)")
    
    # Step 1: Get the user's choice.
    choice = input("Enter 1 or 2: ").strip()

    # Step 2: Retrieve the total monthly income and spending categories.
    total_income = get_total_budget()
    categories = get_categories()
    # Calculate the sum of essential expenses.
    essential_total = sum(cat["Minimum Expense"] for cat in categories)

    if choice == "1":
        # Strategy 1: Smart Tiered Budgeting.
        
        # Step 3a: Prompt the user for emergency fund target and current emergency fund balance.
        # Write your code here to prompt and convert the input values.
        emergency_target = None  # Write your code here.
        current_emergency = None  # Write your code here.
        
        # Step 4a: Use smart_allocation() to compute the allocation for the remaining funds.
        allocation = smart_allocation(total_income, essential_total, emergency_target, current_emergency)
        
        # Step 5a: Build the list of rows for the budget breakdown.
        # Start with essential expenses.
        rows = [{"Category": cat["Category"], "Amount": cat["Minimum Expense"], "AmountRaw": cat["Minimum Expense"]}
                for cat in categories]
        # Then add the allocation entries (e.g., Emergency Fund and Miscellaneous).
        for key, value in allocation.items():
            rows.append({"Category": key, "Amount": value, "AmountRaw": value})
        
        # Step 6a: Display and plot the final budget breakdown.
        display_budget_breakdown(rows, total_income)
        plot_budget_breakdown(rows, title="Smart Tiered Budget Allocation")

    elif choice == "2":
        # Strategy 2: Priority-Based Budgeting.
        
        # Step 3b: Prompt the user for an optional priority for each category.
        print("\nFor each category, enter an optional priority (default is 1.0).")
        for cat in categories:
            try:
                value = input(f"Priority for {cat['Category']} (default 1.0): ").strip()
                cat['Priority'] = float(value) if value else 1.0
            except ValueError:
                print("Invalid input, using default priority 1.0")
                cat['Priority'] = 1.0
        
        # Step 4b: Ask for the percentage of leftover funds to save.
        # Write your code here to prompt for and validate the savings percentage.
        savings_percent = None  # Write your code here.
        
        # Step 5b: Calculate leftover funds after essential expenses.
        leftover = total_income - essential_total
        # Compute the savings amount based on the specified percentage.
        savings_amount = None  # Write your code here.
        # The funds available for dynamic programming allocation.
        budget_for_dp = None  # Write your code here.
        
        # Optionally, display the computed leftover, savings, and budget for optimization.
        # Write your code here.
        
        # Step 6b: Use optimize_budget() on the available funds (essential_total + budget_for_dp) and the categories.
        result = optimize_budget(essential_total + budget_for_dp, categories)
        
        # Step 7b: Build the rows for the budget breakdown from the optimization results.
        rows = []
        for item in result:
            rows.append({
                "Category": item["Category"],
                "Amount": item["Total Allocated"],
                "AmountRaw": item["Total Allocated"]
            })
        # Append the Savings row.
        rows.append({
            "Category": "Savings",
            "Amount": savings_amount,
            "AmountRaw": savings_amount
        })
        
        # Step 8b: Display and plot the final budget breakdown.
        display_budget_breakdown(rows, total_income)
        plot_budget_breakdown(rows, title="Priority-Based Budget Allocation")

    else:
        # If an invalid option is selected, notify the user.
        print("Invalid choice. Exiting.")

# Ensure main() runs when the script is executed directly.
if __name__ == "__main__":
    main()
