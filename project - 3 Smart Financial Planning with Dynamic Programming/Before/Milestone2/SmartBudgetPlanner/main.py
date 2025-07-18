from budget.data_handler import get_total_budget, get_categories

def main():
    """
    Main entry point for Milestone 2 (Before Version) of SmartBudgetPlanner.
    
    In Milestone 2, the application is extended to support two budgeting strategies:
      1. Smart Tiered Budgeting (Emergency fund-aware)
      2. Priority-Based Budgeting (Dynamic Programming)
    
    Steps:
      1. Display a menu to allow the user to select a budgeting strategy.
      2. Retrieve total monthly income and spending categories (Milestone 1 functionality).
      3. Calculate the sum of essential expenses.
      4. Based on the chosen strategy, prompt for additional parameters:
         - For Tiered Budgeting: Prompt for emergency fund target and current emergency fund balance.
         - For Priority-Based Budgeting: Prompt for an optional priority for each category and a savings goal percentage.
      5. Compute the remaining funds after essential expenses.
      6. Compute the allocation using the corresponding budgeting function.
      7. Build and display a textual breakdown of the final budget.
    
    Replace each placeholder ("Write your code here") with your own implementation.
    """
    print("=== SmartBudgetPlanner - Milestone 2 (Before Version) ===")
    print("Choose your budgeting strategy:")
    print("1. Smart Tiered Budgeting (Emergency fund-aware)")
    print("2. Priority-Based Budgeting (Dynamic Programming)")
    
    # Step 1: Get the user's choice.
    choice = input("Enter 1 or 2: ").strip()
    
    # Step 2: Retrieve total monthly income and spending categories.
    total_income = get_total_budget()
    categories = get_categories()
    # Calculate the sum of essential expenses.
    essential_total = sum(cat["Minimum Expense"] for cat in categories)
    
    if choice == "1":
        # Strategy 1: Smart Tiered Budgeting.
        # Step 3a: Prompt the user for their emergency fund target.
        # Write your code here.
        emergency_target = None  # Write your code here.
        
        # Step 4a: Prompt the user for their current emergency fund balance.
        # Write your code here.
        current_emergency = None  # Write your code here.
        
        # Step 5a: Use the smart_allocation function to compute the allocation.
        from budget.tiered_planner import smart_allocation
        allocation = smart_allocation(total_income, essential_total, emergency_target, current_emergency)
        
        # Step 6a: Build a textual breakdown of the budget.
        # Start with essential expenses.
        # Write your code here to build a list (or table) of categories and amounts.
        pass  # Write your code here.
        
    elif choice == "2":
        # Strategy 2: Priority-Based Budgeting.
        # Step 3b: Prompt the user for an optional priority for each category.
        print("\nFor each category, enter an optional priority (default is 1.0).")
        for cat in categories:
            # Write your code here to prompt for priority and update the category dictionary.
            pass  # Write your code here.
        
        # Step 4b: Ask the user for the percentage of leftover funds they want to save.
        # Write your code here.
        savings_percent = None  # Write your code here.
        
        # Step 5b: Calculate leftover funds after essential expenses.
        # Write your code here.
        leftover = None  # Write your code here.
        
        # Step 6b: Compute the savings amount based on the savings percentage.
        # Write your code here.
        savings_amount = None  # Write your code here.
        
        # Step 7b: Calculate the funds available for dynamic programming allocation.
        # Write your code here.
        budget_for_dp = None  # Write your code here.
        
        # Step 8b: Use the optimize_budget function to compute the allocation.
        from budget.dp_planner import optimize_budget
        result = optimize_budget(essential_total + budget_for_dp, categories)
        
        # Step 9b: Build a textual breakdown of the budget including the optimization results and savings.
        # Write your code here.
        pass  # Write your code here.
        
    else:
        print("Invalid choice. Exiting.")
    
if __name__ == "__main__":
    main()
