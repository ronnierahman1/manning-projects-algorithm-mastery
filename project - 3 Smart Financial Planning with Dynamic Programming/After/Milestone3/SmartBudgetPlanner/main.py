# Import necessary functions from the project's modules.
# - get_total_budget and get_categories handle user inputs.
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
    # Create a DataFrame from the allocation data (each row is a dictionary).
    df = pd.DataFrame(rows)
    # Format the "Amount" column for display (e.g., "$1,000.00").
    df["Amount"] = df["Amount"].map("${:,.2f}".format)
    
    # Sum the raw numeric values to compute the total allocated funds.
    total_allocated = sum(row["AmountRaw"] for row in rows)
    
    # Print the detailed budget breakdown.
    print("\n=== Full Budget Breakdown ===")
    print(df[["Category", "Amount"]].to_string(index=False))
    print(f"\nTotal Allocated: ${total_allocated:,.2f}")
    print(f"Total Income:    ${total_income:,.2f}")
    
    # Compare the allocated total with the income and print a confirmation message.
    if abs(total_allocated - total_income) < 0.01:
        print("✅ Budget is balanced.")
    else:
        print("⚠️ Budget mismatch. Check your calculations.")

def main():
    """
    Main entry point of the SmartBudgetPlanner application.
    
    The program performs the following steps:
      1. Displays a menu allowing the user to select between two budgeting strategies:
         a. Smart Tiered Budgeting (emergency fund-aware)
         b. Priority-Based Budgeting (dynamic programming)
      2. Collects the total monthly income and spending categories with minimum expenses.
      3. Based on the selected strategy, it further prompts for additional parameters:
         - For Tiered Budgeting: emergency fund target and current emergency balance.
         - For Priority-Based Budgeting: an optional priority for each category and a savings goal percentage.
      4. Computes the remaining funds after essential expenses.
      5. For Tiered Budgeting, it allocates the remaining funds using fixed percentages.
      6. For Priority-Based Budgeting, it first secures a portion for savings, then optimizes the remaining funds using dynamic programming.
      7. Displays the full budget breakdown as a text table.
      8. Plots a bar chart visualizing the final allocation.
    """
    # Display the main title and options for budgeting strategies.
    print("=== SmartBudgetPlanner ===")
    print("Choose your budgeting strategy:")
    print("1. Smart Tiered Budgeting (Emergency fund-aware)")
    print("2. Priority-Based Budgeting (Dynamic Programming)")

    # Get the user's choice as a string.
    choice = input("Enter 1 or 2: ").strip()

    # Retrieve the total monthly income using a helper function.
    total_income = get_total_budget()
    # Retrieve spending categories (each with a minimum required expense).
    categories = get_categories()
    # Calculate the sum of all essential expenses.
    essential_total = sum(cat["Minimum Expense"] for cat in categories)

    # Handle the first strategy: Smart Tiered Budgeting.
    if choice == "1":
        # Prompt the user for their emergency fund target and current emergency balance.
        emergency_target = float(input("Enter your Emergency Fund target: "))
        current_emergency = float(input("Enter your Current Emergency Fund balance: "))

        # Use the smart_allocation function to decide how to split the remaining funds.
        allocation = smart_allocation(total_income, essential_total, emergency_target, current_emergency)

        # Build a list of rows to represent the budget breakdown:
        # Start with the essential expenses.
        rows = [{"Category": cat["Category"], "Amount": cat["Minimum Expense"], "AmountRaw": cat["Minimum Expense"]}
                for cat in categories]
        # Then add the allocations (e.g., Emergency Fund and Miscellaneous).
        for key, value in allocation.items():
            rows.append({"Category": key, "Amount": value, "AmountRaw": value})

        # Display the complete budget breakdown in text.
        display_budget_breakdown(rows, total_income)
        # Plot the budget breakdown as a bar chart.
        plot_budget_breakdown(rows, title="Smart Tiered Budget Allocation")

    # Handle the second strategy: Priority-Based Budgeting.
    elif choice == "2":
        # Prompt the user for an optional priority for each category.
        print("\nFor each category, enter an optional priority (default is 1.0).")
        for cat in categories:
            try:
                value = input(f"Priority for {cat['Category']} (default 1.0): ").strip()
                # Set the priority: use the user input or default to 1.0 if input is blank.
                cat['Priority'] = float(value) if value else 1.0
            except ValueError:
                # If an invalid input is provided, use the default priority.
                print("Invalid input, using default priority 1.0")
                cat['Priority'] = 1.0

        # Ask the user for the percentage of leftover funds they want to save.
        while True:
            try:
                savings_percent = float(input("\nEnter the % of leftover you want to save (e.g. 20 for 20%): "))
                # Ensure the percentage is between 0 and 100.
                if 0 <= savings_percent <= 100:
                    break
                else:
                    print("Please enter a value between 0 and 100.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        # Calculate the leftover funds after essential expenses.
        leftover = total_income - essential_total
        # Compute the amount to be saved based on the user's specified percentage.
        savings_amount = round((savings_percent / 100) * leftover, 2)
        # The remaining funds for category optimization are the leftover minus the savings amount.
        budget_for_dp = leftover - savings_amount

        # Display the computed amounts for user feedback.
        print(f"\nTotal leftover after essentials: ${leftover:,.2f}")
        print(f"Savings Goal ({savings_percent}%): ${savings_amount:,.2f}")
        print(f"Budget available for category optimization: ${budget_for_dp:,.2f}")

        # Use the optimize_budget function (dynamic programming) on the available funds,
        # adding the essential total to budget_for_dp to obtain the full amount for categories.
        result = optimize_budget(essential_total + budget_for_dp, categories)

        # Build the rows for the budget breakdown from the optimization results.
        rows = []
        for item in result:
            rows.append({
                "Category": item["Category"],
                "Amount": item["Total Allocated"],
                "AmountRaw": item["Total Allocated"]
            })

        # Append the Savings row to the breakdown.
        rows.append({
            "Category": "Savings",
            "Amount": savings_amount,
            "AmountRaw": savings_amount
        })

        # Display the final budget breakdown.
        display_budget_breakdown(rows, total_income)
        # Plot the final budget breakdown as a bar chart.
        plot_budget_breakdown(rows, title="Priority-Based Budget Allocation")

    else:
        # If the user enters an invalid choice, notify and exit.
        print("Invalid choice. Exiting.")

# This ensures that main() is executed when the script is run directly.
if __name__ == "__main__":
    main()
