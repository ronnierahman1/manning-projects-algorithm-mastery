import math
from functools import lru_cache
import matplotlib.pyplot as plt

def optimize_budget(total_budget, categories):
    """
    Optimizes allocation of extra funds (beyond the minimum expenses) using dynamic programming.
    
    The function distributes any funds left after covering the minimum expenses across various
    spending categories, with extra funds allocated based on each category's assigned priority.
    A diminishing returns utility function (utility = Priority × (1 - exp(-x))) is used to
    balance the allocation: high-priority categories receive extra funds more effectively.
    
    Args:
        total_budget (float): Total available budget (income).
        categories (list of dict): A list where each dictionary represents a spending category.
            Each dictionary must include:
              - 'Category': Name of the category (string)
              - 'Minimum Expense': The minimum required expense (float)
              - 'Priority': A numeric value indicating the importance of extra allocation (optional, defaults to 1.0)
              
    Returns:
        list of dict: A list of dictionaries for each category, where each dictionary includes:
            - 'Category': Name of the category.
            - 'Minimum Expense': The minimum required expense.
            - 'Extra Allocation': The extra funds allocated (beyond the minimum).
            - 'Total Allocated': The sum of the minimum expense and the extra allocation.
    """

    # Step 1: Ensure each category has a "Priority" value.
    # If a category does not have a Priority key, set it to the default value 1.0.
    for cat in categories:
        if 'Priority' not in cat:
            cat['Priority'] = 1.0

    # Step 2: Calculate the total of the minimum expenses across all categories.
    min_total = sum(cat['Minimum Expense'] for cat in categories)
    
    # Step 3: Calculate the extra funds available after covering the minimum expenses.
    # We convert the remaining funds to an integer to work in whole dollars.
    remaining = int(total_budget - min_total)

    # Step 4: Check if there are enough funds to cover the minimum expenses.
    # If the extra funds are negative, it means the total budget is insufficient.
    if remaining < 0:
        raise ValueError("Total budget is less than required minimum expenses.")

    # n is the number of categories we need to allocate extra funds to.
    n = len(categories)

    # Step 5: Define a recursive dynamic programming function using memoization.
    # This function, dp(i, rem), computes the maximum achievable utility from category i onwards,
    # given rem extra dollars left to allocate.
    @lru_cache(maxsize=None)
    def dp(i, rem):
        """
        Recursive function to compute maximum utility from categories[i:] with rem extra dollars.
        
        Args:
            i (int): Current index of the category being processed.
            rem (int): The remaining extra dollars available for allocation.
            
        Returns:
            float: The maximum total utility achievable.
                   Returns -math.inf if no valid allocation is possible.
        """
        # Base case: If all categories have been processed.
        # If no extra funds remain, the total utility is 0. Otherwise, it's an invalid allocation.
        if i == n:
            return 0 if rem == 0 else -math.inf

        best = -math.inf  # Initialize the best (maximum) utility to negative infinity.
        # Try all possible extra allocations (x dollars) for the current category.
        for x in range(rem + 1):
            # Calculate the utility for allocating x extra dollars using the diminishing returns function.
            # The utility function is: Priority * (1 - exp(-x))
            utility = categories[i]['Priority'] * (1 - math.exp(-x))
            # Recursively compute the total utility: current utility plus the utility of optimally allocating the remaining funds.
            total_util = utility + dp(i + 1, rem - x)
            # Update the best utility if the current allocation yields a higher value.
            if total_util > best:
                best = total_util
        return best

    # Step 6: Backtracking to reconstruct the actual allocation decision for each category.
    # Starting with the total extra funds (remaining), we iterate through each category and determine
    # the optimal extra allocation based on the DP function's computed values.
    result = []  # This will store the final allocation for each category.
    rem = remaining  # Initialize remaining funds for allocation.
    for i in range(n):
        best_util = -math.inf  # To track the best utility for the current category.
        best_x = 0  # To track the optimal extra dollars to allocate for the current category.
        # Try all possible extra dollar allocations from 0 to the current remaining funds.
        for x in range(rem + 1):
            # Calculate the utility for allocating x extra dollars for category i.
            utility = categories[i]['Priority'] * (1 - math.exp(-x))
            # Add the utility obtained from optimally allocating the remaining funds (using dp function).
            total_util = utility + dp(i + 1, rem - x)
            # If this allocation yields a better total utility, update our best choice.
            if total_util > best_util:
                best_util = total_util
                best_x = x

        # Record the allocation details for the current category:
        # The extra allocation is best_x, and the total allocated is the minimum expense plus best_x.
        result.append({
            "Category": categories[i]["Category"],
            "Minimum Expense": categories[i]["Minimum Expense"],
            "Extra Allocation": best_x,
            "Total Allocated": categories[i]["Minimum Expense"] + best_x
        })
        # Deduct the allocated extra dollars from the remaining funds.
        rem -= best_x

    # Step 7: Return the final list of allocations.
    return result

def plot_allocation(allocation):
    """
    Displays a bar chart of total allocated amounts per category using matplotlib.
    
    Args:
        allocation (list of dict): Each dictionary in the list should have at least:
            - "Category": The name of the spending category.
            - "Total Allocated": The sum of the minimum expense and extra funds allocated.
    
    Steps:
      1. Extract the category names and corresponding total allocated amounts.
      2. Create a bar chart using plt.bar().
      3. Set the title and axis labels for clarity.
      4. Add grid lines for readability.
      5. Use plt.tight_layout() to ensure the layout is neat.
      6. Display the plot using plt.show().
    """
    # Extract category names from the allocation list.
    categories = [item["Category"] for item in allocation]
    # Extract the total allocated amounts for each category.
    totals = [item["Total Allocated"] for item in allocation]

    # Create a new figure with a specified size.
    plt.figure(figsize=(8, 5))
    # Create a bar chart with the categories on the x-axis and allocated amounts on the y-axis.
    plt.bar(categories, totals)
    # Set the title of the plot.
    plt.title("Optimized Budget Allocation")
    # Label the y-axis.
    plt.ylabel("Total Allocated ($)")
    # Label the x-axis.
    plt.xlabel("Category")
    # Add a grid on the y-axis with dashed lines and some transparency.
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    # Adjust the layout so that everything fits without overlap.
    plt.tight_layout()
    # Display the plot.
    plt.show()
