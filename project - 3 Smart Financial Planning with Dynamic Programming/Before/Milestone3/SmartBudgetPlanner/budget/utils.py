import matplotlib.pyplot as plt

def round_to_cents(value):
    """
    Rounds the given numeric value to two decimal places (i.e., to cents).
    
    Args:
        value (float): The numeric value to be rounded.
    
    Returns:
        float: The rounded value, accurate to two decimal places.
    
    Example:
        If value is 123.456, the function returns 123.46.
    """
    # Use Python's built-in round() function to round the value to 2 decimal places.
    return round(value, 2)


def plot_budget_breakdown(rows, title="Budget Allocation"):
    """
    Plots a bar chart representing the final budget breakdown.
    
    This function takes a list of dictionaries, where each dictionary represents 
    a budget category. Each dictionary must contain:
      - "Category": The name of the spending category (string).
      - "AmountRaw": The numerical amount allocated to that category.
    
    Steps:
      1. Extract the category names from the input list.
      2. Extract the corresponding allocated amounts.
      3. Create a new figure with a specific size.
      4. Plot a bar chart with category names on the x-axis and allocated amounts on the y-axis.
      5. Set a title, as well as x and y labels, to clearly describe the chart.
      6. Rotate the x-axis labels for better readability.
      7. Add grid lines on the y-axis to enhance visual clarity.
      8. Adjust the layout to prevent any overlapping elements.
      9. Finally, display the plot.
    
    Args:
        rows (list of dict): List of dictionaries containing budget data.
        title (str): Title for the plot. Defaults to "Budget Allocation".
    """
    # Step 1: Extract category names from the rows.
    categories = [row["Category"] for row in rows]
    
    # Step 2: Extract the allocation amounts (numeric values) for each category.
    amounts = [row["AmountRaw"] for row in rows]
    
    # Step 3: Create a figure with a specified size (10 inches wide and 5 inches tall).
    plt.figure(figsize=(10, 5))
    
    # Step 4: Plot a bar chart with categories on the x-axis and amounts on the y-axis.
    plt.bar(categories, amounts)
    
    # Step 5: Set the title of the chart.
    plt.title(title)
    
    # Step 6: Label the x-axis as "Category" and the y-axis as "Amount ($)".
    plt.xlabel("Category")
    plt.ylabel("Amount ($)")
    
    # Step 7: Rotate the x-axis labels by 30 degrees to improve readability.
    plt.xticks(rotation=30)
    
    # Step 8: Add grid lines on the y-axis using dashed lines with some transparency.
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Step 9: Adjust the layout to ensure no parts of the plot overlap.
    plt.tight_layout()
    
    # Step 10: Display the final plot.
    plt.show()
