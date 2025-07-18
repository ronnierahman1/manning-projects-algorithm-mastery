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
    # Write your code here.
    pass
