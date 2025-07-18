def get_total_budget():
    """
    Prompt the user to enter their total monthly income and ensure it is a valid positive number.
    
    Steps:
      1. Continuously prompt the user until a valid input is received.
      2. Try converting the input into a float.
      3. If the conversion succeeds, check that the number is greater than 0.
      4. If it is, return that number as the total budget.
      5. If the number is less than or equal to 0, inform the user and prompt again.
      6. If the conversion fails (invalid input), catch the ValueError and prompt again.
    """
    while True:
        try:
            # Prompt the user for their total monthly income and convert the input to a float.
            amount = float(input("Enter your total monthly income: "))
            # Check if the entered amount is positive.
            if amount > 0:
                # Valid income entered; return the amount.
                return amount
            # If the income is not positive, notify the user.
            print("Income must be greater than 0.")
        except ValueError:
            # If conversion to float fails, notify the user to enter a numeric value.
            print("Invalid input. Enter a numeric value.")


def get_categories():
    """
    Collect spending categories and their minimum required expenses from the user.
    
    Steps:
      1. Initialize an empty list to hold the category dictionaries.
      2. Start an infinite loop to prompt for category names.
      3. For each category:
          a. Prompt the user to enter a category name.
          b. If the user enters a blank string, exit the loop (finish input).
          c. Otherwise, prompt the user for the minimum expense associated with that category.
      4. Attempt to convert the expense input to a float.
          a. If successful, append a dictionary with the category name and minimum expense to the list.
          b. If the conversion fails (ValueError), inform the user and do not add the category.
      5. Return the list of category dictionaries after the user indicates they are finished.
    """
    categories = []  # Initialize an empty list to store category data.
    
    while True:
        # Prompt for the category name.
        name = input("Category name (leave blank to finish): ").strip()
        # If the user did not enter a name, exit the loop.
        if not name:
            break
        try:
            # Prompt for the minimum expense associated with this category.
            min_expense = float(input(f"Enter minimum expense for '{name}': "))
            # If the expense input is valid, add a dictionary with the category details to the list.
            categories.append({"Category": name, "Minimum Expense": min_expense})
        except ValueError:
            # If the expense input is invalid (cannot convert to float), inform the user.
            print("Invalid expense. Try again.")
    
    # Return the complete list of categories and their expenses.
    return categories
