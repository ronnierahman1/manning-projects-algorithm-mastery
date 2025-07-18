def smart_allocation(total_income, essential_total, emergency_target, current_emergency_balance):
    """
    Allocate the remaining funds after essential expenses based on the emergency fund status.
    
    This function calculates the funds left after subtracting essential expenses from the total income.
    It then determines the allocation of these remaining funds according to the user's progress towards
    an emergency fund target. The allocation is done using fixed percentages:
    
    - If the current emergency fund balance is less than the emergency target:
         * 70% of the remaining funds are allocated to the "Emergency Fund".
         * 30% of the remaining funds are allocated to "Miscellaneous" expenses.
    - Otherwise, if the emergency fund target has been met or exceeded:
         * 90% of the remaining funds are allocated to "Savings/Investments".
         * 10% of the remaining funds are allocated to "Miscellaneous" expenses.
    
    The allocated amounts are rounded to two decimal places.
    
    Args:
        total_income (float): The user's total monthly income.
        essential_total (float): The total amount of essential expenses that must be paid.
        emergency_target (float): The target amount the user aims to have in their emergency fund.
        current_emergency_balance (float): The current amount in the user's emergency fund.
        
    Returns:
        dict: A dictionary with the allocation breakdown. The keys depend on whether the emergency
              target has been met:
                - If not met, keys are "Emergency Fund" and "Miscellaneous".
                - If met, keys are "Savings/Investments" and "Miscellaneous".
    
    Raises:
        ValueError: If the essential expenses exceed the total income (i.e., remaining funds are negative).
    """
    
    # Calculate the remaining funds after paying essential expenses.
    remaining = total_income - essential_total
    
    # If remaining funds are negative, raise an error because income is insufficient.
    if remaining < 0:
        raise ValueError("Essential expenses exceed income.")
    
    # Determine the allocation strategy based on the emergency fund status.
    if current_emergency_balance < emergency_target:
        # Emergency fund is not fully funded:
        # Allocate 70% of the remaining funds to the Emergency Fund,
        # and allocate 30% to Miscellaneous expenses.
        allocation = {
            "Emergency Fund": round(remaining * 0.70, 2),
            "Miscellaneous": round(remaining * 0.30, 2)
        }
    else:
        # Emergency fund target is met:
        # Allocate 90% of the remaining funds to Savings/Investments,
        # and allocate 10% to Miscellaneous expenses.
        allocation = {
            "Savings/Investments": round(remaining * 0.90, 2),
            "Miscellaneous": round(remaining * 0.10, 2)
        }
    
    # Return the dictionary containing the allocated amounts.
    return allocation
