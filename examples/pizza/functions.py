from .agents import (
    triage_agent,
    food_order_agent,
    payment_agent,
    feedback_agent,
)
from copy import deepcopy


def food_ordering_instructions(context_variables):
    return f"""
    You are the Food Ordering Agent. Your role is to assist customers in selecting and managing their food orders.
    - **Recommendations**: Start by retrieving the available menu using the `get_menu` function. Guide customers in choosing items based on their preferences or suggest popular options. Provide clear and concise descriptions of menu items.
    - **Adding to Order**: When customers request to add items, confirm their selection against the menu. Use the `get_menu` function to ensure the item is available. If the item exists, proceed to add it using the `add_to_cart` function. For unavailable items, politely inform the customer and suggest alternatives from the menu.
    - **Order Management**: Customers may also want to modify their order. Use the `get_cart` function to show the current order details and confirm requested changes. Use `remove_from_cart` to delete specific items as needed and provide an updated summary of the cart.
    - **Guiding and Redirecting**: If customers are unsure of what they want, offer assistance by highlighting popular dishes or providing recommendations. Redirect customers to the Triage Agent if they need help beyond ordering or recommendations.
    - **Tone and Accuracy**: Maintain a polite and professional tone, ensuring order details are accurate at every step. Always summarize the current state of the cart after any action.
    Redirect customers to the Triage Agent if they need help with non-ordering tasks.
    Remember: Customers can only order items listed in the menu. Validate all item requests before proceeding with any order changes.
    Customer Preferences: {context_variables.get("preferences", "N/A")}
    """


def to_food_order():
    """Route to the food order agent."""
    return food_order_agent


def to_payment():
    """Route to the payment agent."""
    return payment_agent


def to_feedback():
    """Route to the feedback agent."""
    return feedback_agent


def to_triage():
    """Route to the triage agent."""
    return triage_agent


def get_menu():
    """Return a list of menu items for a pizza place."""
    menu = [
        {"name": "Margherita Pizza", "cost": 12.99},
        {"name": "Pepperoni Pizza", "cost": 14.49},
        {"name": "BBQ Chicken Pizza", "cost": 15.99},
        {"name": "Veggie Supreme Pizza", "cost": 13.99},
        {"name": "Meat Lovers Pizza", "cost": 16.99},
        {"name": "Garlic Knots", "cost": 5.99},
        {"name": "Cheesy Breadsticks", "cost": 6.49},
        {"name": "Classic Caesar Salad", "cost": 9.49},
        {"name": "Buffalo Wings (8 pieces)", "cost": 10.99},
        {"name": "Mozzarella Sticks", "cost": 7.99},
        {"name": "Chocolate Chip Cannoli", "cost": 4.99},
        {"name": "Tiramisu Slice", "cost": 5.99},
        {"name": "Fountain Soda", "cost": 2.49},
        {"name": "Sparkling Lemonade", "cost": 3.49},
        {"name": "Iced Tea", "cost": 2.99},
        {"name": "Craft Beer (Pint)", "cost": 6.99},
        {"name": "House Red Wine (Glass)", "cost": 7.99},
    ]
    return menu


def add_to_cart(item: str, cost: int, quantity: int, context_variables: dict):
    """Add an item to the cart."""
    if item in context_variables["cart"]:
        context_variables["cart"][item]["quantity"] += quantity
    else:
        context_variables["cart"][item] = {"quantity": quantity, "unit_cost": cost}
    return f"Added {quantity} of {item} to your cart."


def remove_from_cart(item: str, quantity: int, context_variables: dict):
    """Remove an item from the cart by quantity."""
    if item in context_variables["cart"]:
        if quantity >= context_variables["cart"][item]["quantity"]:
            del context_variables["cart"][item]
            return f"Removed {item} from your cart."
        else:
            context_variables["cart"][item]["quantity"] -= quantity
            return f"Decreased quantity of {item} by {quantity}."
    else:
        return f"{item} is not in your cart."


def get_cart(context_variables: dict):
    """Return the contents of the cart."""
    cart = context_variables.get("cart", {})
    if cart:
        cart_items = [
            f"{item} (x{details['quantity']}) - ${details['quantity'] * details['unit_cost']:.2f}"
            for item, details in cart.items()
        ]
        return f"Your cart contains: {', '.join(cart_items)}"
    else:
        return "Your cart is empty."


def process_payment(context_variables: dict):
    """Process the payment and clear the cart."""
    context_variables["receipt"] = deepcopy(context_variables["cart"])
    context_variables["cart"] = {}
    return "Payment processed successfully!"


def get_receipt(context_variables: dict):
    """Return the receipt."""
    receipt = context_variables.get("receipt", {})
    if receipt:
        receipt_items = [
            f"{item} (x{details['quantity']}) - ${details['quantity'] * details['unit_cost']:.2f}"
            for item, details in receipt.items()
        ]
        return f"Your receipt contains: {', '.join(receipt_items)}"
    else:
        return "No receipt available."


def provide_feedback():
    """Submit a review."""
    return "Review submitted successfully!"
