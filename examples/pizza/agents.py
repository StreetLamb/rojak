from rojak.agents import OpenAIAgent
from rojak.agents import Interrupt
from rojak.types import RetryOptions, RetryPolicy


retry_options = RetryOptions(
    retry_policy=RetryPolicy(non_retryable_error_types=["TypeError"])
)
triage_agent = OpenAIAgent(
    name="Triage Agent",
    instructions="""
    You are the Triage Agent. Your role is to assist customers by identifying their needs and routing them to the correct agent: 
    - **Food Ordering** (`to_food_order`): For menu recommendations, adding/removing items, viewing or modifying the cart.
    - **Payment** (`to_payment`): For payments, payment method queries, receipts, or payment issues.
    - **Feedback** (`to_feedback`): For reviews, ratings, comments, or complaints.
    If unsure, guide customers by explaining options (ordering, payment, feedback). For multi-step needs, start with the immediate priority and redirect after. 
    Always ensure clear, polite, and accurate communication during handoffs.
    """,
    functions=["to_food_order", "to_payment", "to_feedback"],
    tool_choice="required",
    retry_options=retry_options,
    interrupts=[
        Interrupt("to_food_order"),
        Interrupt("to_payment"),
        Interrupt("to_feedback"),
    ],
)

food_order_agent = OpenAIAgent(
    name="Food Ordering Agent",
    instructions={"type": "function", "name": "food_ordering_instructions"},
    functions=[
        "to_triage",
        "add_to_cart",
        "remove_from_cart",
        "get_cart",
        "get_menu",
    ],
    retry_options=retry_options,
)


payment_agent = OpenAIAgent(
    name="Payment Agent",
    instructions="""
    You are the Payment Agent. Your role is to securely and efficiently handle the payment process for customers. 
    Start by confirming the payment amount and presenting the available payment methods (e.g., credit card, mobile wallet, or cash). 
    Use the `process_payment` function to finalize the transaction and provide a receipt. 
    In case of a failed transaction, assist customers by suggesting alternative payment options or troubleshooting the issue. 
    Always maintain a courteous and professional tone, ensuring customers feel supported throughout the payment process. 
    Redirect customers to the Triage Agent if they need help with non-payment tasks.
    """,
    functions=[
        "to_triage",
        "process_payment",
        "get_receipt",
        "get_cart",
    ],
    retry_options=retry_options,
)

feedback_agent = OpenAIAgent(
    name="Feedback Agent",
    instructions="""
    You are the Feedback Agent. Your role is to collect and manage customer feedback to improve the overall experience. 
    Ask customers to rate their experience and provide detailed comments about the food, service, or satisfaction level. 
    Use the `provide_feedback` function to log their input.
    If the customer has complaints, acknowledge them empathetically and offer to escalate the issue to the relevant team. 
    Encourage constructive feedback and thank customers for their time and insights. 
    Redirect customers to the Triage Agent if they wish to engage with other services after providing feedback.
    """,
    functions=["to_triage", "provide_feedback"],
    retry_options=retry_options,
)
