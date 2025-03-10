import streamlit as st
import asyncio
from temporalio.client import Client
from rojak import Rojak
from rojak.agents.agent import ResumeRequest, ResumeResponse
from rojak.workflows.orchestrator_workflow import OrchestratorResponse, TaskParams

# Import your pizza chatbot agent; you can adapt this to your agent name and settings.
from examples.streamlit_pizza.agents import triage_agent

# Use a constant session id so that conversation history is maintained across runs.
SESSION_ID = "session_pizza_4"


# Helper to run async coroutines synchronously (Streamlit scripts are synchronous)
def run_async(coro):
    return asyncio.run(coro)


##############################
# Async functions for Rojak calls
##############################


async def generate_response(rojak: Rojak, user_message: str, agent):
    """
    Calls rojak.run() to process a user message.
    Returns either an OrchestratorResponse or a ResumeRequest if human intervention is needed.
    """
    response = await rojak.run(
        id=SESSION_ID,
        type="persistent",
        task=TaskParams(
            messages=[{"role": "user", "content": user_message}],
            agent=agent,
        ),
        # Example context variables for the pizza chatbot
        context_variables={
            "name": "John",  # Example customer name
            "cart": st.session_state.get(
                "cart", {}
            ),  # Use the current cart if available
            "preferences": "Loves extra cheese and pepperoni.",
        },
        debug=True,
    )
    return response


async def generate_resume(
    rojak: Rojak, action: str, tool_id: str, content: str | None = None
):
    """
    Sends a ResumeResponse (approve/reject) back to rojak.
    """
    return await rojak.run(
        id=SESSION_ID,
        resume=ResumeResponse(action=action, tool_id=tool_id, content=content),
    )


async def process_resume(rojak: Rojak, resume_req: ResumeRequest):
    """
    Displays an interrupt message for a pending tool call and lets the user choose to approve or reject it.
    Returns the new run response after sending a ResumeResponse.
    """
    with st.chat_message("assistant"):
        st.markdown("An action requires your approval")
        st.markdown(f"**Name:** {resume_req.tool_name}")
        st.markdown(f"**Arguments:** `{resume_req.tool_arguments}`")
        question = resume_req.question or "Do you approve this tool call?"
        st.markdown(f"**Question:** {question}")

        # Allow the user to optionally enter a reason if rejecting the tool call
        reject_reason = st.text_input(
            "Reason for rejection (optional):", key=f"rej_{resume_req.tool_id}"
        )

        col1, col2 = st.columns(2)
        with col1:
            approve_clicked = st.button("Approve", key=f"approve_{resume_req.tool_id}")
        with col2:
            reject_clicked = st.button("Reject", key=f"reject_{resume_req.tool_id}")

        if approve_clicked:
            new_resp = await generate_resume(
                rojak, action="approve", tool_id=resume_req.tool_id
            )
            return new_resp
        if reject_clicked:
            new_resp = await generate_resume(
                rojak,
                action="reject",
                tool_id=resume_req.tool_id,
                content=reject_reason,
            )
            return new_resp

    return None  # If neither button was clicked


##############################
# Initialization & Session State Setup
##############################


def initialise():
    """
    Initializes the Rojak client, the pizza agent, conversation state, and cart in st.session_state.
    """
    if "rojak" not in st.session_state:
        # Connect to your Temporal server (make sure it is running)
        client = asyncio.run(Client.connect("localhost:7233"))
        st.session_state.rojak = Rojak(client, task_queue="tasks")
    if "agent" not in st.session_state:
        st.session_state.agent = triage_agent  # your pizza chatbot agent instance
    if "messages" not in st.session_state:
        st.session_state.messages = []  # Will hold conversation history as a list of dicts
    if "pending_resume" not in st.session_state:
        st.session_state.pending_resume = None  # To store any pending ResumeRequest
    if "cart" not in st.session_state:
        st.session_state.cart = {}  # Initialize an empty cart


##############################
# Main Streamlit App
##############################


def main():
    st.set_page_config(page_title="🍕 Pizza Chatbot", layout="centered")
    initialise()
    rojak: Rojak = st.session_state.rojak
    agent = st.session_state.agent
    messages = st.session_state.messages

    st.title("🍕 Pizza Chatbot")
    st.caption("Chat with our pizza ordering assistant!")

    # Sidebar: Show the current contents of the cart
    st.sidebar.markdown("# 🍕 Pizza Chatbot")
    st.sidebar.markdown("## Chat with our pizza ordering assistant!")
    st.sidebar.header("🛒 Your Cart")
    cart = st.session_state.get("cart", {})
    if cart:
        # Render the cart as a list of items; you can customize the formatting
        for item, qty in cart.items():
            st.sidebar.write(f"**{item}**: {qty}")
    else:
        st.sidebar.write("Your cart is empty.")

    # Display conversation history using Streamlit's chat UI elements
    for msg in messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # If there is a pending ResumeRequest (human-in-the-loop interrupt), process it first
    if st.session_state.pending_resume:
        new_resp = run_async(process_resume(rojak, st.session_state.pending_resume))
        if new_resp is not None:
            if isinstance(new_resp.result, ResumeRequest):
                # Still needs human intervention
                st.session_state.pending_resume = new_resp.result
            elif isinstance(new_resp.result, OrchestratorResponse):
                # Update conversation history and update the current agent
                assistant_reply = new_resp.result.messages[-1].content
                messages.append({"role": "assistant", "content": assistant_reply})
                st.session_state.agent = new_resp.result.agent  # update current agent
                # Update the cart if it's present in the context_variables
                st.session_state.cart = new_resp.result.context_variables.get(
                    "cart", {}
                )
                st.session_state.pending_resume = None
            st.rerun()  # Rerun to update the UI

    # Chat input: get new user message
    user_input = st.chat_input("Type your message here...")
    if user_input:
        # Append the user's message to conversation history and display it
        messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # Send the message to the Rojak orchestrator
        run_resp = run_async(generate_response(rojak, user_input, agent))
        if isinstance(run_resp.result, ResumeRequest):
            st.session_state.pending_resume = run_resp.result
        elif isinstance(run_resp.result, OrchestratorResponse):
            # Normal response: update the conversation with the assistant's reply
            assistant_reply = run_resp.result.messages[-1].content
            messages.append({"role": "assistant", "content": assistant_reply})
            # Update the current agent so that future messages are sent to the new agent if a handoff occurred
            st.session_state.agent = run_resp.result.agent
            # Update the cart using the context_variables returned by the orchestrator
            st.session_state.cart = run_resp.result.context_variables.get("cart", {})
        st.rerun()  # Rerun to update the UI


if __name__ == "__main__":
    main()
