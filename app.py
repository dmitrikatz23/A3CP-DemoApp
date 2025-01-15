import streamlit as st

# Set up the page configuration
st.set_page_config(page_title="A3CP App", page_icon="🌟", layout="centered")

# Display the title and information
st.title("Welcome to the A3CP App")
st.markdown(
    """
    ### About the A3CP App
    The A3CP App is your gateway to advanced assistive communication and collaboration tools. 
    Explore features designed to enhance accessibility and communication for all users.
    """
)

st.info("👈 Use the sidebar to navigate through the app's features!")

# Add a footer or additional details if needed
st.markdown(
    """
    ---
    🛠️ **Built with [Streamlit](https://streamlit.io)** | 🌐 **Designed for Assistive Technology**
    """
)
