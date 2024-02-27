import streamlit as st
from model_predictor import generate_text  # Import your model's text generation function
st.set_page_config(
    page_title="Shakespeare Language Model",
    page_icon = ":performing_arts:"
)

page = st.radio("Choose a page:",
        ["Demo", "Explanation"], captions = ["Small demo app.","Written explanation on the creation of the model."])

if page =="Demo":
    # Set the title of your app
    st.title('🎭 Shakespearean Text Generator')

    # Create a text input for users to enter a seed text
    seed_text = st.text_input("Enter a seed text to inspire the Shakespearean text generator:", placeholder="To be or not to be")
    length = st.slider("How many characters would you like to generate:", 10,500,10)

    model_path = 'tiny_shakespeare_model.pkl'

    # Text generation button
    if st.button('Generate Text'):
        if seed_text:
            # Generate text using your model
            with st.spinner("📝Writing text..."):
                generated_text = generate_text(seed_text,model_path,max_length=length)
                # Display the generated text
                st.write(generated_text)
        else:
            st.write("Please enter some seed text to generate Shakespearean text.")

elif page =="Explanation":
    st.write("## The process I took to create the Tiny Shakespeare model.")
