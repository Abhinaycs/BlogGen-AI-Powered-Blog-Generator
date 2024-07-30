import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from huggingface_hub import login

# Replace 'your_huggingface_token_here' with the token you just created
token = "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"
login(token)

## Function to get response from LLaMA 2 model
def getLLamaResponse(input_text, no_words, blog_style):
    try:
        # Example model path; use a correct and accessible model path
        llm = CTransformers(
            model='',  # Adjust this path based on your model
            model_type='llama',
            config={'max_new_tokens': 256, 'temperature': 0.01}
        )
        
        # Prompt Template
        template = """
        Write a blog for {blog_style} job profile for a topic {input_text} within {no_words} words.
        """
        
        prompt = PromptTemplate(
            input_variables=["blog_style", "input_text", "no_words"],
            template=template
        )
        
        # Generate the response from the LLaMA 2 model
        formatted_prompt = prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words)
        response = llm(formatted_prompt)
        
        return response
    
    except Exception as e:
        return str(e)

st.set_page_config(page_title="Generate Blogs", page_icon='🤖', layout='centered', initial_sidebar_state='collapsed')
st.header("Generate Blogs 🤖")

input_text = st.text_input("Enter the Blog Topic")

# Creating two more columns for additional 2 fields
col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('Number of Words')
with col2:
    blog_style = st.selectbox('Writing the blog for', ('Researchers', 'Data Scientist', 'Common People'), index=0)
    
submit = st.button("Generate")

# Final response
if submit:
    response = getLLamaResponse(input_text, no_words, blog_style)
    st.write(response)
