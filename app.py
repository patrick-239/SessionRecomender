import streamlit as st
from openai import OpenAI
import os
from streamlit_extras.stylable_container import stylable_container
from dotenv import load_dotenv

#Env parameters
load_dotenv()
model_chat_url = os.environ.get('MODEL_CHAT_URL')
openai_api_key = os.environ.get('OPENAI_API_KEY')

#CSS for background
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://storage.ai.nebius.cloud/public-demo/session-recommender/back.png");
background-size: cover;
}

[data-testid="stHeader"] {
background: rgba(0,0,0,0);
}
</style>
"""

#Clients init
chat_client = OpenAI(
    api_key=openai_api_key,
    base_url=model_chat_url,
)

def get_context(prompt):
    #docs = db.similarity_search_with_relevance_scores(prompt, k=5)
    #concatenated_texts = ' '.join([dict(doc[0])['page_content'] for doc in docs])
    #formatted_prompt = prompt_template.format(content=concatenated_texts)
    f = open("./sessions_sun.txt", "r")
    context =  f.read()
    return context

#Main function
def main():
    st.set_page_config(
        page_title="Nebius Sessions Recommender",
        page_icon=":robot:",
    )
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>Sessions Recommender</h1>", unsafe_allow_html=True)

    #get list of models
    models_list = chat_client.models.list().to_dict()
    models_ids =  [model["id"] for model in models_list["data"]]

    #sidebar
    st.sidebar.header('Model Parameters')
    model_chat = st.sidebar.selectbox('Slect model',models_ids)
    temperature = st.sidebar.slider(
        label = 'Temperature', 
        min_value = 0.00,
        max_value = 2.00,
        value = 0.00,
        step = 0.01,
        help = 'What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.'
    )
    maxTokens = st.sidebar.slider(
        label = 'Max Tokens',
        min_value=0,
        max_value=1000,
        value = 0,
        step = 1,
        help='The maximum number of tokens that can be generated in the chat completion.'
        )
    presencePenalty = st.sidebar.slider(
        label = 'Presence Penalty ',
        min_value= -2.00,
        max_value= 2.00, 
        value = 0.00,
        step = 0.01,
        help = 'Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model likelihood to talk about new topics.'
        )
    topP = st.sidebar.slider(
        label = 'Top P',
        min_value=0.00,
        max_value=1.00, 
        value = 0.00,
        step = 0.01,
        help = 'An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.'
        )
    with stylable_container(
        key = 'messages',
        css_styles = """
            [data-testid="stVerticalBlockBorderWrapper"]{
            background-color: rgba(255, 255, 255, 1);
            }
            [data-testid="stChatMessage"]{
            background-color: rgba(255, 255, 255, 1);
            }
            """
    ):
        messages_cont = st.container(border=True)
        
    #Chat
    messages_cont.chat_message('assistant').write('Hello, I am the conference sessions recommender. Could you tell me about your interests for today?')
    with st.container(border=True):
        user_input = st.chat_input("Could you tell me about your interests for today?")
    if prompt := user_input:
        context  = get_context(prompt)
        messages = [{
            "role": "system",
            "content": "You are a conference sessions recommender. Recomend user 2 sessions most close to his area of interests. Use only real information. List of sessions is comming from the database and not from the user. Include in your answer session name, type, time, place and reason to attend."
        }, {
            "role": "user",
            "content": prompt
        }]

        messages[-1]['content'] = messages[-1]['content'] + ' List of sessions from database' + context
        messages_cont.chat_message("user").write(prompt)

        with messages_cont.chat_message("assistant"):
            # Initiate streaming session
            try:
                stream = chat_client.chat.completions.create(
                    model=model_chat,
                    messages=messages,
                    stream=True
                )

                # Write the response to the stream
                response = st.write_stream(stream)

                # Append the response to the messages
                messages.append({"role": "assistant", "content": response})
                print(response)

            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()