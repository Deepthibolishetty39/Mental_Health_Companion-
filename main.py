import streamlit as st
from transformers import pipeline
import pandas as pd
from datetime import datetime
import time
import random
import requests

# gemini api key
import os
from google import genai

client =genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# function for getting response and giving it to chatbot
from google.genai.errors import ServerError
import time

def get_chatbot_response(user_input, mood):
    prompt = f"""
    You are a friendly, empathetic mental health support assistant.
    The user is feeling {mood}.
    Respond to their message with kindness, understanding,
    and practical, supportive suggestions.
    Keep your response concise, around 2-4 sentences.

    User says: "{user_input}"
    """

    # Try the primary model, then use a fallback if unavailable.
    models = [
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite"
    ]

    for model in models:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt
                )

                if response.text:
                    return response.text

                return "I'm here to listen. Could you tell me a little more?"

            except ServerError:
                if attempt == 0:
                    time.sleep(3)
                else:
                    break

    return (
        "I'm sorry, the AI service is temporarily busy. "
        "Please wait a minute and try sending your message again."
    )


#  HuggingFace emotion detection model
@st.cache_resource
def load_emotion_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

emotion_model = load_emotion_model()

mood_tips = {
    "sadness": "Try journaling or calling a friend. Listening to your favorite music might help lift your mood.",
    "joy": "Enjoy this moment! Maybe do something fun or share your happiness with someone.",
    "anger": "Take a short walk or some deep breaths. Listening to calming music can also help.",
    "fear": "Try grounding exercises or slow breathing to calm yourself.",
    "neutral": "Keep going! Maybe take a short break or stretch.",
    "surprise": "Take a moment to process your feelings and enjoy the experience.",
    "love": "Spread positivity or do something kind for yourself or others."
}


tab1, tab2,tab3 = st.tabs(["Chat with Buddy", "Journal","Relax & Laugh"])


with tab1:
    # Main title
    st.title("Mental Health Companion Chatbot ")
    st.write("Your buddy in stressful situations")

    #greetings

    hour = datetime.now().hour

    if hour < 12:
        greeting = "Good morning! How are you feeling today?"
    elif hour < 18:
        greeting = "Good afternoon! How’s your day going?"
    else:
        greeting = "Good evening! How are you feeling tonight?"

    st.write(greeting)

    # made session state for our chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    #  made session state for our chat history
    if "mood" not in st.session_state:
        st.session_state.mood = []
     

    chat_container = st.container()
    # user input
    user_input = st.chat_input("how are you feeling buddy")

    #displaying user input
    if user_input:
        # Detect mood
        result = emotion_model(user_input)[0]
        emotion = result["label"]
        #chatbot response calling
        chatbot_response = get_chatbot_response(user_input, emotion)

        # get tip for detected mood
        tip = mood_tips.get(emotion.lower(), "")
        # append tip to chatbot response
        chatbot_response += f"\n\n💡 Tip: {tip}"


        # Save detected mood
        st.session_state.mood.append(emotion)
        
        # saving user input
        st.session_state.messages.append({"role":"user", "content": user_input})
        
        # saving chatbot input
        st.session_state.messages.append({"role":"assistant", "content": chatbot_response})

        with chat_container:
            for msg in st.session_state.messages[-10:]:
                with st.chat_message(msg["role"], avatar="🧑" if msg["role"]=="user" else "🤖"):
                    st.write(msg["content"])
       
                

    # reset button
    if st.sidebar.button("Reset Chat"):
        st.session_state.messages = []
        st.session_state.mood = []    


    # Mapping HuggingFace emotion labels to emojis
    mood_emoji = {
        "joy": "😄",
        "sadness": "😢",
        "anger": "😡",
        "fear": "😰",
        "neutral": "😐",
        "surprise": "😲",
        "love": "❤️"
    }

    #  other names for mood error was occuring cause hugging face moods were not mappimg with our dictionary one
    label_friendly = {
        "joy": "Happy",
        "sadness": "Sad",
        "anger": "Angry",
        "fear": "Anxious",
        "neutral": "Neutral",
        "surprise": "Surprised",
        "love": "Love"
    }
  #sidebar code
  
    st.sidebar.title("Mood History")

    # recent moods
    st.sidebar.subheader("Recent Moods 💖")

    if st.session_state.mood:
        recent_moods = st.session_state.mood[-5:]  # last 5 moods
        for idx, m in enumerate(recent_moods[::-1], 1):  # reverse to show latest first
            emoji = mood_emoji.get(m.lower(), "❓")
            st.sidebar.markdown(f"**{emoji} {m.capitalize()}**")
    else:
        st.sidebar.write("No moods detected yet.")

    st.sidebar.markdown("---")

    # overall mood
    st.sidebar.subheader("Mood Distribution 🌈")
    total = len(st.session_state.mood)
    for mood, emoji in mood_emoji.items():
        count = st.session_state.mood.count(mood)
        if count > 0:
            st.sidebar.write(f"{emoji} {mood.capitalize()} ({count})")
            st.sidebar.progress(count / total)


with tab2:
    st.title("Your Personal Journal 📝")
    
    #  session state for journal entries
    if "journal_entries" not in st.session_state:
        st.session_state.journal_entries = []

    # Text area for writing new entry
    new_entry = st.text_area("Write about your day, feelings, or anything you want:")

    # Save entry button
    if st.button("Save Entry"):
        if new_entry.strip() != "":
            entry_data = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "content": new_entry.strip()
            }
            st.session_state.journal_entries.append(entry_data)
            st.success("Journal entry saved! ✨")
        else:
            st.warning("Write something before saving.")

    st.markdown("---")

    # Show previous entries
    if st.session_state.journal_entries:
        st.subheader("Previous Entries")
        for entry in reversed(st.session_state.journal_entries):
            st.markdown(f"**{entry['time']}**")
            st.markdown(f"{entry['content']}")
            st.markdown("---")
    else:
        st.write("No journal entries yet.")





with tab3:
    st.title("Relax & Laugh 🌈")

    # Breathing Exercise 
    st.subheader("💨 Breathing Exercise")
    pattern = st.selectbox("Choose your breathing pattern:", ["Box Breathing (4-4-4-4)", "4-7-8 Breathing"])
    rounds = st.slider("How many cycles?", 1, 5, 3)

    if st.button("Start Breathing"):
        placeholder = st.empty()
        for i in range(rounds):
            if pattern == "Box Breathing (4-4-4-4)":
                steps = [("Inhale", 4), ("Hold", 4), ("Exhale", 4), ("Hold", 4)]
            else:  # 4-7-8
                steps = [("Inhale", 4), ("Hold", 7), ("Exhale", 8)]

            for step, seconds in steps:
                placeholder.markdown(f"**{step}** for {seconds} seconds")
                time.sleep(seconds)
        placeholder.markdown("✅ Breathing exercise completed!")

    st.markdown("---")

    #  Jokes / Motivational Tips 
    st.subheader("😄 Mood Boosters")
    option = st.selectbox("What would you like?", ["Joke", "Motivational Tip"])
    
    if st.button("Get it!"):
        if option == "Joke":
            try:
                response = requests.get("https://official-joke-api.appspot.com/jokes/random")
                joke = response.json()
                st.success(f"{joke['setup']}\n\n{joke['punchline']}")
            except:
                st.error("Couldn't fetch a joke. Try again later!")
        else:  # Motivational Tip
            try:
                response = requests.get("https://zenquotes.io/api/random")
                tip = response.json()[0]['q'] + " —" + response.json()[0]['a']
                st.success(tip)
            except:
                st.error("Couldn't fetch a tip. Try again later!")