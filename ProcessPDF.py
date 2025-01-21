from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import tkinter as tk
import PyPDF2
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from gtts import gTTS
import os
from PIL import Image, ImageTk  # Import Image and ImageTk from Pillow
from tkinter import Scrollbar, Text, Entry, Button, END, PhotoImage, Toplevel
import threading
import pygame
from pygame import mixer  # Import mixer from pygame

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):   
    load_dotenv()
    # # extract the text
    text = ""
    with open(pdf_path, "rb") as pdf_file:
     pdf_reader = PyPDF2.PdfReader(pdf_file)
     for page in pdf_reader.pages:
        text += page.extract_text()
    # split into chunks
    text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
      )
    chunks = text_splitter.split_text(text)
      
      # create embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)   
    return knowledge_base

  # Function to interact with LLM and get response from the database
def get_llmResponse(user_question):     
     # show user input
      #user_question = user_entry.get()
      if user_question.lower() == "exit":
        root.destroy()
      if user_question:
        docs = knowledge_base.similarity_search(user_question)
        
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
          response = chain.run(input_documents=docs, question=user_question)
          print(cb)
          return response

# Function to handle user input and display chat responses
def write_tochat():
    user_question = user_entry.get()
    user_entry.delete(0, END)

    chat_response = get_llmResponse(user_question)

    chat_display.config(state=tk.NORMAL)
    chat_display.insert(tk.END, f"User: {user_question}\n")
    chat_display.insert(tk.END, f"ChatGPT: {chat_response}\n\n")
    chat_display.config(state=tk.DISABLED)
    chat_display.see(tk.END)

    if speech_enabled:
        # Convert the chat response to speech and play it using pygame
        speech = gTTS(text=chat_response, lang='en')
        speech.save("chat_response.mp3")

        # Initialize pygame mixer
        pygame.mixer.init()

        # Load and play the audio file
        pygame.mixer.music.load("chat_response.mp3")
        pygame.mixer.music.play()

# Function to toggle speech on/off
def toggle_speech():
    global speech_enabled

    speech_enabled = not speech_enabled

    if speech_enabled:
        speech_button.config(text="Speech Enabled")
    else:
        speech_button.config(text="Speech Disabled", image=speech_disabled_icon)

# Initialize the knowledge_base variable (Replace with your PDF extraction logic)
knowledge_base = None
pdf_path = "Novus-Op.pdf"  # Replace with your PDF file path
knowledge_base = extract_text_from_pdf(pdf_path)

# Initialize speech status and the main window
speech_enabled = False
root = tk.Tk()
root.title("Nova Chat")
root.geometry("400x500")

# Create a Text widget for displaying the chat
chat_display = Text(root, wrap=tk.WORD)
chat_display.pack(fill=tk.BOTH, expand=True)
chat_display.config(state=tk.DISABLED)

# Create a Scrollbar for the chat display
scrollbar = Scrollbar(chat_display)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
scrollbar.config(command=chat_display.yview)
chat_display.config(yscrollcommand=scrollbar.set)

# Create an Entry widget for user input
user_entry = Entry(root)
user_entry.pack(fill=tk.X)

# Create a Send button to send user input
send_button = Button(root, text="Send", command=write_tochat)
send_button.pack()

# Load speech icons
speech_enabled_icon = PhotoImage(file="speech_enabled.png")
speech_disabled_icon = PhotoImage(file="speech_disabled.png")

# Initialize the main speech button
speech_button = Button(root, text="Speech Disabled", command=toggle_speech, compound="left", image=speech_disabled_icon)
speech_button.pack()

# Start the GUI main loop
root.mainloop()

