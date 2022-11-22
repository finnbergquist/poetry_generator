'''
Author: Finn Bergquist
This module runs the REST API which serves as the backend for the poetry 
generation UI. The app implements an instance of poetry_agent, and uses that
to create and display poetry in the front end. The frontend is rendered 
with html/css, using files in the templates directory.
'''
from flask import Flask, request, render_template
from poetry_agent import PoetryAgent
import requests
import os

#Flask recognizes this variable as the app
app = Flask(__name__)

#Global Variables
global poetry_agent 
poetry_agent = PoetryAgent()
poetry_agent.load('models')
global current_poem
current_poem = []
global saved_poems
saved_poems = []
global current_poem_errors
current_poem_errors = []
global saved_poem_errors
saved_poem_errors = []

@app.route("/")
def home():
    '''Renders the home page'''
    return render_template('/home.html', current_poem=current_poem)

@app.route("/poems")
def poems():
    '''Renders the poem list page'''
    return render_template('/poems.html',saved_poems=saved_poems, 
    saved_poem_errors=saved_poem_errors)

@app.route("/add_word", methods=["POST"])
def add_word():
    '''This  API call asks the poetry ML agent for a next word. The method then
    looks for the last word that is not a new line symbol and passes that as
    the seed_text two the poetry_agent, which will return the recommended word.
    If there are no words yet, then the method requests a word from the poetry
    agent with a seed text of a blank string. This renders an approximately even
    probability distribution, selecting words proportionally to how often they
    occur, regardless of the context. Finally, it adds to new word to the global
    current_poemn list.'''
    global current_poem
    global current_poem_errors

    if current_poem:
        #if the last word is a new line symbol, then take the word before it
        last_word = current_poem[-1]
        if last_word == "\n":
            last_word = current_poem[-2]        
        next_word,error = poetry_agent.next_word(last_word)
    else:
        next_word,error = poetry_agent.next_word("")
    current_poem.append(next_word)
    current_poem_errors.append(error)
    return home()

@app.route("/add_word_manually", methods=["POST"])
def add_word_manually():
    '''This call allows the user to specify their own addition to the poem. It
    asks the user for a word that they want to add in a text box, and then 
    appends that word to the current_poem global list.'''
    global current_poem
    word_to_add = request.form.get("manual_word")
    try:
        current_poem.append(str(word_to_add))
    except:
        print("Error while adding word")
    return home()

def rmse(errors):
    '''Calculates the root mean squared error from a list of the error
    calculations. Each element in the errors list represents (y'-y) in the 
    formula rmse = sqrt(sum((y'-y)^2)/n)'''
    error_sum = 0
    for error in errors:
        error_sum += error**2
    n = len(errors)+1
    return (error_sum/n)**(1/2)

@app.route("/save_poem", methods=["GET"])
def save_poem():
    ''' This is a button that can be selected from the header bar, and it saves
    the poem to be displayed on the pther page, Recorded Poems.'''
    global saved_poems
    global current_poem
    global current_poem_errors
    global saved_poem_errors
    global poetry_agent

    #save rmse evaluative metric for poem
    saved_poem_errors.append(round(rmse(current_poem_errors), 3))
    current_poem_errors = []

    #save the poem text
    saved_poems.append(current_poem)
    current_poem = []
    return home()

def list_to_str(list):
    """Converts a poem in the form of a list to a string. This helper function
    serves the purpose of making the poems audibly readable by the operating 
    system. There is some extra logic for correct apostrophe pronunciation"""
    poem = ""
    for word in list:
        if word == '\n':#for pronunciation pause after each line
            word = '. '
        poem = poem + ' ' + word
    return poem.replace("'","\''")

@app.route("/speak_poem", methods=["POST"])
def speak_poem():
    '''Recieves the number corresponding to a poem and reads that poem using
    os system call in the Alex voice'''
    poem_number = int(request.form.get('poem_number'))
    poem_as_list = saved_poems[poem_number-1]
    poem_as_str = list_to_str(poem_as_list)
    terminal_command = "say -v Alex -r 140 '" + poem_as_str + "'"
    os.system(terminal_command)
    return poems() 

@app.route("/new_line", methods=["POST"])
def new_line():
    '''Generates a newline symbol as the next word in the poem, and it appends 
    a corresponding error of zero'''
    global current_poem
    current_poem.append('\n')
    current_poem_errors.append(0)
    return home()

