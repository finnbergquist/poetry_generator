'''
Author: Finn Bergquist
This module trains the recurrent neural network. The purpose for seperating
this into its own file is so that the model can just be loaded rather than
re-trained each time the program is run.
'''
from poetry_agent import PoetryAgent

def main():
    poetry_agent = PoetryAgent()
    poetry_agent.train()
    poetry_agent.save('models')

if __name__ == "__main__":
    main()