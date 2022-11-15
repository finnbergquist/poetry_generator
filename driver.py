from poetry_agent import PoetryAgent

def main():
    poetry_agent = PoetryAgent()
    #poetry_agent.train()
    #poetry_agent.save('models')
    #print(len(poetry_agent.X))
    #print(len(poetry_agent.y))

    poetry_agent.load('models')

    seed_text =  'I'
    total_text = seed_text
    words = []

    for i in range(50):
        word = poetry_agent.next_word(seed_text)
        words.append(word)
        seed_text = words[-1]
        total_text = total_text + ' ' + word

   
    print(total_text)

if __name__=="__main__":
    main()