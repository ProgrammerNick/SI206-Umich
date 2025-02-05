# Your name: Nicolas Newberry
# Your student id: 41052667
# Your email: nnicolas@umich.edu
# List of who or what you worked with on this homework: ChatGPT
# If you used Generative AI, say that you used it and also how you used it.
# I used generative AI to help me with the randomization portion of the project. I wasn't exactly sure how to randomize the answer.
# I also used it to help me with understanding how to fix the for loop when it came to printing answer frequencies, by transforming it into a set, it only looked for unique answers, which got me my desired output.


import random

class MagicEightBall:
    def __init__(self, answers):
        self.answers_list = answers
        self.previous_questions = []
        self.previous_answers = []

    def __str__(self):
        if not self.previous_questions:
            return "Questions Asked:  Answers Given:"
        questions = ", ".join(self.previous_questions)
        answers = ", ".join(self.answers_list[i] for i in self.previous_answers)
        return f"""Questions Asked: {questions}\nAnswers Given: {answers}"""

    def get_fortune(self, question):
        if question in self.previous_questions:
            return "I've already answered this question"
        
        answer_index = random.randint(0, len(self.answers_list) - 1)
        self.previous_questions.append(question)
        self.previous_answers.append(answer_index)
        return self.answers_list[answer_index]

    def play_game(self):
        print("Welcome to the Magic Eight Ball game!")
        while True:
            question = input("Please enter a question: ")
            if question.lower() == "done":
                print("Goodbye")
                break
            fortune = self.get_fortune(question)
            print(f"Magic Eight Ball says: {fortune}")

    def print_answer_frequencies(self):
        if not self.previous_answers:
            print("I have not told your fortune yet")
            return {}

        frequencies = {}
        for i in set(self.previous_answers): #by transforming this into a set, it eliminates repeated answers which impacts the amount of times the output was given
            answer = self.answers_list[i]
            count = self.previous_answers.count(i)
            frequencies[answer] = count
            print(f"The answer '{answer}' has been given {count} times.")
        return frequencies

def main():
    possible_answers = [
        "Definitely", "Most Likely", "It is certain", "Maybe", 
        "Ask again later", "Very doubtful", "Don’t count on it", "Absolutely not"
    ]
    magic_eight_ball = MagicEightBall(possible_answers)

    print(magic_eight_ball)
    magic_eight_ball.play_game()
    magic_eight_ball.print_answer_frequencies()
    print(magic_eight_ball)

if __name__ == "__main__":
    main()
