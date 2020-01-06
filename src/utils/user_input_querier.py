import time


class UserInputQuerier:
    def __init__(self):
        pass

    @staticmethod
    def ask(question):
        user_input = ""
        while user_input != "y" and user_input != "n":
            time.sleep(0.01)  # Prevents problems of race condition with the logger
            user_input = input("\n" + question)

        return user_input == "y"
