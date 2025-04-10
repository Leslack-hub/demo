def hello2(num):
    print(num)

    def w2rapper(func):
        def wrapper():
            print("Hello, World!", 3243232)
            func()

        return wrapper

    return w2rapper


@hello2(23)
@hello2(23)
@hello2(23)
@hello2(23)
@hello2(23)
@hello2(23)
@hello2(23)
def heelo():
    print("Hello, World!")


heelo()
