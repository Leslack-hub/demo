from typing import Self


class A:

    def __init__(self, a):
        self.a = a

    def __add__(self, a: Self) -> Self:
        self.a += a.a
        return A(self.a)


a = A(1)
b = A(3)
c = a + b
print(c.a)
