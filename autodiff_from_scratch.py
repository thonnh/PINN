from abc import ABC, abstractmethod

class DifferentiableSymbolicOperation(ABC):
    @abstractmethod
    def backward(self, var):
        pass

    @abstractmethod
    def compute(self):
        pass
    
class Const(DifferentiableSymbolicOperation):
    def __init__(self, value):
        self.value = value

    def backward(self, var):
        return Const(0)

    def compute(self):
        return self.value

    def __repr__(self):
        return str(self.value)
    
class Var(DifferentiableSymbolicOperation):
    def __init__(self, name, value=None):
        self.name, self.value = name, value

    def backward(self, var):
        return Const(1) if self == var else Const(0)

    def compute(self):
        if self.value is None:
            raise ValueError('unassigned variable')
        return self.value

    def __repr__(self):
        return f'{self.name}'

class Sum(DifferentiableSymbolicOperation):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def backward(self, var):
        return Sum(self.x.backward(var), self.y.backward(var))

    def compute(self):
        return self.x.compute() + self.y.compute()

    def __repr__(self):
        return f'({self.x} + {self.y})'
    
class Mul(DifferentiableSymbolicOperation):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def backward(self, var):
        return Sum(
            Mul(self.x.backward(var), self.y),
            Mul(self.x, self.y.backward(var))
        )

    def compute(self):
        return self.x.compute() * self.y.compute()

    def __repr__(self):
        return f'({self.x} * {self.y})'

x = Var('x', 3)
y = Var('y', 2)

z = Sum(
    Sum(
        Mul(x, x),
        Mul(Const(3), Mul(x, y))
    ),
    Const(1)
)

print(z)  # Outputs: (((x * x) + (3 * (x * y))) + 1)
print(z.compute())  # Computes the value of the expression

print(z.backward(x))  # Computes the derivative of z with respect to x
print('eiei')  # Computes the derivative of z with respect to x
print(z.backward(x).backward(x))  # Computes the derivative of z with respect to x

print(z.backward(x).backward(x).compute())  # Computes the derivative of z with respect to x