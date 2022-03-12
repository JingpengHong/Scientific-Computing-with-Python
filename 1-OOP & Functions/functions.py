"""
A library of functions
"""
import numpy as np
import matplotlib.pyplot as plt
import numbers

class AbstractFunction:
    """
    An abstract function class
    """

    def derivative(self):
        """
        returns another function f' which is the derivative of x
        """
        raise NotImplementedError("derivative")


    def __str__(self):
        return "AbstractFunction"


    def __repr__(self):
        return "AbstractFunction"


    def evaluate(self, x):
        """
        evaluate at x

        assumes x is a numeric value, or numpy array of values
        """
        raise NotImplementedError("evaluate")


    def __call__(self, x):
        """
        if x is another AbstractFunction, return the composition of functions

        if x is a string return a string that uses x as the indeterminate

        otherwise, evaluate function at a point x using evaluate
        """
        if isinstance(x, AbstractFunction):
            return Compose(self, x)
        elif isinstance(x, str):
            return self.__str__().format(x)
        else:
            return self.evaluate(x)


    # the rest of these methods will be implemented when we write the appropriate functions
    def __add__(self, other):
        """
        returns a new function expressing the sum of two functions
        """
        return Sum(self, other)


    def __mul__(self, other):
        """
        returns a new function expressing the product of two functions
        """
        return Product(self, other)


    def __neg__(self):
        return Scale(-1)(self)


    def __truediv__(self, other):
        return self * other**-1


    def __pow__(self, n):
        return Power(n)(self)
    


    def plot(self, vals=np.linspace(-1,1,100), **kwargs):
        """
        plots function on values
        
        pass kwargs to plotting function
        """
        p = plt.plot(vals, self.evaluate(vals), **kwargs)
        return p
    
    def taylor_series(self, x0, deg=5):
       """
       Returns the Taylor series of f centered at x0 truncated to degree k.
       """
       T  = Constant(self(x0))
       d  = Affine(1, -x0) # x - x0
       f_dk = self.derivative()
       
       factorial = 1 # calculates factorial(k)
       for k in range(1, deg+1):
           
           factorial = factorial * k
           T  = T + Constant(f_dk(x0) / factorial) * d**k
           f_dk = f_dk.derivative() # derivative in higher degree

       return T
    



class Polynomial(AbstractFunction):
    """
    polynomial c_n x^n + ... + c_1 x + c_0
    """

    def __init__(self, *args):
        """
        Polynomial(c_n ... c_0)

        Creates a polynomial
        c_n x^n + c_{n-1} x^{n-1} + ... + c_0
        """
        self.coeff = np.array(list(args))


    def __repr__(self):
        return "Polynomial{}".format(tuple(self.coeff))


    def __str__(self):
        """
        We'll create a string starting with leading term first

        there are a lot of branch conditions to make everything look pretty
        """
        s = ""
        deg = self.degree()
        for i, c in enumerate(self.coeff):
            if i < deg-1:
                if c == 0:
                    # don't print term at all
                    continue
                elif c == 1:
                    # supress coefficient
                    s = s + "({{0}})^{} + ".format(deg - i)
                else:
                    # print coefficient
                    s = s + "{}({{0}})^{} + ".format(c, deg - i)
            elif i == deg-1:
                # linear term
                if c == 0:
                    continue
                elif c == 1:
                    # suppress coefficient
                    s = s + "{0} + "
                else:
                    s = s + "{}({{0}}) + ".format(c)
            else:
                if c == 0 and len(s) > 0:
                    continue
                else:
                    # constant term
                    s = s + "{}".format(c)

        # handle possible trailing +
        if s[-3:] == " + ":
            s = s[:-3]

        return s


    def evaluate(self, x):
        """
        evaluate polynomial at x
        """
        if isinstance(x, numbers.Number):
            ret = 0
            for k, c in enumerate(reversed(self.coeff)):
                ret = ret + c * x**k
            return ret
        elif isinstance(x, np.ndarray):
            x = np.array(x)
            # use vandermonde matrix
            return np.vander(x, len(self.coeff)).dot(self.coeff)


    def derivative(self):
        if len(self.coeff) == 1:
            return Polynomial(0)
        return Polynomial(*(self.coeff[:-1] * np.array([n+1 for n in reversed(range(self.degree()))])))


    def degree(self):
        return len(self.coeff) - 1


    def __add__(self, other):
        """
        Polynomials are closed under addition - implement special rule
        """
        if isinstance(other, Polynomial):
            # add
            if self.degree() > other.degree():
                coeff = self.coeff
                coeff[-(other.degree() + 1):] += other.coeff
                return Polynomial(*coeff)
            else:
                coeff = other.coeff
                coeff[-(self.degree() + 1):] += self.coeff
                return Polynomial(*coeff)

        else:
            # do default add
            return super().__add__(other)


    def __mul__(self, other):
        """
        Polynomials are closed under multiplication - implement special rule
        """
        if isinstance(other, Polynomial):
            return Polynomial(*np.polymul(self.coeff, other.coeff))
        else:
            return super().__mul__(other)


class Affine(Polynomial):
    """
    affine function a * x + b
    """
    def __init__(self, a, b):
        super().__init__(a, b)



class Scale(Polynomial):
    """
    Scale a * x + 0
    
    Child class of Polynomial
    """
    def __init__(self, a):
        """
        Scale a * x + 0

        Creates a scale a * x
        """
        super().__init__(a, 0)
        
        
class Constant(Polynomial):
    """
    Constant c
    
    Child class of Polynomial
    """
    def __init__(self, c):
        """
        Constant c
        
        Creates a constant c
        """
        super().__init__(c)
        



class Compose(AbstractFunction):
    """
    Compose class, Compose(f, g)(x) acts as f(g(x))
    
    Child class of AbstractFunction
    """
    
    def __init__(self, f, g):
        """
        Compose(f,g)
        
        Creates a Compose operation f(g(x))
        """
        self.f = f
        self.g = g
        
    def __str__(self):
        """
        Creates a string to show the compose operation
        
        Indeterminants is expressed as {0}
        """
        return self.f(self.g.__str__())
    
    def __repr__(self):
        """
        Creates a representation of Compose(f,g)
        
        Compose f of g
        """
        return "Compose {} of {}".format(self.f, self.g)
    
    def derivative(self):
        """
        Calculates the derivative of the Compose using chain rule
        
        returns an AbstractFunction object
        """
        return self.f.derivative()(self.g) * self.g.derivative()
    
    def evaluate(self, x):
        """
        evaluates the Compose at x
        """
        return self.f.evaluate(self.g.evaluate(x))
        
        
    

class Product(AbstractFunction):
    """
    Product class, Product(f, g)(x) act as f(x) * g(x)
    
    Child class of AbstractFunction
    """
    def __init__(self, f, g):
        """
        Product(f,g)
        
        Creates a Product operation f(x) * g(x)
        """
        self.f = f
        self.g = g
        
    
    def __str__(self):
        """
        Creates a string to show the product operation
        
        Indeterminants is expressed as {0}
        """
        return "({}) * ({})".format(self.f.__str__(), self.g.__str__())
    
    def __repr__(self):
        """
        Creates a representation of Product(f,g)
        
        Product of f and g
        """
        return "Product of {} and {}".format(self.f, self.g)

    
    def derivative(self):
        """
        Calculates the derivative of the Product using product rule
        
        returns an AbstractFunction object
        """
        return self.f.derivative() * self.g + self.f * self.g.derivative()
    
    def evaluate(self, x):
        """
        evaluates the Product at x
        """
        return self.f.evaluate(x) * self.g.evaluate(x)

    


class Sum(AbstractFunction):
    """
    Sum Class, Sum(f, g)(x) act as f(x) + g(x)
    
    Child class of AbstractFunction
    """
    def __init__(self, f, g):
        """
        Sum(f,g)
        
        Creates a Product operation f(x) + g(x)
        """
        self.f = f
        self.g = g
        
    
    def __str__(self):
        """
        Creates a string to show the sum operation
        
        Indeterminants is expressed as {0}
        """
        return "({}) + ({})".format(self.f.__str__(), self.g.__str__())
    
    def __repr__(self):
        """
        Creates a representation of Sum(f,g)
        
        Sum of f and g
        """
        return "Sum of {} and {}".format(self.f, self.g)

    
    def derivative(self):
        """
        Calculates the derivative of the Sum
        
        returns an AbstractFunction object
        """
        return self.f.derivative() + self.g.derivative()
    
    def evaluate(self, x):
        """
        evaluates the Sum at x
        """
        return self.f.evaluate(x) + self.g.evaluate(x)
        
    

    
class Power(AbstractFunction):
    """
    Power class, Power(n)(x) act as x**n
    
    Child class of AbstractFunction
    """
    def __init__(self, n):
        """
        Power(n)
        
        Creates a Power x^n
        """
        self.n = n
        
    def __str__(self):
        """
        Creates a string to show the Power
        
        Indeterminants is expressed as {0}
        """
        return f"({{0}})^{self.n}"
    
    def __repr__(self):
        """
        Creates a representation of Power(n)
        
        returns Power(n)
        """
        return f"Power({self.n})"
    
    def derivative(self):     
        """
        Calculates the derivative of the Power
        
        returns a Constant multiplied by a Power, an AbstractFunction object
        """
        return Constant(self.n) * Power(self.n - 1)
    
    def evaluate(self, x):
        """
        evaluates the Power at x
        """
        return (x ** self.n)
    

class Log(AbstractFunction):
    """
    Log class, Log()(x) act as np.log(x)
    
    Child class of AbstractFunction
    """
    def __init__(self):
        """
        Log()
        
        Creates log()
        """
        pass
        
    def __str__(self):
        """
        Creates a string to show the log
        
        Indeterminants is expressed as {0}
        """
        return "log({0})"
    
    def __repr__(self):
        """
        Creates a representation of Log()
        
        returns Log()
        """
        return "Log()"
    
    def derivative(self):
        """
        Calculates the derivative of the Log
        
        returns Power(-1), an AbstractFunction object
        """
        return Power(-1)
    
    def evaluate(self, x):
        """
        evaluates the Log at x
        """
        return np.log(x)


class Exponential(AbstractFunction):
    """
    Exponential class, Exponential()(x) act as np.exp(x)
    
    Child class of AbstractFunction
    """   
    def __init__(self):
        """
        Exponential()
        
        Creates exp()
        """
        pass
        
    def __str__(self):
        """
        Creates a string to show the exponential
        
        Indeterminants is expressed as {0}
        """
        return "e^({0})"
    
    def __repr__(self):
        """
        Creates a representation of Exponential()
        
        returns Exponential()
        """
        return "Exponential()"
    
    def derivative(self):
        """
        Calculates the derivative of the Log
        
        returns itself Exponential(), an AbstractFunction object
        """
        return Exponential()
    
    def evaluate(self, x):
        """
        evaluates the Exponential at x
        """
        return np.exp(x)


class Sin(AbstractFunction):
    """
    Sin class, Sin()(x) act as np.sin(x)
    
    Child class of AbstractFunction
    """   
    def __init__(self):
        """
        Sin()
        
        Creates sin()
        """
        pass
        
    def __str__(self):
        """
        Creates a string to show the sin
        
        Indeterminants is expressed as {0}
        """
        return "sin({0})"
    
    def __repr__(self):
        """
        Creates a representation of Sin()
        
        returns Sin()
        """
        return "Sin()"
    
    def derivative(self):
        """
        Calculates the derivative of the Log
        
        returns Cos(), an AbstractFunction object
        """
        return Cos()
    
    def evaluate(self, x):
        """
        evaluates the Sin at x
        """
        return np.sin(x)


class Cos(AbstractFunction):
    """
    Cos class, Cos()(x) act as np.cos(x)
    
    Child class of AbstractFunction
    """   
    def __init__(self):
        """
        Cos()
        
        Creates cos()
        """
        pass
        
    def __str__(self):
        """
        Creates a string to show the cos
        
        Indeterminants is expressed as {0}
        """
        return "cos({0})"
    
    def __repr__(self):
        """
        Creates a representation of Cos()
        
        returns Cos()
        """
        return "Cos()"
    
    def derivative(self):
        """
        Calculates the derivative of the Log
        
        returns Constant(-1) times Sin(), an AbstractFunction object
        """
        return Constant(-1) * Sin()
        
    
    def evaluate(self, x):
        """
        evaluates the Cos at x
        """
        return np.cos(x)


class Symbolic(AbstractFunction):
    """
    Symbolic function class, creates a symbol for a function using the function name string
    
    Child class of AbstractFunction
    """
    def __init__(self, s):
        """
        Creates a function symbol using the function name string
        """
        self.s = s
        
    def __str__(self):
        """
        Creates a string for printing out the Symbolic
        
        Indeterminants are expressed as {0}
        """
        return f"{self.s}({{0}})"
    
    def __call__(self, x):
        """
        Print a string with whatever the input is inside the function symbol
        """
        return (f"{self.s}({x})")
    
    def derivative(self): 
        """
        Add an apostrophe to the end of the string name
        
        returns a Symbolic, an AbstractFunction object
        """
        a = self.s + "'"
        return Symbolic(a)
        
        

        
