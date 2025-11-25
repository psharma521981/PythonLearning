
#class
#function
def myFunc(name):
    print(f"Myname is {name}")

def add(a,b):
    return a+b

class Test:
    def __init__(self,a,b):
        self.a=a
        self.b=b
    def myFunc(self,name):
        print(f"Myname is {name} and sum is {self.a+self.b}")

test = Test(10,5)
test.myFunc("Pankaj")

