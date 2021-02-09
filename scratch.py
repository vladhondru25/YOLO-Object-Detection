def f(a, b):
    print(a)
    print(b)
    
def funt(a, b, c, d):
    print(a)
    print(b)
    f(*c)
    print(d)


x = 1
y = 2
funt(0, 0, (1,2), 0)