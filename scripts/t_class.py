def geta():
    class A():
        a1=100
        a2=200
    a=A()
    return a


a=geta()
print(a.a1, a.a2)