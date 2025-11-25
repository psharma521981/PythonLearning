#data structure
#List
#Tupels
#Set
#Directory


aList = [2,1,3,4]
aTuple = (4,5,6,"g")
aSet = {3,2,1,"bye"}
aDict = {1:"a",2:"b",3:"c"}

for i in aList:
    print(i, end=" ")
print()
for i in aTuple:
    print(i, end=" ")
print()
for i in aSet:
    print(i, end=" ")
print()
for i in aDict:
    print(i, end=" ")
print()
for i in aDict:
    print(f"{i} is {aDict[i]}")