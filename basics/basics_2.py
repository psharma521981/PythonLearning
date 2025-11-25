#conditional statements
#loops
#relational operator

a = 16

if (a % 2) == 0:
    print("even")
else:
    print("odd")

age = [4,5,6,7,9,11,15,30]

for i in age:
    if i % 3 == 0 or i % 5 == 0:
        print(f"{i} divisible by either 3 & 5 ");
    else:
        print(f"{i} not divisible by 3 & 5");
