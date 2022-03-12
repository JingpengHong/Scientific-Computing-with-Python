"""
fizzbuzz

Write a python script which prints the numbers from 1 to 100,
but for multiples of 3 print "fizz" instead of the number,
for multiples of 5 print "buzz" instead of the number,
and for multiples of both 3 and 5 print "fizzbuzz" instead of the number.
"""

for i in range(1, 101):
    '''
    Print "fizz", "bizz" or "fizzbuzz" for the multiples of 3, 5 or both.
    '''
    # check if i is the multiple of three
    if i % 3 == 0:
        fizz = "fizz" 
    else:
        fizz = ""
        
    # check if i is the multiple of five
    if i % 5 == 0:
        bizz = "bizz"
    else:
        bizz = "" 
        
    print(f"{fizz}{bizz}" or i) # use f-strings to combine fizz and bizz
        