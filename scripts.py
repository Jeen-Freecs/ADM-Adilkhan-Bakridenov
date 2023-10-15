#Introduction
#Say "Hello, World!" With Python

if __name__ == '__main__':
    print("Hello, World!")

def is_leap(year):
    leap = False
    if year % 4 ==0 and year % 100 != 0:
        leap = True
    elif year % 100 == 0 and year % 400 == 0:
        leap = True
    return leap

#Python If-Else
import math
import os
import random
import re
import sys

if __name__ == '__main__':
    n = int(input().strip())
    if n % 2 == 0 and n<=5 and n>=2:
        print('Not Weird')
    elif n % 2 == 0 and n<=20 and n>=6:
        print('Weird')
    elif n % 2 == 0 and n>20:
        print('Not Weird')
    elif n % 2 != 0:
        print('Weird')

year = int(input())
print(is_leap(year))

#Python: Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a//b)
    print(a/b)
#Loops
if __name__ == '__main__':
    n = int(input())
    for i in range(n):
        print(i**2)

#Write a function
def is_leap(year):
    leap = False
    if year % 4 ==0 and year % 100 != 0:
        leap = True
    elif year % 100 == 0 and year % 400 == 0:
        leap = True
    return leap

#Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a+b)
    print(a-b)
    print(a*b)

#Print Function
if __name__ == '__main__':
    n = int(input())
    values = [i for i in range(1, n+1)]
    print(*values, sep='', end='\n')

#Basic Data Types
#List Comprehensions

if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    
    answer = [[i,j,k] for i in range(x+1) for j in range(y+1) for k in range(z+1) if i+j+k !=n]
    print(answer)

#Find the Runner-Up Score!
if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    
    print(sorted(list(set(list(arr))), reverse = True)[1])

#Nested Lists

if __name__ == '__main__':
    records = []
    for _ in range(int(input())):
        name = input()
        score = float(input())
        temp_record = [name, score]
        records.append(temp_record)
    scores = [record[1] for record in records]
    u_scores = sorted(set(scores))
    answers = sorted([record[0] for record in records if record[1]==u_scores[1]])
    [print(name) for name in answers]
    

#Finding the percentage
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    scores = student_marks[query_name]
    ans = sum(scores)/len(scores)
    print("{:.2f}".format(ans))

#Tuples
if __name__ == '__main__':
    n = int(input())
    integers = map(int, input().split())
    
    print(hash(tuple(integers)))

#sWAP cASE
def swap_case(s):
    ans = []
    for l in s:
        if l.isupper():
            l = l.lower()
        else:
            l = l.upper()
        ans.append(l)
    return ''.join(ans)

if __name__ == '__main__':
    s = input()
    result = swap_case(s)
    print(result)

#String Split and Join
def split_and_join(line):
    ans = line.split(' ')
    ans = '-'.join(ans)
    return ans

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

#What's Your Name?
def print_full_name(first, last):
    print(f'Hello {first} {last}! You just delved into python.')

if __name__ == '__main__':
    first_name = input()
    last_name = input()
    print_full_name(first_name, last_name)

#Mutations
def mutate_string(string, position, character):
    ls = list(string)
    ls[position] = character
    ans = ''.join(ls)
    return ans

if __name__ == '__main__':
    s = input()
    i, c = input().split()
    s_new = mutate_string(s, int(i), c)
    print(s_new)

#Find a string
def count_substring(string, sub_string):
    k = 0
    for i in range(len(string)):
        if string.startswith(sub_string, i):
            k += 1
    return k

if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()
    
    count = count_substring(string, sub_string)
    print(count)

#String Validators
if __name__ == '__main__':
    s = input()  
    print(any(i.isalnum() for i in s))
    print(any(i.isalpha() for i in s))
    print(any(i.isdigit() for i in s))
    print(any(i.islower() for i in s))
    print(any(i.isupper() for i in s))

#Text Wrap
import textwrap

def wrap(string, max_width):
    ans = textwrap.fill(string,max_width)
    return ans

if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)

#Merge the Tools!

import textwrap
def merge_the_tools(string, k):
    parts = textwrap.wrap(string,k) 
    for part in parts:  
        check = set()
        ans = [l for l in part if not (l in check or check.add(l))]
        print(''.join(ans))

if __name__ == '__main__':
    string, k = input(), int(input())
    merge_the_tools(string, k)

#Introduction to Sets
def average(array):
    ans = set(array)
    ans = sum(ans)/len(ans)
    return ans
    # your code goes here

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)

#Symmetric Difference
n = int(input())
a = input()
n_1 = int(input())
b = input()
a_list = set(map(int, a.split()))
b_list = set(map(int, b.split()))
ans = a_list.difference(b_list)
ans.update(b_list.difference(a_list))
tot_ans = sorted(list(ans))
for i in range(len(tot_ans)):
    print(tot_ans[i])

#Set .add()
n= int(input())
countries = set()
for i in range(n):
    country = input()
    countries.add(country)
ans = len(countries)
print(ans)

#Set .discard(), .remove() & .pop()
n = int(input())
s = set(map(int, input().split()))
n_com = int(input())
for i in range(n_com):
    command = input()
    if command == 'pop':
        s.pop()
    elif command.split()[0] == 'remove':
        s.remove(int(command.split()[1]))
    else:
        s.discard(int(command.split()[1]))
print(sum(s))

#No Idea!
n, m = map(int, input().split())
arr = list(map(int, input().split()))
A = set(map(int, input().split()))
B = set(map(int, input().split()))
ans = 0
for i in arr:
    if i in A:
        ans += 1
    elif i in B:
        ans -= 1
print(ans)

#Set .intersection() Operation
n = int(input())
A = set(map(int, input().split()))
n_1 = int(input())
B = set(map(int, input().split()))
ans = A.intersection(B)
print(len(ans))

#Set .difference() Operation
n = int(input())
A = set(map(int, input().split()))
n_1 = int(input())
B = set(map(int, input().split()))
ans = A.difference(B)
print(len(ans))

#Set .symmetric_difference() Operation
n = int(input())
A = set(map(int, input().split()))
n_1 = int(input())
B = set(map(int, input().split()))
ans = A.symmetric_difference(B)
print(len(ans))

#Set Mutations
n = int(input())
A = set(map(int, input().split()))
n_1 = int(input())
for i in range(n_1):
    command = input().split()[0]
    B = set(map(int, input().split()))
    if command == 'intersection_update':
        A.intersection_update(B)
    elif command == 'update':
        A.update(B)
    elif command == 'symmetric_difference_update':
        A.symmetric_difference_update(B)
    else:
        A.difference_update(B)
print(sum(A))

#Set .union() Operation
n = int(input())
A = set(map(int, input().split()))
n_1 = int(input())
B = set(map(int, input().split()))
ans = A.union(B)
print(len(ans))

#Check Subset
n = int(input())
for i in range(n):
    n_1 = int(input())
    A = set(map(int, input().split()))
    n_2 = int(input())
    B = set(map(int, input().split()))
    if A.issubset(B):
        print(True)
    else:
        print(False)

#Check Strict Superset
A = set(map(int, input().split()))
n = int(input())
ans = True
for i in range(n):
    B = set(map(int, input().split()))
    if not A.issuperset(B):
        ans = False
        break
print(ans)

#The Captain's Room
k = int(input())
rooms = list(map(int, input().split()))
s = set()
cancel = []
for i in rooms:
    if i in s:
        s.remove(i)
        cancel.append(i)
    elif i in cancel:
        continue
    else:
        s.add(i)
print(s.pop())

#collections.Counter()
from collections import Counter
n = int(input())
sizes = Counter(map(int, input().split()))
n_1 = int(input())
ans = 0
for i in range(n_1):
    size, price = map(int, input().split())
    if sizes[size] > 0:
        ans += price
        sizes[size] -= 1
print(ans)

#DefaultDict Tutorial
from collections import defaultdict
n, m = map(int, input().split())
A = defaultdict(list)
for i in range(n):
    A[input()].append(i+1)
for i in range(m):
    print(*A[input()] or [-1])

#Collections.namedtuple()
from collections import namedtuple
n = int(input())
columns = input().split()
Student = namedtuple('Student', columns)
ans = 0
for i in range(n):
    s = Student(*input().split())
    ans += int(s.MARKS)
print("{:.2f}".format(ans/n))

#Collections.OrderedDict()
from collections import OrderedDict
n = int(input())
d = OrderedDict()
for i in range(n):
    item, space, price = input().rpartition(' ')
    d[item] = d.get(item, 0) + int(price)
for item, price in d.items():
    print(item, price)

#Word Order
from collections import OrderedDict
n = int(input())
d = OrderedDict()
for i in range(n):
    word = input()
    d[word] = d.get(word, 0) + 1
print(len(d))
print(*d.values())

#Collections.deque()
from collections import deque
d = deque()
n = int(input())
for i in range(n):
    command = input().split()
    if command[0] == 'append':
        d.append(command[1])
    elif command[0] == 'appendleft':
        d.appendleft(command[1])
    elif command[0] == 'pop':
        d.pop()
    else:
        d.popleft()
print(*d)

#Piling Up!
from collections import deque
n = int(input())
for i in range(n):
    m = int(input())
    d = deque(map(int, input().split()))
    ans = 'Yes'
    if d[0] >= d[-1]:
        max = d.popleft()
    else:
        max = d.pop()
    while len(d) > 0:
        if d[0] >= d[-1] and d[0] <= max:
            max = d.popleft()
        elif d[0] < d[-1] and d[-1] <= max:
            max = d.pop()
        else:
            ans = 'No'
            break
    print(ans)

#Company Logo
from collections import Counter
s = input()
letters = Counter(s)
letters = sorted(letters.items(), key = lambda x: (-x[1], x[0]))
for i in range(3):
    print(*letters[i])

#Calendar Module
import calendar
month, day, year = map(int, input().split())
ans = calendar.weekday(year, month, day)
if ans == 0:
    print('MONDAY')
elif ans == 1:
    print('TUESDAY')
elif ans == 2:
    print('WEDNESDAY')
elif ans == 3:
    print('THURSDAY')
elif ans == 4:
    print('FRIDAY')
elif ans == 5:
    print('SATURDAY')
else:
    print('SUNDAY')

#Exceptions
n = int(input())
for i in range(n):
    a, b = input().split()
    try:
        print(int(a)//int(b))
    except ZeroDivisionError as e:
        print('Error Code:', e)
    except ValueError as e:
        print('Error Code:', e)

#Zipped!
n, x = map(int, input().split())
ans = []
for i in range(x):
    ans.append(list(map(float, input().split())))
for i in zip(*ans):
    print(sum(i)/len(i))

#Athlete Sort
nm = input().split()

n = int(nm[0])

m = int(nm[1])
arr = []
for _ in range(n):
    arr.append(list(map(int, input().rstrip().split())))
k = int(input())
arr = sorted(arr, key = lambda x: x[k])
for i in arr:
    print(*i)

#ginortS
s = input()
lower = []
upper = []
odd = []
even = []
for i in s:
    if i.islower():
        lower.append(i)
    elif i.isupper():
        upper.append(i)
    elif int(i) % 2 == 0:
        even.append(i)
    else:
        odd.append(i)
lower = sorted(lower)
upper = sorted(upper)
odd = sorted(odd)
even = sorted(even)
ans = lower + upper + odd + even
print(*ans, sep='')

#Map and Lambda Function
cube = lambda x: x**3 # complete the lambda function 

def fibonacci(n):
    n_1 = 0
    n_2 = 1
    ans = []
    for i in range(n):
        if i == 0:
            temp = 0
        elif i == 1:
            temp = 1
        else:
            temp = ans[-2] + ans[-1]
        ans.append(temp)
        
        
    return ans

if __name__ == '__main__':
    n = int(input())
    print(list(map(cube, fibonacci(n))))

#Detect Floating Point Number
import re
n = int(input())
for i in range(n):
    s = input()
    ans = re.match(r'^[-+]?[0-9]*\.[0-9]+$', s)
    if ans:
        print('True')
    else:
        print('False')

#Re.split()
regex_pattern = r"[.,]+"	# Do not delete 'r'.

import re
print("\n".join(re.split(regex_pattern, input())))

#Group(), Groups() & Groupdict()
import re
s = input()
ans = re.search(r'([a-zA-Z0-9])\1+', s)
if ans:
    print(ans.group(1))
else:
    print(-1)

#Re.findall() & Re.finditer()
import re
s = input()
ans = re.findall(r'(?<=[^aeiouAEIOU])[aeiouAEIOU]{2,}(?=[^aeiouAEIOU])', s)
if ans:
    for i in ans:
        print(i)
else:
    print(-1)

#Validating Roman Numerals
regex_pattern = r"^M{,3}(CM|CD|D?C{,3})(XC|XL|L?X{,3})(IX|IV|V?I{,3})$"	# Do not delete 'r'.

import re
print(str(bool(re.match(regex_pattern, input()))))

#Validating phone numbers
import re
n = int(input())
for i in range(n):
    s = input()
    ans = re.match(r'^[789][0-9]{9}$', s)
    if ans:
        print('YES')
    else:
        print('NO')

#Validating and Parsing Email Addresses
import re
n = int(input())
for i in range(n):
    name, email = input().split()
    ans = re.match(r'<[a-zA-Z]([\w\.\-])+@([a-zA-Z])+\.([a-zA-Z]){1,3}>$', email)
    if ans:
        print(name, email)

#Validating card numbers
import re
n = int(input())
for i in range(n):
    s = input()
    ans = re.match(r'^[456][0-9]{3}(-?)[0-9]{4}\1[0-9]{4}\1[0-9]{4}$', s)
    if ans:
        s = s.replace('-', '')
        ans = re.search(r'([0-9])\1{3,}', s)
        if ans:
            print('Invalid')
        else:
            print('Valid')
    else:
        print('Invalid')

#Re.start() & Re.end()
import re
s = input()
k = input()
ans = re.finditer(r'(?=('+k+'))', s)
flag = False
for i in ans:
    print((i.start(1), i.end(1)-1))
    flag = True
if not flag:
    print((-1, -1))

#XML 1 - Find the score
import sys
import xml.etree.ElementTree as etree

def get_attr_number(node):
    ans = 0
    for i in node.iter():
        ans += len(i.attrib)
    return ans

if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))

#XML2 - Find the Maximum Depth
import xml.etree.ElementTree as etree

maxdepth = 0
def depth(elem, level):
    global maxdepth
    if level == maxdepth:
        maxdepth += 1
    for i in elem:
        depth(i, level+1)

if __name__ == '__main__':
    n = int(input())
    xml = ""
    for i in range(n):
        xml =  xml + input() + "\n"
    tree = etree.ElementTree(etree.fromstring(xml))
    depth(tree.getroot(), -1)
    print(maxdepth)

#Standardize Mobile Number Using Decorators
def wrapper(f):
    def fun(l):
        # complete the function
        f(['+91 ' + i[-10:-5] + ' ' + i[-5:] for i in l])
    return fun

@wrapper
def sort_phone(l):
    print(*sorted(l), sep='\n')

if __name__ == '__main__':
    l = [input() for _ in range(int(input()))]
    sort_phone(l) 

#Decorators 2 - Name Directory

import operator

def person_lister(f):
    def inner(people):
        # complete the function
        return [f(person) for person in sorted(people, key = lambda x: int(x[2]))]
    return inner

@person_lister
def name_format(person):
    return ("Mr. " if person[3] == "M" else "Ms. ") + person[0] + " " + person[1]

if __name__ == '__main__':
    people = [input().split() for i in range(int(input()))]
    print(*name_format(people), sep='\n')


#Arrays
import numpy

def arrays(arr):
    # complete this function
    # use numpy.array
    return numpy.array(arr, float)[ ::-1]

arr = input().strip().split(' ')
result = arrays(arr)
print(result)

#Shape and Reshape
import numpy

ls = list(map(int, input().split()))
ans = numpy.array(ls).reshape(3,3)
print(ans)

#Transpose and Flatten
import numpy

n, m = map(int, input().split())
ls = []
for i in range(n):
    ls.append(list(map(int, input().split())))
ans = numpy.array(ls)
print(ans.transpose())
print(ans.flatten())

#Concatenate

import numpy

n, m, p = map(int, input().split())
ls_1 = []
ls_2 = []
for i in range(n):
    ls_1.append(list(map(int, input().split())))
array_1 = numpy.array(ls_1).reshape(n,p)
for i in range(m):
    ls_2.append(list(map(int, input().split())))
array_2 = numpy.array(ls_2).reshape(m,p)

print(numpy.concatenate((array_1, array_2), axis = 0))

#Zeros and Ones
import numpy

ls = list(map(int, input().split()))
print(numpy.zeros(ls, int))
print(numpy.ones(ls, int))

#Eye and Identity

import numpy
numpy.set_printoptions(legacy='1.13')
n, m = map(int, input().split())
print(numpy.eye(n, m))

#Array Mathematics
import numpy

n, m = map(int, input().split())
ls_1 = []
ls_2 = []
for i in range(n):
    ls_1.append(list(map(int, input().split())))
array_1 = numpy.array(ls_1)
for i in range(n):
    ls_2.append(list(map(int, input().split())))
array_2 = numpy.array(ls_2)
print(array_1 + array_2)
print(array_1 - array_2)
print(array_1 * array_2)
print(array_1 // array_2)
print(array_1 % array_2)
print(array_1 ** array_2)

#Floor, Ceil and Rint
import numpy
numpy.set_printoptions(legacy='1.13')
ls = list(map(float, input().split()))
array = numpy.array(ls)
print(numpy.floor(array))
print(numpy.ceil(array))
print(numpy.rint(array))

#Sum and Prod

import numpy

n, m = map(int, input().split())
ls = []
for i in range(n):
    ls.append(list(map(int, input().split())))
array = numpy.array(ls)
print(numpy.prod(numpy.sum(array, axis = 0)))

#Min and Max

import numpy

n, m = map(int, input().split())
ls = []
for i in range(n):
    ls.append(list(map(int, input().split())))
array = numpy.array(ls)
print(numpy.max(numpy.min(array, axis = 1)))

#Mean, Var, and Std
import numpy

n, m = map(int, input().split())
ls = []
for i in range(n):
    ls.append(list(map(int, input().split())))
array = numpy.array(ls)
print(numpy.mean(array, axis = 1))
print(numpy.var(array, axis = 0))
print(round(numpy.std(array), 11))

#Dot and Cross
import numpy

n = int(input())
ls_1 = []
ls_2 = []
for i in range(n):
    ls_1.append(list(map(int, input().split())))
array_1 = numpy.array(ls_1)
for i in range(n):
    ls_2.append(list(map(int, input().split())))
array_2 = numpy.array(ls_2)
print(numpy.dot(array_1, array_2))

#Inner and Outer

import numpy

array_1 = numpy.array(list(map(int, input().split())))
array_2 = numpy.array(list(map(int, input().split())))
print(numpy.inner(array_1, array_2))
print(numpy.outer(array_1, array_2))

#Polynomials
import numpy

array = numpy.array(list(map(float, input().split())))
x = float(input())
print(numpy.polyval(array, x))

#Linear Algebra

import numpy

n = int(input())
ls = []
for i in range(n):
    ls.append(list(map(float, input().split())))
array = numpy.array(ls)
print(round(numpy.linalg.det(array), 2))



#Algorithms

#Birthday Cake Candles

import sys
from collections import Counter
#
# Complete the 'birthdayCakeCandles' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY candles as parameter.
#

def birthdayCakeCandles(candles):
    # Write your code here
    return Counter(candles)[max(candles)]

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()

#Number Line Jumps

import math
import os
import random
import re
import sys

# Complete the kangaroo function below.

#!/bin/python3

import math
import os
import random
import re
import sys

# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. INTEGER x1
#  2. INTEGER v1
#  3. INTEGER x2
#  4. INTEGER v2
#

def kangaroo(x1, v1, x2, v2):
    # Write your code here
    ans_dec = (x1-x2)%(v2-v1)
    ans = (x1-x2)/(v2-v1)
    if v1 == v2:
        if x1 == x2:
            return 'YES'
        else:
            return 'NO'
    else:
        if ans_dec == 0 and ans > 0:
            return 'YES'
        else:
            return 'NO'

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()

    
#viralAdvertising
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'viralAdvertising' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER n as parameter.
#

def viralAdvertising(n):
    ans = 0
    shared = 5
    for i in range(n):
        liked = math.floor(shared/2)
        ans +=liked
        shared = 3*liked
    return ans
        
        
        

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()

#Recursive Digit Sum
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'superDigit' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. STRING n
#  2. INTEGER k
#

def superDigit(n, k):
    if len(k*n) > 1:
        ans = 0
        for i in n:
            ans += int(i)
        ans = ans*k
        return superDigit(str(ans), 1)
    else:
        return int(n) 
        
    # Write your code here

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]

    k = int(first_multiple_input[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()


#Insertion Sort - Part 1
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'insertionSort1' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#

def insertionSort1(n, arr):
    # Write your code here
    check = arr[-1]
    for i in range(n):
        if check < arr[n-2-i]:
            if n-2-i < 0:
                break
            arr[n-1-i] = arr[n-2-i]
        else:
            arr[n-1-i] = check
            print(*arr, sep=' ')
            break
        print(*arr, sep=' ')
    if check < arr[0]:
        arr[0] = check
        print(*arr, sep=' ')
            

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)

    
#Insertion Sort - Part 2

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'insertionSort2' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#

def insertionSort2(n, arr):
    # Write your code here
    for i in range(1, n):
        check = arr[i]
        for j in range(i):
            if check < arr[i-1-j]:
                arr[i-j] = arr[i-1-j]
            else:
                arr[i-j] = check
                break
        if check < arr[0]:
            arr[0] = check
        print(*arr, sep=' ')
    

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)



    







































