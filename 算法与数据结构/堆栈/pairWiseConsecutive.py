# Python3 program to check if successive  
# pair of numbers in the stack are  
# consecutive or not  
  
# Function to check if elements are  
# pairwise consecutive in stack  
def pairWiseConsecutive(s): 
      
    # Transfer elements of s to aux.  
    aux = [] 
    while (len(s) != 0):  
        aux.append(s[-1])  
        s.pop() 
  
    # Traverse aux and see if elements  
    # are pairwise consecutive or not.  
    # We also need to make sure that  
    # original content is retained.  
    result = True
    while (len(aux) > 1):  
  
        # Fetch current top two  
        # elements of aux and check  
        # if they are consecutive.  
        x = aux[-1]  
        aux.pop()  
        y = aux[-1]  
        aux.pop()  
        if (abs(x - y) != 1):  
            result = False
  
        # append the elements to  
        # original stack.  
        s.append(x)  
        s.append(y) 
  
    if (len(aux) == 1):  
        s.append(aux[-1])  
  
    return result 
  
# Driver Code 
if __name__ == '__main__': 
  
    s = [] 
    s.append(4)  
    s.append(5)  
    s.append(-2)  
    s.append(-3)  
    s.append(11)  
    s.append(10)  
    s.append(5)  
    s.append(6)  
    s.append(20)  
    s.append(11)  
  
    if (pairWiseConsecutive(s)):  
        print("Yes")  
    else: 
        print("No") 
  
    print("Stack content (from top)", 
               "after function call")  
    while (len(s) != 0): 
        print(s[-1], end = " ")  
        s.pop() 
