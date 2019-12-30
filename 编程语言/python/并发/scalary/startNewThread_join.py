root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# cat startNewThread_join.py 
import threading

def print_something(something):
    print(something)

t = threading.Thread(target=print_something, args=("hello",))
t.start()
print("thread started")
t.join()
