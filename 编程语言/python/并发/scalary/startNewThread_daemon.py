root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# python37 startNewThread_daemon.py 
hello
thread started
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# cat startNewThread_daemon.py 
import threading

def print_something(something):
    print(something)

t = threading.Thread(target=print_something, args=("hello",))
t.daemon = True
t.start()
print("thread started")
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# 
