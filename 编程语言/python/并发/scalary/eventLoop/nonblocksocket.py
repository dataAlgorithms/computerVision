root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# cat nonblocksocket.py 
import socket
s = socket.create_connection(("httpbin.org", 80))
s.setblocking(False)
s.send(b"GET /delay/5 HTTP/1.1\r\nHost: httpbin.org\r\n\r\n")
buf = s.recv(1024)
print(buf)
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# python37 nonblocksocket.py 
Traceback (most recent call last):
  File "nonblocksocket.py", line 5, in <module>
    buf = s.recv(1024)
BlockingIOError: [Errno 11] Resource temporarily unavailable
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# 
