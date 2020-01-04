root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# cat selectsocket.py 
import select
import socket

s = socket.create_connection(("httpbin.org", 80))
s.setblocking(False)
s.send(b"GET /delay/1 HTTP/1.1\r\nHost: httpbin.org\r\n\r\n")
while True:
    ready_to_read, ready_to_write, in_error = select.select([s], [], [])
    if s in ready_to_read:
        buf = s.recv(1024)
        print(buf)
        break
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# python37 selectsocket.py 
b'HTTP/1.1 200 OK\r\nAccess-Control-Allow-Credentials: true\r\nAccess-Control-Allow-Origin: *\r\nContent-Type: application/json\r\nDate: Sat, 04 Jan 2020 07:39:48 GMT\r\nReferrer-Policy: no-referrer-when-downgrade\r\nServer: nginx\r\nX-Content-Type-Options: nosniff\r\nX-Frame-Options: DENY\r\nX-XSS-Protection: 1; mode=block\r\nContent-Length: 198\r\nConnection: keep-alive\r\n\r\n{\n  "args": {}, \n  "data": "", \n  "files": {}, \n  "form": {}, \n  "headers": {\n    "Host": "httpbin.org"\n  }, \n  "origin": "123.127.41.165, 123.127.41.165", \n  "url": "https://httpbin.org/delay/1"\n}\n'
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# 
