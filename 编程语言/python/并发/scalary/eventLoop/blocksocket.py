root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# cat blocksocket.py 
import socket

s = socket.create_connection(("httpbin.org", 80))
s.send(b"GET /delay/5 HTTP/1.1\r\nHost: httpbin.org\r\n\r\n")
buf = s.recv(1024)
print(buf)
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# python37 blocksocket.py 
b'HTTP/1.1 200 OK\r\nAccess-Control-Allow-Credentials: true\r\nAccess-Control-Allow-Origin: *\r\nContent-Type: application/json\r\nDate: Sat, 04 Jan 2020 07:38:49 GMT\r\nReferrer-Policy: no-referrer-when-downgrade\r\nServer: nginx\r\nX-Content-Type-Options: nosniff\r\nX-Frame-Options: DENY\r\nX-XSS-Protection: 1; mode=block\r\nContent-Length: 198\r\nConnection: keep-alive\r\n\r\n{\n  "args": {}, \n  "data": "", \n  "files": {}, \n  "form": {}, \n  "headers": {\n    "Host": "httpbin.org"\n  }, \n  "origin": "123.127.41.165, 123.127.41.165", \n  "url": "https://httpbin.org/delay/5"\n}\n'
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# 
