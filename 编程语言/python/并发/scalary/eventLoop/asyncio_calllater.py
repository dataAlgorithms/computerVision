root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# python37 asyncio_calllater.py 
Hello world!
^CTraceback (most recent call last):
  File "asyncio_calllater.py", line 8, in <module>
    loop.run_forever()
  File "/home/zhouping/anaconda3/lib/python3.7/asyncio/base_events.py", line 539, in run_forever
    self._run_once()
  File "/home/zhouping/anaconda3/lib/python3.7/asyncio/base_events.py", line 1739, in _run_once
    event_list = self._selector.select(timeout)
  File "/home/zhouping/anaconda3/lib/python3.7/selectors.py", line 468, in select
    fd_event_list = self._selector.poll(timeout, max_ev)
KeyboardInterrupt
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# cat asyncio_calllater.py 
import asyncio

def hello_world():
    print("Hello world!")

loop = asyncio.get_event_loop()
loop.call_later(1, hello_world)
loop.run_forever()
