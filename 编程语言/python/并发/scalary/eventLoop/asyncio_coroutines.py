root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# python37 asyncio_coroutines.py 
hello world!
Adding 42
65
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# cat asyncio_coroutines.py 
import asyncio

async def add_42(number):
    print("Adding 42")
    return 42 + number

async def hello_world():
    print("hello world!")
    result = await add_42(23)
    return result

event_loop = asyncio.get_event_loop()
try:
    result = event_loop.run_until_complete(hello_world())
    print(result)
finally:
    event_loop.close()
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# 
