import time
import asyncio

async def main():
    print(f'{time.ctime()} Hello!')
    await asyncio.sleep(1.0)
    print(f'{time.ctime()} Goodbye!')
    loop.stop()

loop = asyncio.get_event_loop()
loop.create_task(main())
loop.run_forever()
pending = asyncio.Task.all_tasks(loop=loop)
group = asyncio.gather(*pending,return_exceptions=True)
loop.run_until_complete(group)
loop.close()

# 输出
root@deepface-19:/data/zhouping/anaconda3/nn/asycioLearn# python36 quickstart.py 
Tue Oct 29 09:14:48 2019 Hello!
Tue Oct 29 09:14:49 2019 Goodbye!
