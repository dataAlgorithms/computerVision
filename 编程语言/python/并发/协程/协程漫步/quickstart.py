root@ubuntu:/home/zhouping/anaconda3/nn/asyncioLn# cat quickstart.py 
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
group = asyncio.gather(*pending, return_exceptions=True)
loop.run_until_complete(group)
loop.close()
root@ubuntu:/home/zhouping/anaconda3/nn/asyncioLn# python3 quickstart.py 
Fri Nov 29 09:29:13 2019 Hello!
Fri Nov 29 09:29:14 2019 Goodbye!
root@ubuntu:/home/zhouping/anaconda3/nn/asyncioLn# 

'''
loop = asyncio.get_event_loop()
You need a loop instance before you can run any coroutines, and this is how you get one

task = loop.create_task(coro)
Your coroutine function will not be executed until you do this

loop.run_until_complete(coro) and loop.run_forever()
run_until_complete 来运行 loop ，等到 future 完成，run_until_complete 也就返回了。
run_forever 会一直运行，直到 stop 被调用（在python3.7中已经取消了）

loop.stop() and loop.close()
stop() is usually called as a consequence of some kind of shutdown signal being received,
and close() is usually the final action: it must be called on a stopped loop, and it will
clear all queues and shut down the Executor. A “stopped” loop can be restarted; a “closed”
loop is gone for good
'''

#######################################
asyncio run_until_complete和run_forever运行对比
#######################################
第一种运行方式
root@ubuntu:/home/zhouping/anaconda3/nn/asyncioLn# python3 run_until_complete.py 
2019-11-29 09:36:53,940 [*] MainProcess  MainThread  Waiting :1
2019-11-29 09:36:54,942 [*] MainProcess  MainThread  Done :1
2019-11-29 09:36:54,942 [*] MainProcess  MainThread  Waiting :3
2019-11-29 09:36:57,945 [*] MainProcess  MainThread  Done :3
2019-11-29 09:36:57,946 [*] MainProcess  MainThread  <程序退出> 总用时：4.005772113800049
root@ubuntu:/home/zhouping/anaconda3/nn/asyncioLn# cat run_until_complete.py 
import asyncio
import functools
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [*] %(processName)s  %(threadName)s  %(message)s"
)


# def done_callback (loop, futu):
#     loop.stop()


async def work01 (x):
    logging.info(f'Waiting :{str(x)}')
    await asyncio.sleep(x)
    logging.info(f'Done :{str(x)}')


# async def work02 (
#         loop,  # 第二种运行方式
#         x
# ):
#     logging.info(f'Waiting :{str(x)}')
#     await asyncio.sleep(x)
#     logging.info(f'Done :{str(x)}')
#     loop.stop()  # 第二种运行方式


if __name__ == '__main__':
    start = time.time()
    
    loop = asyncio.get_event_loop()
    # 第一种运行方式
    loop.run_until_complete(work01(1))
    loop.run_until_complete(work01(3))
    # 第二种运行方式( 第二个协程没结束，loop 就停止了——被先结束的那个协程给停掉的。)
    # asyncio.ensure_future(work02(loop, 1))
    # asyncio.ensure_future(work02(loop, 3))
    # 解决第二种运行方式的最佳方法
    # futus = asyncio.gather(work02(loop, 1), work02(loop, 3))
    # futus.add_done_callback(functools.partial(done_callback, loop))
    
    # loop.run_forever()
    loop.close()
    logging.info(f"<程序退出> 总用时：{time.time() - start}")

第二种运行方式( 第二个协程没结束，loop 就停止了——被先结束的那个协程给停掉的。)
root@ubuntu:/home/zhouping/anaconda3/nn/asyncioLn# cat run_forever.py 
# coding=utf-8
import asyncio
import functools
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [*] %(processName)s  %(threadName)s  %(message)s"
)


# def done_callback (loop, futu):
#     loop.stop()


# async def work01 (x):
#     logging.info(f'Waiting :{str(x)}')
#     await asyncio.sleep(x)
#     logging.info(f'Done :{str(x)}')


async def work02 (
        loop,  # 第二种运行方式
        x
):
    logging.info(f'Waiting :{str(x)}')
    await asyncio.sleep(x)
    logging.info(f'Done :{str(x)}')
    loop.stop()  # 第二种运行方式


if __name__ == '__main__':
    start = time.time()
    
    loop = asyncio.get_event_loop()
    # 第一种运行方式
    # loop.run_until_complete(work01(1))
    # loop.run_until_complete(work01(3))
    # 第二种运行方式( 第二个协程没结束，loop 就停止了——被先结束的那个协程给停掉的。)
    asyncio.ensure_future(work02(loop, 1))
    asyncio.ensure_future(work02(loop, 3))
    # 解决第二种运行方式的最佳方法
    # futus = asyncio.gather(work02(loop, 1), work02(loop, 3))
    # futus.add_done_callback(functools.partial(done_callback, loop))
    
    loop.run_forever()
    # loop.close()
    logging.info(f"<程序退出> 总用时：{time.time() - start}")
root@ubuntu:/home/zhouping/anaconda3/nn/asyncioLn# python3 run_forever.py 
2019-11-29 09:40:08,548 [*] MainProcess  MainThread  Waiting :1
2019-11-29 09:40:08,548 [*] MainProcess  MainThread  Waiting :3
2019-11-29 09:40:09,549 [*] MainProcess  MainThread  Done :1
2019-11-29 09:40:09,549 [*] MainProcess  MainThread  <程序退出> 总用时：1.002007007598877

第二种运行方式的最佳方法
root@ubuntu:/home/zhouping/anaconda3/nn/asyncioLn# cat run_forever_best.py 
# coding=utf-8
import asyncio
import functools
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [*] %(processName)s  %(threadName)s  %(message)s"
)


def done_callback (loop, futu):
    loop.stop()


# async def work01 (x):
#     logging.info(f'Waiting :{str(x)}')
#     await asyncio.sleep(x)
#     logging.info(f'Done :{str(x)}')


async def work02 (
        loop,  # 第二种运行方式
        x
):
    logging.info(f'Waiting :{str(x)}')
    await asyncio.sleep(x)
    logging.info(f'Done :{str(x)}')
    # loop.stop()  # 第二种运行方式


if __name__ == '__main__':
    start = time.time()
    
    loop = asyncio.get_event_loop()
    # 第一种运行方式
    # loop.run_until_complete(work01(1))
    # loop.run_until_complete(work01(3))
    # 第二种运行方式( 第二个协程没结束，loop 就停止了——被先结束的那个协程给停掉的。)
    # asyncio.ensure_future(work02(loop, 1))
    # asyncio.ensure_future(work02(loop, 3))
    # 解决第二种运行方式的最佳方法
    futus = asyncio.gather(work02(loop, 1), work02(loop, 3))
    futus.add_done_callback(functools.partial(done_callback, loop))
    
    loop.run_forever()
    loop.close()
    logging.info(f"<程序退出> 总用时：{time.time() - start}")

root@ubuntu:/home/zhouping/anaconda3/nn/asyncioLn# python3 run_forever_best.py 
2019-11-29 09:41:31,555 [*] MainProcess  MainThread  Waiting :1
2019-11-29 09:41:31,555 [*] MainProcess  MainThread  Waiting :3
2019-11-29 09:41:32,556 [*] MainProcess  MainThread  Done :1
2019-11-29 09:41:34,557 [*] MainProcess  MainThread  Done :3
2019-11-29 09:41:34,558 [*] MainProcess  MainThread  <程序退出> 总用时：3.003403902053833
root@ubuntu:/home/zhouping/anaconda3/nn/asyncioLn# 
