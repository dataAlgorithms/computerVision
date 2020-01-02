root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# python37 futuristPeroid.py 
l:  1.0002472400665283
l:  2.0004618167877197
l:  3.0006580352783203
stats: [<Watcher(metrics={'runs': 3, 'elapsed': 0.00013842247426509857, 'elapsed_waiting': 0.00016989931464195251, 'failures': 0, 'successes': 3, 'requested_stop': False}, work=Work(name='__main__.every_one', callback=<function every_one at 0x7f572b105d90>, args=(1577930056.023526,), kwargs={})) object at 0x7f572b0f74a8>, <Watcher(metrics={'runs': 0, 'elapsed': 0, 'elapsed_waiting': 0, 'failures': 0, 'successes': 0, 'requested_stop': False}, work=Work(name='__main__.print_stats', callback=<function print_stats at 0x7f572b105f28>, args=(), kwargs={})) object at 0x7f572b0f7ef0>]
l:  4.000800848007202
l:  5.000994443893433
l:  6.001175165176392
l:  7.00137186050415
stats: [<Watcher(metrics={'runs': 7, 'elapsed': 0.0002713734284043312, 'elapsed_waiting': 0.0003277808427810669, 'failures': 0, 'successes': 7, 'requested_stop': False}, work=Work(name='__main__.every_one', callback=<function every_one at 0x7f572b105d90>, args=(1577930056.023526,), kwargs={})) object at 0x7f572b0f74a8>, <Watcher(metrics={'runs': 1, 'elapsed': 0.00012180395424365997, 'elapsed_waiting': 3.950018435716629e-05, 'failures': 0, 'successes': 1, 'requested_stop': False}, work=Work(name='__main__.print_stats', callback=<function print_stats at 0x7f572b105f28>, args=(), kwargs={})) object at 0x7f572b0f7ef0>]
l:  8.001550197601318
l:  9.00175666809082
l:  10.001925468444824
l:  11.00210690498352
stats: [<Watcher(metrics={'runs': 11, 'elapsed': 0.00041705556213855743, 'elapsed_waiting': 0.0005011735484004021, 'failures': 0, 'successes': 11, 'requested_stop': False}, work=Work(name='__main__.every_one', callback=<function every_one at 0x7f572b105d90>, args=(1577930056.023526,), kwargs={})) object at 0x7f572b0f74a8>, <Watcher(metrics={'runs': 2, 'elapsed': 0.000257895328104496, 'elapsed_waiting': 9.677000343799591e-05, 'failures': 0, 'successes': 2, 'requested_stop': False}, work=Work(name='__main__.print_stats', callback=<function print_stats at 0x7f572b105f28>, args=(), kwargs={})) object at 0x7f572b0f7ef0>]
l:  12.00228476524353
l:  13.002461671829224
l:  14.002643823623657
l:  15.002853155136108
stats: [<Watcher(metrics={'runs': 15, 'elapsed': 0.0005660653114318848, 'elapsed_waiting': 0.0006808815523982048, 'failures': 0, 'successes': 15, 'requested_stop': False}, work=Work(name='__main__.every_one', callback=<function every_one at 0x7f572b105d90>, args=(1577930056.023526,), kwargs={})) object at 0x7f572b0f74a8>, <Watcher(metrics={'runs': 3, 'elapsed': 0.00039380602538585663, 'elapsed_waiting': 0.00015453994274139404, 'failures': 0, 'successes': 3, 'requested_stop': False}, work=Work(name='__main__.print_stats', callback=<function print_stats at 0x7f572b105f28>, args=(), kwargs={})) object at 0x7f572b0f7ef0>]
l:  16.00302505493164
l:  17.0032160282135
l:  18.003432989120483
l:  19.00360894203186
stats: [<Watcher(metrics={'runs': 19, 'elapsed': 0.0007187714800238609, 'elapsed_waiting': 0.0008651772513985634, 'failures': 0, 'successes': 19, 'requested_stop': False}, work=Work(name='__main__.every_one', callback=<function every_one at 0x7f572b105d90>, args=(1577930056.023526,), kwargs={})) object at 0x7f572b0f74a8>, <Watcher(metrics={'runs': 4, 'elapsed': 0.0005132211372256279, 'elapsed_waiting': 0.00019438005983829498, 'failures': 0, 'successes': 4, 'requested_stop': False}, work=Work(name='__main__.print_stats', callback=<function print_stats at 0x7f572b105f28>, args=(), kwargs={})) object at 0x7f572b0f7ef0>]
l:  20.003787994384766
l:  21.003971099853516
l:  22.004157304763794
l:  23.004361867904663
stats: [<Watcher(metrics={'runs': 23, 'elapsed': 0.00086943618953228, 'elapsed_waiting': 0.0010483236983418465, 'failures': 0, 'successes': 23, 'requested_stop': False}, work=Work(name='__main__.every_one', callback=<function every_one at 0x7f572b105d90>, args=(1577930056.023526,), kwargs={})) object at 0x7f572b0f74a8>, <Watcher(metrics={'runs': 5, 'elapsed': 0.0006334912031888962, 'elapsed_waiting': 0.00023525021970272064, 'failures': 0, 'successes': 5, 'requested_stop': False}, work=Work(name='__main__.print_stats', callback=<function print_stats at 0x7f572b105f28>, args=(), kwargs={})) object at 0x7f572b0f7ef0>]
^CTraceback (most recent call last):
  File "futuristPeroid.py", line 16, in <module>
    w.start()
  File "/home/zhouping/anaconda3/lib/python3.7/site-packages/futurist/periodics.py", line 888, in start
    self._run(executor, runner, auto_stop_when_empty)
  File "/home/zhouping/anaconda3/lib/python3.7/site-packages/futurist/periodics.py", line 740, in _run
    _process_scheduled()
  File "/home/zhouping/anaconda3/lib/python3.7/site-packages/futurist/periodics.py", line 662, in _process_scheduled
    self._waiter.wait(when_next)
  File "/home/zhouping/anaconda3/lib/python3.7/threading.py", line 300, in wait
    gotit = waiter.acquire(True, timeout)
KeyboardInterrupt
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# cat futuristPeroid.py 
import time
from futurist import periodics

@periodics.periodic(1)
def every_one(started_at):
    print("l:  %s" % (time.time() - started_at))

w = periodics.PeriodicWorker([
  (every_one, (time.time(),),{}),])

@periodics.periodic(4)
def print_stats():
    print("stats: %s" % list(w.iter_watchers()))

w.add(print_stats)
w.start()
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# 
