root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# python37 futuristLimitQueueSize.py 
<ExecutorStatistics object at 0x7f9dadcd2d38 (failures=0, executed=5, runtime=0.01, cancelled=0)>
<ExecutorStatistics object at 0x7f9dadcd2dc8 (failures=0, executed=6, runtime=0.02, cancelled=0)>
Results: [50137, 51061, 51930, 49412, 51424, 50501]
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# vi futuristLimitQueueSize.py       
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# 
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# python37 futuristLimitQueueSize.py 
<ExecutorStatistics object at 0x7f044dd70288 (failures=0, executed=6, runtime=0.02, cancelled=0)>
<ExecutorStatistics object at 0x7f044dd6ae58 (failures=0, executed=7, runtime=0.02, cancelled=0)>
Results: [50798, 50501, 51242, 51106, 49819, 49959, 50577]
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# vi futuristLimitQueueSize.py       
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# 
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# python37 futuristLimitQueueSize.py 
<ExecutorStatistics object at 0x7fa9821c4708 (failures=0, executed=7, runtime=0.02, cancelled=0)>
<ExecutorStatistics object at 0x7fa9821c4798 (failures=0, executed=8, runtime=0.02, cancelled=0)>
Results: [51079, 50634, 50382, 50288, 49983, 50898, 51085, 50313]
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# vi futuristLimitQueueSize.py       
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# python37 futuristLimitQueueSize.py 
<ExecutorStatistics object at 0x7f1e1dd7c6c0 (failures=0, executed=7, runtime=0.02, cancelled=0)>
<ExecutorStatistics object at 0x7f1e1dd7c6c0 (failures=0, executed=9, runtime=0.03, cancelled=0)>
Results: [50880, 49563, 50865, 50442, 49168, 50492, 48816, 52353, 50106]
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# 
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# vi futuristLimitQueueSize.py       
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# 
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# 
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# python37 futuristLimitQueueSize.py 
Traceback (most recent call last):
  File "futuristLimitQueueSize.py", line 12, in <module>
    futs = [executor.submit(compute) for _ in range(10)]
  File "futuristLimitQueueSize.py", line 12, in <listcomp>
    futs = [executor.submit(compute) for _ in range(10)]
  File "/home/zhouping/anaconda3/lib/python3.7/site-packages/futurist/_futures.py", line 188, in submit
    self._check_and_reject(self, self._work_queue.qsize())
  File "/home/zhouping/anaconda3/lib/python3.7/site-packages/futurist/rejection.py", line 30, in _rejector
    max_backlog))
futurist._futures.RejectedSubmission: Current backlog 2 is not allowed to go beyond 2
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# cat futuristLimitQueueSize.py 
import futurist
from futurist import rejection
import random

def compute():
    return sum(
         [random.randint(1, 100) for i in range(1000)])

with futurist.ThreadPoolExecutor(
    max_workers=8,
    check_and_reject=rejection.reject_when_reached(2)) as executor:
    futs = [executor.submit(compute) for _ in range(10)]
    print(executor.statistics)

results = [f.result() for f in futs]
print(executor.statistics)
print('Results: %s' % results)
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# 
