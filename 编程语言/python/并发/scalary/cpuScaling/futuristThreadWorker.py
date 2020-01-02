root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# python37 futuristThreadWorker.py 
<ExecutorStatistics object at 0x7efd0582b240 (failures=0, executed=5, runtime=0.60, cancelled=0)>
<ExecutorStatistics object at 0x7efd0582b360 (failures=0, executed=8, runtime=0.79, cancelled=0)>
Results: [504846, 506437, 504148, 509405, 510836, 504985, 507168, 506300]
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# 
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# 
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# cat futuristThreadWorker.py 
import futurist
from futurist import waiters
import random

def compute():
    return sum([random.randint(1, 100) for i in range(10000)])

with futurist.ThreadPoolExecutor(max_workers=8) as executor:
    futs = [executor.submit(compute) for _ in range(8)]
    print(executor.statistics)

results = waiters.wait_for_all(futs)
print(executor.statistics)
print('Results: %s' % [r.result() for r in results.done])
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# 
