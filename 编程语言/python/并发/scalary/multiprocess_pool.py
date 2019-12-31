root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# python37 multiprocess_pool.py 
Results: [50423732, 50557384, 50483463, 50544934, 50472001, 50482985, 50506092, 50497032]
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# cat multiprocess_pool.py 
import multiprocessing
import random

def compute(n):
    return sum(
    [random.randint(1, 100) for i in range(1000000)])

# Start 8 workers
pool = multiprocessing.Pool(processes=8)
print("Results: %s" % pool.map(compute, range(8)))
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# 
