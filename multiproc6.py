import time
import torch
import sys
import subprocess

argslist = list(sys.argv)[1:]
num_gpus = 20
argslist.append('--n_gpus={}'.format(num_gpus))
workers = []
job_id = time.strftime("%Y_%m_%d-%H%M")
argslist.append("--group_name=group_{}".format(job_id))

for i in [12,13]:
    argslist.append('--rank={}'.format(i))
    stdout = None if i == 0 else open("log/{}_GPU_{}.log".format(job_id, i),
                                      "w")
    print(argslist)
    p = subprocess.Popen([str(sys.executable)]+argslist, stdout=stdout)
    workers.append(p)
    argslist = argslist[:-1]

for p in workers:
    p.wait()
