bash_pre = '''#!/bin/bash
#
# Please no empty lines before the PBS instructions.
#
#--- job name:
#PBS -N corr_spec_ph1_east
#--- allocate 1 nodes using 8 core each
#PBS -l nodes=1:ppn=8
#--- request 5 minutes wall time (format: dd:mm:ss)
#PBS -l walltime=08:00:00
#--- select queue: workq
#PBS -q workq
#--- set e-mail address for job notifications
#PBS -M lalc@dtu.dk
#--- request notifications for (a)bort, (b)egin and (e)nd of execution
#PBS -m abe

cd /mnt/mimer/lalc/repository_v1
module load easy
module load  Anaconda3/4.2.0
source activate v0 

'''
name = 'rec_all_cluster_phase_1_east'

for i in range(1,21):    

    run_n = str(int(i))

    
    with open ('C:/Users/lalc/Documents/PhD/Python Code/repository_v1/data_process/codes/job_corr_east_spec_ph1_sh_d'+run_n+'.sh', 'w') as rsh:
        rsh.write(bash_pre+'python ' + name + '.py ' + str(i-1))


#!/bin/bash
# declare an array called array and define 3 vales
cd /mnt/mimer/lalc/repository_v1
array=( 1  2  3  4  5  6  7  8  9 10 11 12 13 14 16 17 18 19 20 21)
for i in "${array[@]}"
do
	qsub 'job_corr_east_spec_ph1_sh_d'$i'.sh'
done

















