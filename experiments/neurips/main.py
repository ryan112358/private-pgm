import sys
import argparse
import itertools
import subprocess
import time
import numpy as np

def run_sequentially(commands):
    for cmd in commands:
        print('Running command:', cmd)
        subprocess.run(cmd, shell=True)

def run_with_slurm(commands, sbatch):
    opts = ' '.join(['--%s %s' % (k,v) for k,v in sbatch.items()])
    slurm = 'sbatch --job-name %s -n 1 --cpu_per_task 2 --time 30'
    for cmd in commands:
        cmd = 'sbatch %s --wrap "%s"' % (opts, cmd)
        subprocess.run(cmd, shell=True) 
        time.sleep(0.05)

def generate_all_commands(base, args):
    """
    :param base: the command to run the python file
    :param args: the configurations to use
        keys are arguments
        values are list possible settings
    
    Note: runs cartesian product of commands implied by args
    """
    keys = args.keys()
    vals = args.values()
    commands = []
    for config in itertools.product(*vals):
        opts = [' --%s %s' % (k,v) for k,v in zip(keys, config)]
        commands.append( base + ''.join(opts) )

    return commands

def gum():
    args = {}
    trials = 5
    args['seed'] = [0]
    args['dataset'] = ['adult'] #['fire','adult','loans','stroke','msnbc','titanic']
    args['measurements'] = [32] #[1,2,4,8,16,32,64,128,256,512]
    args['epsilon'] = [0.01]#list(np.logspace(-2,2,9))
    args['kway'] = [2]
    args['oracle'] = ['convex'] #'gum','exact','convex']
    args['iters'] = [10000]
    args['save'] = ['results/gum3.csv']
    
    commands = trials*generate_all_commands('python gum_compare.py', args)

    sbatch = {}
    sbatch['job-name'] = 'gum'
    sbatch['ntasks'] = 1
    sbatch['cpus-per-task'] = 1
    sbatch['partition'] = 'defq'
    sbatch['mem'] = 2048
    sbatch['time'] = '12:00:00'

    return commands, sbatch



def exact_vs_approx_small():
    args = {}
    args['seed'] = [2457026461, 2643031147, 2094276510, 2378741324, 1961979125]
    args['attributes'] = [8]
    args['domain'] = [4]
    args['temperature'] = [3.0, 0.5]
    args['measurements'] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30,35,40,45,50,55]
    args['oracle'] = ['approx']
    args['iters'] = [10000]
    args['save'] = ['results/exact_vs_approx.csv']
    
    commands = generate_all_commands('python exact_vs_approx.py', args)
    
    sbatch = {}
    sbatch['job-name'] = 'exact-approx-small'
    sbatch['ntasks'] = 1
    sbatch['cpus-per-task'] = 2
    sbatch['partition'] = 'defq'
    sbatch['mem'] = 16384
    sbatch['time'] = '4:00:00'

    return commands, sbatch

def exact_vs_approx_large():
    args = {}
    args['seed'] = [0]
    args['attributes'] = [100]
    args['domain'] = [10]
    #args['measurements'] = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
    args['measurements'] = list(np.unique(np.logspace(0,4,50).astype(int)))
    args['oracle'] = ['exact', 'approx']
    args['selection'] = ['random','greedy']
    args['iters'] = [25]
    args['learning_rate'] = [1e-5]
    args['skiperror'] = ['']
    args['save'] = ['results/exact_vs_approx_large.csv']
    
    commands = generate_all_commands('python exact_vs_approx.py', args)

    #args['oracle'] = ['approx']
    #args['measurements'] += [125,150,175,200,225,250,300,400,500,600,700,800,900,1000]
    #args['measurements'] += [1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    #commands += generate_all_commands('python exact_vs_approx.py', args)
    
    sbatch = {}
    sbatch['job-name'] = 'exact-approx-large'
    sbatch['ntasks'] = 1
    sbatch['cpus-per-task'] = 2
    sbatch['partition'] = 'defq'
    sbatch['mem'] = 16384
    sbatch['time'] = '2:00:00'

    return commands, sbatch

def mwem_new():
    args = {}
    args['engine'] = ['MW', 'MB-exact', 'MB-convex']
    args['dataset'] = ['fire']
    args['epsilon_per_iter'] = [0.1, 0.1/2, 0.1*2]
    args['iters'] = [2500]
    args['seed'] = [2457026461, 2643031147, 2094276510, 2378741324, 1961979125] 
    args['save'] = ['']
    
    commands = generate_all_commands('python mwem_new.py', args)
    
    sbatch = {}
    sbatch['job-name'] = 'mwem'
    sbatch['ntasks'] = 1
    sbatch['cpus-per-task'] = 2
    sbatch['partition'] = 'longq'
    sbatch['mem'] = 16384
    sbatch['time'] = '72:00:00'

    return commands, sbatch




def hdmm_new():
    args = {}
    trials = 4
    args['oracle'] = ['convex']
    args['dataset'] = ['fire']
    args['workload'] = [128] #1,2,4,8,16,32,64,128,256,455]
    args['epsilon'] = [1.0]
    args['delta'] = [0.0]
    args['restarts'] = [1]
    args['iters'] = [10000]
    args['seed'] = [0]
    args['save'] = ['results/hdmm_new.csv']
    
    commands = trials*generate_all_commands('python hdmm_new.py', args)
    
    sbatch = {}
    sbatch['job-name'] = 'hdmm'
    sbatch['ntasks'] = 1
    sbatch['cpus-per-task'] = 2
    sbatch['partition'] = 'longq'
    sbatch['mem'] = 16384
    sbatch['time'] = '48:00:00'

    return commands, sbatch

def mwem_no_noise():
    args = {}
    args['engine'] = ['approx']
    args['dataset'] = ['fire','adult','titanic','msnbc','loans','stroke']
    args['save'] = ['']
    
    commands = generate_all_commands('python mwem_no_noise.py', args)
    
    sbatch = {}
    sbatch['job-name'] = 'mwem-inf'
    sbatch['ntasks'] = 1
    sbatch['cpus-per-task'] = 2
    sbatch['partition'] = 'longq'
    sbatch['mem'] = 16384
    sbatch['time'] = '8:00:00'

    return commands, sbatch

def fem():
    args = {}
    args['iters'] = [1000]
    args['epsilon'] = [0.1, 0.15, 0.2, 0.25, 0.5, 1]
    args['seed'] = [2457026461, 2643031147, 2094276510, 2378741324, 1961979125] 
    args['save'] = ['results/fem2.csv']
    
    commands = generate_all_commands('python fem.py', args)
    
    sbatch = {}
    sbatch['job-name'] = 'fem'
    sbatch['ntasks'] = 1
    sbatch['cpus-per-task'] = 2
    sbatch['partition'] = 'longq'
    sbatch['mem'] = 16384
    sbatch['time'] = '72:00:00'

    return commands, sbatch
   

def mwem_experiments():
    args = {}
    args['seed'] = [2457026461, 2643031147, 2094276510, 2378741324, 1961979125]
    args['dataset'] = ['adult', 'titanic', 'loans', 'stroke']
    args['engine'] = ['MW', 'MB']
    args['save'] = ['']
    
    commands = generate_all_commands('timeout 120m python mwem.py', args)
    
    sbatch = {}
    sbatch['job-name'] = 'mwem'
    sbatch['ntasks'] = 1
    sbatch['cpus-per-task'] = 2
    sbatch['partition'] = 'defq'
    sbatch['mem'] = 16384
    sbatch['time'] = '2:00:00'

    return commands, sbatch

def mwem2_experiments():
    args = {}
    args['seed'] = [2457026461, 2643031147, 2094276510, 2378741324, 1961979125]
    args['dataset'] = ['adult', 'titanic', 'loans', 'stroke']
    args['engine'] = ['MW', 'MB', 'CVX']
    args['epsilon'] = [0.1, 0.31622776601683794, 1.0, 3.1622776601683795, 10.0]
    args['save'] = ['']
    
    commands = generate_all_commands('python mwem_accuracy.py', args)
    
    sbatch = {}
    sbatch['job-name'] = 'mwem'
    sbatch['ntasks'] = 1
    sbatch['cpus-per-task'] = 2
    sbatch['partition'] = 'defq'
    sbatch['mem'] = 16384
    sbatch['time'] = '4:00:00'

    return commands, sbatch

def privbayes_experiments():
    args = {}
    args['seed'] = [162672562, 35172897, 1330747486, 1511511086, 3455968]
    args['dataset'] = ['adult', 'titanic', 'loans', 'stroke']
    args['epsilon'] = [0.1, 0.31622776601683794, 1.0, 3.1622776601683795, 10.0]
    #args['dataset'] = ['msnbc', 'stroke']
    args['save'] = ['']

    commands = generate_all_commands('python privbayes.py', args)

    sbatch = {}
    sbatch['job-name'] = 'privbayes'
    sbatch['ntasks'] = 1
    sbatch['cpus-per-task'] = 2
    sbatch['partition'] = 'defq'
    sbatch['mem'] = 16384
    sbatch['time'] = '6:00:00'

    return commands, sbatch

def hdmm_experiments():
    args = {}
    args['seed'] = [468130658, 1928512456, 804613545, 1106580885, 1713056646]
    args['dataset'] = ['adult', 'titanic', 'loans', 'stroke']
    args['epsilon'] = [0.1, 0.31622776601683794, 1.0, 3.1622776601683795, 10.0]
    args['save'] = ['']

    commands = generate_all_commands('python hdmm1.py', args)

    sbatch = {}
    sbatch['job-name'] = 'hdmm'
    sbatch['ntasks'] = 1
    sbatch['cpus-per-task'] = 2
    sbatch['partition'] = 'defq'
    sbatch['mem'] = 16384
    sbatch['time'] = '6:00:00'

    return commands, sbatch

def sensitivity_experiments():
    args = {}
    args['seed'] = [468130658, 1928512456, 804613545, 1106580885, 1713056646]
    args['dataset'] = ['adult']
    args['epsilon'] = [0.01, 0.03162277660168379, 0.1, 0.31622776601683794, 1.0, 3.1622776601683795, 10.0]
    args['save'] = ['']

    commands = generate_all_commands('python hdmm.py', args)

    sbatch = {}
    sbatch['job-name'] = 'sensitivity'
    sbatch['ntasks'] = 1
    sbatch['cpus-per-task'] = 2
    sbatch['partition'] = 'defq'
    sbatch['mem'] = 16384
    sbatch['time'] = '6:00:00'
    
    return commands, sbatch

def scalability_experiments():
    args = {}
    args['columns'] = [3,4,5,6,7,8]
    args['engine'] = ['MB','LS','MW']
    args['iters'] = [25]
    args['save'] = ['']

    commands = generate_all_commands('python scalability.py', args)

    args['columns'] = [10,20,40,60,80,100,200,400,600,800,1000]
    args['engine'] = ['MB']
    args['iters'] = [25]
    args['save'] = ['']

    commands += generate_all_commands('python scalability.py', args)

    sbatch = {}
    sbatch['job-name'] = 'scalability'
    sbatch['ntasks'] = 1
    sbatch['cpus-per-task'] = 2
    sbatch['partition'] = 'defq'
    sbatch['mem'] = 16384
    sbatch['time'] = '2:00:00'
    
    return commands, sbatch

def dualquery_experiments():
    args = {}
    args['seed'] = [1137248860, 1582439880, 319854320, 242657693, 525881409]
    args['dataset'] = ['adult','titanic', 'loans', 'stroke']
    args['epsilon'] = [0.1, 0.31622776601683794, 1.0, 3.1622776601683795, 10.0]
    args['save'] = ['']

    commands = generate_all_commands('python dual_query.py', args)

    sbatch = {}
    sbatch['job-name'] = 'dq'
    sbatch['ntasks'] = 1
    sbatch['cpus-per-task'] = 2
    sbatch['partition'] = 'longq'
    sbatch['mem'] = 16384
    sbatch['time'] = '24:00:00'
    
    return commands, sbatch

if __name__ == '__main__':
    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('experiment', choices=['mwem', 'privbayes', 'hdmm', 'dualquery', 'mwem2', 'convergence', 'sensitivity', 'scalability', 'exact_vs_approx_small', 'exact_vs_approx_large', 'hdmm_new','mwem_no_noise','fem', 'mwem_new', 'gum'], help='experiment to run')
    parser.add_argument('--slurm', action='store_true', help='run commands on slurm')

    args = parser.parse_args()

    if args.experiment == 'mwem':
        commands, sbatch = mwem_experiments()
    elif args.experiment == 'privbayes':
        commands, sbatch = privbayes_experiments()
    elif args.experiment == 'hdmm':
        commands, sbatch = hdmm_experiments()
    elif args.experiment == 'dualquery':
        commands, sbatch = dualquery_experiments()
    elif args.experiment == 'mwem2':
        commands, sbatch = mwem2_experiments()
    elif args.experiment == 'sensitivity':
        commands, sbatch = sensitivity_experiments()
    elif args.experiment == 'scalability':
        commands, sbatch = scalability_experiments()
    elif args.experiment == 'exact_vs_approx_small':
        commands, sbatch = exact_vs_approx_small()
    elif args.experiment == 'exact_vs_approx_large':
        commands, sbatch = exact_vs_approx_large()
    elif args.experiment == 'hdmm_new':
        commands, sbatch = hdmm_new()
    elif args.experiment == 'mwem_no_noise':
        commands, sbatch = mwem_no_noise()
    elif args.experiment == 'fem':
        commands, sbatch = fem()
    elif args.experiment == 'mwem_new':
        commands, sbatch = mwem_new()
    elif args.experiment == 'gum':
        commands, sbatch = gum()

    if args.slurm:
        run_with_slurm(commands, sbatch)
    else:
        run_sequentially(commands)   
 
