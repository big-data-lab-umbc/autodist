import json
import sys
import os, os.path
import errno

# Taken from https://stackoverflow.com/a/600612/119527
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def safe_open_w(path):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    mkdir_p(os.path.dirname(path))
    return open(path, 'w')


def main():

    ipports = [(ip[:-1] + ":5000", ip[:-1] + ":6000") for ip in sys.stdin]
    sys.stdin = open("/dev/tty")

    ipport_workers, ipport_ps = zip(*ipports)

    tasks_workers = []
    tasks_ps = []


    cluster = {
        "worker": ipport_workers,
        "ps": ipport_ps
    }
    #print("How many parameter servers ?")
    #nb_ps = int(input())

    print("Let's configure the workers !")

    for i, ipport in enumerate(ipport_workers):
        print(f"Worker {str(i)} - {ipport} :")
        
        print("Strategy (average):")
        strategy = input()
        if strategy == "":
            strategy = "Median"

        print("Attack (None):")
        attack = input()
        if attack == "":
            attack = "None"
        task = {"type": "worker", "index": i, "strategy": strategy, "attack": attack}


        with safe_open_w("config/" + ipport.split(':')[0] + "/TF_CONFIG_W") as f:
            f.write(json.dumps({
                "cluster": cluster,
                "task": task
            }))
        
    print("Let's configure the parameter servers !")

    
    for i, ipport in enumerate(ipport_ps):
        print(f"PS {str(i)} - {ipport} :")
        
        print("Strategy (average):")
        strategy = input()
        if strategy == "":
            strategy = "Average"

        print("Attack (None):")
        attack = input()
        if attack == "":
            attack = "None"
        task = {"type": "ps", "index": i, "strategy": strategy, "attack": attack}

        with safe_open_w("config/" + ipport.split(':')[0] + "/TF_CONFIG_PS") as f:
            f.write(json.dumps({
                "cluster": cluster,
                "task": task
            }))
        
    
if __name__ == "__main__":
    main()