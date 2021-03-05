import json
import sys


def main():

    ipports = [ip[:-1] + ":5000" for ip in sys.stdin]
    sys.stdin = open("/dev/tty")

    ipport_workers = []
    ipport_ps = []
    tasks_workers = []
    tasks_ps = []

    print("How many workers ?")
    nb_workers = int(input())

    if nb_workers > len(ipports):
        print("There are more workers than available nodes.")
        exit(0)

    ipport_workers = ipports[:nb_workers]
    ipport_ps = ipports[nb_workers:]
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
            strategy = "average"

        print("Attack (None):")
        attack = input()
        if attack == "":
            attack = "None"
        task = {"type": "worker", "index": i, "strategy": strategy, "attack": attack}

        f = open("config/TF_CONFIG_" + ipport.split(':')[0], "w")
        f.write(json.dumps({
            "cluster": cluster,
            "task": task
        }))
        f.close

    print("Let's configure the parameter servers !")

    
    for i, ipport in enumerate(ipport_ps):
        print(f"PS {str(i)} - {ipport} :")
        
        print("Strategy (average):")
        strategy = input()
        if strategy == "":
            strategy = "average"

        print("Attack (None):")
        attack = input()
        if attack == "":
            attack = "None"
        task = {"type": "ps", "index": i, "strategy": strategy, "attack": attack}

        f = open("config/TF_CONFIG_" + ipport.split(':')[0], "w")
        f.write(json.dumps({
            "cluster": cluster,
            "task": task
        }))
        f.close

    
if __name__ == "__main__":
    main()