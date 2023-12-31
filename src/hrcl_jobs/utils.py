import subprocess 

class machineResources:
    def __init__(
        self,
        name: str,
        cores: int,
        threads: int,
        memory: int,
        memory_per_core="4 gb",
        omp_threads: int = 2,
    ):  # GB
        self.name = name
        self.cores = cores
        self.threads = threads
        self.memory = memory  # Total Memory
        self.memory_per_thread = memory_per_core
        self.omp_threads = omp_threads


def machine_list_resources(rank_0_one_thread=True) -> []:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    machines = {
        "ds2": machineResources("ds2", 9, 9, 80),
        "hex6": machineResources("hex6", 6, 6, 62),
        "hex8": machineResources("hex8", 6, 6, 62),
        "hex9": machineResources("hex9", 6, 6, 58),
        "hex11": machineResources("hex11", 6, 6, 62),
    }
    uname_n = subprocess.check_output("uname -n", shell=True).decode("utf-8").strip()

    machine = machines[uname_n]

    name = machine.name
    cores = machine.cores
    threads = machine.threads
    memory = machine.memory

    unames = comm.allgather(uname_n)
    current_machine_cnt = 0
    start_rank = 0
    start = True
    for n, i in enumerate(unames):
        if start and i == uname_n:
            current_machine_cnt += 1
            end_rank = n
            continue
        elif i == uname_n:
            current_machine_cnt += 1
            start == True
            start_rank = n

    if current_machine_cnt == 0:
        raise ValueError("No machines found")

    on_rank_0 = False
    if uname_n == unames[0]:
        on_rank_0 = True

    if rank_0_one_thread and on_rank_0:
        threads -= 1
        current_machine_cnt -= 1
        memory -= 4

    evenly_divided_omp = threads // current_machine_cnt
    remainder_omp = threads % current_machine_cnt
    if end_rank - rank < remainder_omp:
        omp_threads = evenly_divided_omp + 1
        marked_for_more_mem = True
    else:
        omp_threads = evenly_divided_omp
        marked_for_more_mem = False

    if rank == 0:
        machine.omp_threads = 1
        machine.memory_per_thread = 4
    else:
        machine.omp_threads = omp_threads
        machine.memory_per_thread = int(memory * (omp_threads / threads))
    if rank_0_one_thread and on_rank_0:
        threads += 1
        memory += 4
    comm.barrier()
    print(
        f"rank {rank} is using {machine.name} with {machine.omp_threads} / {threads} and {machine.memory_per_thread}/{machine.memory} GB"
    )
    return machine
