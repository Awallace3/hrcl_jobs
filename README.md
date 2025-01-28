# Hierarchical Python Jobs (hrcl_jobs)
This package provides a simple way to run hierarchical jobs in Python.
Leveraging a SQL or PostgreSQL database, it allows users to create MPI jobs
that distribute tasks from a main thread to multiple worker threads for
high-throughput computation. 

# Installation
```bash 
pip install hrcl-jobs
```
**NOTE** if you are using macos, you might need to do the following to install with pip
```bash 
brew install mpich
sudo find / -name mpicc
```
And then run the pip install with the path to your mpicc
```bash
env MPICC=/yourpath/mpicc pip3 install hrcl-jobs
```

# Example Usage
