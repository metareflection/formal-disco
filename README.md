# Formal Disco: A framework for scalable, distributed discovery systems.


Basic idea:
- We have an 'agenda' consisting of tasks and objects
- Workers of different types can manipulate tasks and objects in the agenda
- A "scheduler" runs workers continuously.

We use [Hydra](https://hydra.cc/) for composable configuration.

## Running an example

### 1. Launch a simple scheduler

Schedulers run forever, but we can limit them with a timeout.

Example:

```bash
$ timeout 0.5 python scheduler.py +agenda=local +scheduler=example
```

At first, there is no checkpoint, so this prints:


```text
No checkpoint; starting empty agenda.
Checkpointed agenda to local-agenda.pkl.
Worker DummyIdeaGenerator finished a turn.
Checkpointed agenda to local-agenda.pkl.
...
```

On subsequent runs, the agenda resumes from this checkpoint:

```text
Checkpointed agenda to local-agenda.pkl.
Worker DummyIdeaGenerator finished a turn.
Worker DummyImplementer finished a turn.
Checkpointed agenda to local-agenda.pkl.
...
```

### 2. Materialize objects

At any point you can dump the agenda's objects into a real directory:

```sh
$ du -h local-agenda.pkl
56K    local-agenda.pkl
$ python materialize.py local-agenda.pkl discoveries
Wrote discoveries/idea/932.txt
Wrote discoveries/idea/207.txt
Wrote discoveries/idea/977.txt
...
```

### 3. Explore results

```sh
$ ls discoveries/
idea/  programs/

$ ls discoveries/idea
106.txt 207.txt 293.txt 932.txt ...

$ cat discoveries/idea/443.txt
Write a method named m443

$ cat discoveries/programs/443.dfy
method m443() {
   print "hello from idea 443\n";
}
```

# To-Dos:

- [ ] Implement distributed agenda
- [ ] Implement actual workers
- [ ] Profit
