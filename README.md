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

# To run with LLMImplementer and LLMFixer

Download the dataset from this [Link](https://drive.google.com/file/d/1CYAxgFezCMd6E6LdHgnt6rMYYPkKI1fy/view?usp=sharing) and save it to `data/gh-readmes-100k.jsonl`.

Then, you can run:

```sh
$ python scheduler.py +scheduler=readme_ideas scheduler.agenda=local
```

This will by default log to wandb, which you can use to follow the run.

# To-Dos:

- [ ] Implement distributed agenda
- [X] Implement initial API-based workers
- [ ] Bug: Make workers (implementer, fixer) read a batch of tasks at once to avoid taking the same task twice in one round
- [ ] Design edit/extend actions (we have an Implementer but not an "Editor" yet to implement "extend" tasks)
- [ ] Implement local LLM workers
- [ ] Make local LLM workers self-improving
- [ ] Profit
