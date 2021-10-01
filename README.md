# Quantum Knowledge Seeking Agent

Latest pre-release : v09 [![DOI](https://zenodo.org/badge/342195575.svg)](https://zenodo.org/badge/latestdoi/342195575)

Recent pre-print with motivation, core thesis and baseline framework: [here](<./21-07-05 - QKSA.pdf>), [arXiv](https://arxiv.org/abs/2107.01429)

[![Project pitch](https://img.youtube.com/vi/RPHbsUFjZcI/0.jpg)](https://www.youtube.com/watch?v=RPHbsUFjZcI)

QKSA extends the universal reinforcement learning (URL) agent models of artificial general intelligence to quantum environments.
The utility function of a classical exploratory Knowledge Seeking Agent, a generalization of AIXI, is diversified to a pool of distance measures from quantum information theory.
Quantum process tomography (QPT) algorithms form the agent policy for modeling classical and quantum environmental dynamics.
The optimal QPT policy is selected based on a mutable cost function based on both algorithmic complexity as well as computational resource complexity.
Instead of Turing machines, we estimate the cost metrics on high-level language to ease experimentation.
The entire agent design is encapsulated in a self-replicating quine which mutates the cost function based on the predictive value of the optimal policy choosing scheme. 
Thus, multiple agents with pareto-optimal QPT policies can evolve using genetic programming, mimicking the development of physical theories each with different resource cost.

Despite its importance, few quantum reinforcement learning exists in contrast to the current thrust in quantum machine learning.
QKSA is the first proposal for a framework that resembles the classical URL models.
It can be applied for simulating and studying aspects of quantum information theory like control automation for quantum computing, multiple observer paradoxes, course-graining, distance measures, resource complexity trade-offs, quantum game theory, etc.
Similar to how AIXI-tl is a resource-bounded active version of Solomonoff universal induction, QKSA is a resource-bounded participatory observer framework to the recently proposed algorithmic information based reconstruction of quantum mechanics.

### How to run
```
python QKSA.py
```

### Features

Docs website : https://advanced-research-centre.github.io/QKSA/ (under construction)

#### Version changelog
listed [here](https://github.com/Advanced-Research-Centre/QKSA/blob/main/legacy_vers/changelog.md)

#### In progress
* v13: LEAST metric on QPT and smart QPT based on action-perception history.

#### Planned upgrades
* Hidden and visible qubits. GHZ test with 2 visible qubits. ++++
* Neighbour/visible qubits based on quantum interaction graph. ++
* Quantum Kolmogorov Complexity as a metric for quantum agents with entanglement. +
* Quantum Entanglement Entropy and Mutual Information as a function of number of visible qubits for quantum agents. +
* Formulate reward and replication policy for tomography. Integrate agent with Goedel machine. +++
* Hypervisor for automatic threaded execution of generated quines. ++
* Quantum Cellular Automata Rule/Composibility Learning. +
* Explainable Ansatz Learning for Variational Quantum Circuits. +
* Integrate qThought and extended Wigner's friend experiments. +
* Neuro-evolution and NTM as an alternative for AIXI. +
* BraIIinSSss!
* mOoorEEe BraIIinSSss!

Note: order of ugrades subject to change (plus denote near-term focus).

### Contributors

The QKSA project was started as part of the PhD research of [Aritra Sarkar](https://qutech.nl/person/aritra-sarkar/) at Delft University of Technology.
It is part of the research within the [Department of Quantum & Computer Engineering](https://www.tudelft.nl/en/eemcs/the-faculty/departments/quantum-computer-engineering).

Currently various aspects of this project are being pursued as a research project under [QWorld](https://qworld.net/).

The details of the contributions from collaborators are available [here](https://github.com/Advanced-Research-Centre/QKSA/blob/main/contributors.md).

### Important references
* [AIXIjs](https://www.aslanides.io/aixijs/)
* [Qiskit documentation](https://qiskit.org/documentation/)
* [Build An Optimal Scientist, Then Retire](https://hplusmagazine.com/2010/01/05/build-optimal-scientist-then-retire/)
* [The Online Algorithmic Complexity Calculator](http://complexitycalculator.com/)
* [Law without law: from observer states to physics via algorithmic information theory](https://quantum-journal.org/papers/q-2020-07-20-301/)
* [Estimating Algorithmic Information Using Quantum Computing for Genomics Applications](https://www.mdpi.com/2076-3417/11/6/2696)
