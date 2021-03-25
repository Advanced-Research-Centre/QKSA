# AutomatedQuantumPhysicist

Artificial-Life + General-Reinforcement-Learning + Digital-Physics

### Suggestive titles
* ALife path to AGI, by combining MC-AIXI-tl with Universal Constructor in Python
* Automated Design and Inference of Quantum Foundations Experiments on the Circuit Model using a Swarm of Evolving General Reinforcement Learning Agents
* Participatory Observer States to Quantum Physics using Evolving Universal Reinforcement Learning Self-Replicating Agents
* Evolving universal AGI agents for discovering physical laws
* Fittest of the fittest artificial physicists: using
* Agent Based Modelling of Foundational Experiment in Quantum Environments
* Automated (Gate-based) Quantum Experiments using Universal Evolving Constructors

### Version changelog
* v01: Minimal quine in Python from Wikipedia example.
* v02: Added an empty extra function as part of the quine.
* v03: Create quine as Python file based on current filename, instead of console print.
* v04: Seperate functions for reproduction (constructor) and brain (goedel machine). Hardcoded 3-bit prediction and reward based on Hamming distance with perception from console input. Reproduce when Hamming distance is 3 with current perception mutated as the new prediction.
* v05: Seperate class and code files for agent and quine. Agent class defines learning model variables and functions. Genes include prediction and reward thresholds for death and reproduction. Reproduce when Hamming distance within these thresholds. Die (halt) if distance more than death threshold. Quine prediction mutates as the current perception.
* v06: Agent testing code without quines. Agent function to calculate BDM and logical depth of perception history and predictions. Choose prediction based on lowest BDM.
* v07: OpenAI gym guessing game tested as an environment.
* v08: Build environment as a OpenQASM code of Bell state and return perception of neighbour qubits in Z-basis measurement.
* v09: Per qubit 3 axis measure action. Basis selection policy based on BDM of percept history of tomographic basis.

### Planned upgrades
* v10: Hidden and visible qubits. GHZ test with 2 visible qubits. ++++
* v11: Neighbour/visible qubits based on quantum interaction graph. ++
* v12: Quantum Kolmogorov Complexity as a metric for quantum agents with entanglement. +
* v13: Quantum Entanglement Entropy and Mutual Information as a function of number of visible qubits for quantum agents. +
* v14: Smart tomography based on TM Solomonoff induction instead of NN. +++
* v15: Integrate test environment with agent. ++++
* v16: Formulate reward and replication policy for tomography. Integrate agent with Goedel machine. +++
* v17: Hypervisor for automatic threaded execution of generated quines. ++
* v18: Quantum Cellular Automata Rule/Composibility Learning. +
* v19: Explainable Ansatz Learning for Variational Quantum Circuits. +
* v20: Integrate qThought and extended Wigner's friend experiments. +
* v21: Neuro-evolution and NTM as an alternative for AIXI. +
* v22: BraIIinSSss!
* v23: BraIIinSSss!
* v24: BraIIinSSss!
* v25: BraIIinSSss!

Note: order of ugrades subject to change (plus denote near-term focus).

### Important references
* [AIXIjs](https://www.aslanides.io/aixijs/)
* [Qiskit documentation](https://qiskit.org/documentation/)
* [Build An Optimal Scientist, Then Retire](https://hplusmagazine.com/2010/01/05/build-optimal-scientist-then-retire/)
* [The Online Algorithmic Complexity Calculator](http://complexitycalculator.com/)
* [Law without law: from observer states to physics via algorithmic information theory](https://quantum-journal.org/papers/q-2020-07-20-301/)
* [Estimating Algorithmic Information Using Quantum Computing for Genomics Applications](https://www.mdpi.com/2076-3417/11/6/2696)

### Cite this as
