# GoedelConstructor
ALife path to AGI, by combining MC-AIXI-tl with Universal Constructor in Python

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
* v10: [TBD] Agent neighbours based on quantum interaction graph.