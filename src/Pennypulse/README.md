# Pennypulse  

This repository is designed to support the development of my Master’s Thesis *Hardware-Adapted Quantum Machine Learning*. It includes modifications and additional functionalities for the PennyLane library.  

## Purpose  

The main objectives of this repository are:  
- **Customizing PennyLane**: Modify and extend specific features of the library to align with the requirements of my research.  
- **Centralized Access**: Consolidate all custom functionalities into a single location for streamlined imports.  
- **Ease of Installation**: Enable the use of this package without relying on the correct working directory by making it installable.  

## Project Structure  

The project is organized into three main folders:  

1. **`src/shape`**: Contains various pulse shapes used for different quantum operations.  
2. **`src/utils`**: Includes utility functions that assist in the implementation and management of quantum operations.  
3. **`src/`**:  
   - **`hamiltonian.py`**: Defines the Hamiltonian of the transmon base and transmon interaction, modified from the original PennyLane source code.  
   - **`observables.py`**: Integrates the modifications made to the Hamiltonian, providing necessary tools for observable measurements.  
   - **`pulses.py`**: Implements the Trotterized evolution of quantum states under the influence of a driving pulse.  
   - **`trotterization.py`**: Applies Trotterization to evolve the quantum state, using a different approach to the previous implementation in `pulses.py`.  

4. **`tests/`**: Contains test scripts to validate the functionalities and modifications in the repository.  

## Usage  

To install and integrate the custom functionalities into your projects, run:  
```bash  
pip install .  
