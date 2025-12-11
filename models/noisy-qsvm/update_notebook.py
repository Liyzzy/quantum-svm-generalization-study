"""
Script to update the noise model in noisy_qsvm_spambase.ipynb 
to be consistent with em-lungcancer-qsvm.ipynb

Changes:
1. Import ReadoutError
2. Update noise parameters to realistic values:
   - p_gate_1q = 0.001 (0.1% for 1q gates)
   - p_gate_2q = 0.01 (1.0% for 2q gates)  
   - p_readout = 0.02 (2.0% readout error)
3. Add ReadoutError to noise model
4. Update default_shots from 256 to 8192
"""

import json

# Load the notebook
notebook_path = r'noisy_qsvm_spambase.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update cell 4 (index 3) - Qiskit Imports - add ReadoutError import
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        # Fix imports - add ReadoutError
        if 'from qiskit_aer.noise import NoiseModel, depolarizing_error' in source and 'ReadoutError' not in source:
            cell['source'] = [
                "# --- Qiskit Imports ---\n",
                "from qiskit.circuit.library import ZZFeatureMap\n",
                "from qiskit_aer import AerSimulator\n",
                "from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError\n",
                "from qiskit_aer.primitives import SamplerV2 as AerSampler\n",
                "from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager\n",
                "from qiskit_machine_learning.state_fidelities import ComputeUncompute\n",
                "from qiskit_machine_learning.kernels import FidelityQuantumKernel\n",
                "from qiskit_machine_learning.algorithms import QSVC, PegasosQSVC"
            ]
            print("Updated: Added ReadoutError import")
        
        # Fix noise model cell
        if 'p_error = 0.04' in source:
            cell['source'] = [
                "# Quantum Kernel Implementation with Noise\n",
                "# Noise Model implementation (Depolarizing error and Readout Error)\n",
                "# Error rates (realistic, not too high)\n",
                "p_gate_1q = 0.001   # 0.1% error for single-qubit gates (u1, u2, u3)\n",
                "p_gate_2q = 0.01    # 1.0% error for two-qubit gates (cx)\n",
                "p_readout = 0.02    # 2.0% chance of wrong measurement\n",
                "\n",
                "noise_model = NoiseModel()\n",
                "noise_model.add_all_qubit_quantum_error(depolarizing_error(p_gate_1q, 1), ['u1', 'u2', 'u3'])\n",
                "noise_model.add_all_qubit_quantum_error(depolarizing_error(p_gate_2q, 2), ['cx'])\n",
                "\n",
                "readout_error = ReadoutError([[1 - p_readout, p_readout], [p_readout, 1 - p_readout]])\n",
                "noise_model.add_all_qubit_readout_error(readout_error)\n",
                "\n",
                "print(f\"Depolarizing noise: {p_gate_1q*100}% (1q), {p_gate_2q*100}% (2q), Readout: {p_readout*100}%\\n\")"
            ]
            # Clear old output
            cell['outputs'] = []
            print("Updated: Noise model parameters")
        
        # Fix sampler shots
        if 'default_shots = 256' in source:
            cell['source'] = [
                "# Noisy Sampler with high shots (8192 for better statistics)\n",
                "noise_sampler = AerSampler.from_backend(\n",
                "    backend = noisy_backend,\n",
                "    default_shots = 8192\n",
                ")\n",
                "print(\"Noisy Sampler created !\")"
            ]
            # Clear old output
            cell['outputs'] = []
            print("Updated: default_shots to 8192")

# Save the updated notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("\nNotebook updated successfully!")
print("Noise model is now consistent with em-lungcancer-qsvm.ipynb")
