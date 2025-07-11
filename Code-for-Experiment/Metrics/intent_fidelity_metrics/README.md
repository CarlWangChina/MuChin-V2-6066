This directory contains the code for computing Intent Fidelity metrics, specifically Semantic Audio Alignment (SAA) and Acoustic Reference Alignment (ARA), as detailed in our research.  

**Model Weights Location**:  
The required weights are stored in the "Checkpoints-for-SAA-ARA" folder of the Hugging Face repository:  
https://huggingface.co/karl-wang/ama-prof-divi/tree/main/models  

Key file to locate: `630k-audioset-best.pt` (1.86GB).  

**Setup Instructions**:  
1. Download the weights from the link above.  
2. Place the files in the directory:  
   `GitHub/MuChin-V2-6066/Code-for-Experiment/Metrics/intent_fidelity_metrics/checkpoints-for-SAA-ARA`  
3. Once positioned, the code can be executed to compute the SAA and ARA metrics.