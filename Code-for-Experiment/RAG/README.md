## Note on Storing Model Weights and Large Files

Some of the model weights and large files in this project are not stored directly in the GitHub repository. Instead, they are hosted on Hugging Face Hub. To use the relevant features, please download the necessary files from the following address:

- Hugging Face models directory: [https://huggingface.co/karl-wang/ama-prof-divi/tree/main/models](https://huggingface.co/karl-wang/ama-prof-divi/tree/main/models)

**After downloading, please place the weight files into the corresponding local directory (e.g., `Code-for-Experiment/RAG/Checkpoint-for-CLAP-Long_MiniBatch_Training/`).**

https://huggingface.co/karl-wang/ama-prof-divi/tree/main/models/checkpoint-15000

https://huggingface.co/karl-wang/ama-prof-divi/tree/main/models/checkpoint-23000

https://huggingface.co/karl-wang/ama-prof-divi/tree/main/models/checkpoint-500



> Note: Due to the limited free quota for GitHub LFS, no large files have been uploaded to this repository. For automatic downloads, you can refer to the Hugging Face Hub [official documentation](https://huggingface.co/docs/hub/how-to-download-files) or use the `huggingface_hub` Python package.

---

## Automatic Download for Required Large Files (Optional)

Alternatively, you can add the following logic to your code for automatic downloads:

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="karl-wang/ama-prof-divi",
    filename="models/xxx.pt",
    local_dir="./Code-for-Experiment/RAG/Checkpoint-for-CLAP-Long_MiniBatch_Training/"
)
