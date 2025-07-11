# How the amateur-professional divide in musical language biases generative AI

This repository contains the dataset, code, and resources for the paper "How the amateur-professional divide in musical language biases generative AI". Our research leverages generative AI as a novel computational lens to provide a large-scale, quantitative analysis of the cognitive and linguistic divide between experts (professionals) and novices (amateurs) in the domain of music.

The empirical foundation of this research is the **MuChin Dataset**, a large-scale corpus specifically constructed to capture and quantify this descriptive divide.

---

### Key Features & Contributions

-   **A Novel Dataset for Cognitive Science**: The MuChin dataset, featuring paired descriptions from both amateur and professional perspectives for thousands of musical pieces.

-   **Computational Analysis of the Divide**: Methodologies and code to quantitatively analyze the linguistic and semantic differences between user groups.

-   **Probing AI with Human Language**: A framework to test how generative models respond to different human linguistic patterns.

-   **AI-driven Interventions**: Code for exploring interventions (e.g., RAG, targeted training) designed to bridge the human-AI communication gap.

---

### Repository Content Overview

This repository maintains a comprehensive structure containing all related scripts and modules used throughout our research. The main components are:

-   **/Datasets-for-MuChin-V2/**: Contains data samples, metadata, and detailed documentation about the MuChin dataset. The full audio and database files are hosted externally.

-   **/Code-for-Experiment/**: The central hub for our experimental code. It is organized into the following key areas:
    -   **/Metrics/**: Implementations for evaluating model outputs. This includes our primary **Semantic Similarity** analysis tools used to quantify the descriptive divide, as well as metrics for assessing the downstream impact on AI behaviour, such as **Intent Alignment** (SAA, ARA).
    -   **/RAG/**: Our implementation of a Retrieval-Augmented Generation system, focusing on a CLAP-based text-to-audio retrieval pipeline to enhance prompt understanding. Key components include data preparation, vector search frontend, and audio processing utilities.
    -   **/Targeted-Training/**: Contains various scripts and configurations for model training. This includes reference implementations for different architectures and tools like **NVIDIA Apex** for high-performance mixed-precision training.

-   **/Code-for-MuChin-AP/**: The complete source code for our custom-built data annotation platform, **MuChin-AP**.

-   **/user_manual_munchin_ap/**: Documentation, guidelines, and internal planning documents related to the annotation platform and process.

---

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/CarlWangChina/MuChin-V2-6066.git](https://github.com/CarlWangChina/MuChin-V2-6066.git)
    cd MuChin-V2-6066
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install general dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install dependencies for specific modules:**
    -   **Semantic Analysis**: To use the semantic similarity tools, install the specific libraries required by the `bge-large-zh-v1.5` model:
        ```bash
        pip install -U FlagEmbedding
        pip install transformers==4.34.0
        ```
    -   **NVIDIA Apex (for advanced training)**: For performance and full functionality, we recommend installing Apex with CUDA and C++ extensions:
        ```bash
        # Navigate to the apex directory
        cd Code-for-Experiment/Metrics/music_understanding_model/apex
        pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
        cd ../../../../../.. # Return to root
        ```

---

### Core Modules & Experiments Guide

This section provides a more detailed look into the key components of our research and how to use them.

#### 1. Semantic Similarity Analysis

This module assesses the semantic similarity between two sets of text labels (e.g., amateur vs. professional).

-   **Download the Embedding Model**:
    This code uses `bge-large-zh-v1.5` for analyzing Chinese text.
    ```bash
    cd ./Code-for-Experiment/Semantic-Analysis/
    mkdir -p models
    cd models
    git clone [https://huggingface.co/BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5)
    cd ../../../.. # Return to root
    ```

-   **Usage in Python**:
    ```python
    # Ensure your script is run from the project root
    from Code-for-Experiment.Semantic-Analysis.semantic_similarity import MuChindata_Analyzer
    
    analyzer = MuChindata_Analyzer(
        'path/to/your/labels-free.xlsx', 
        './Code-for-Experiment/Semantic-Analysis/models/bge-large-zh-v1.5'
    )

    list_1 = [["Nostalgia", "Pop Songs"], ["Slow", "Medium"]]
    list_2 = [["Nostalgia", "Style and Genre"], ["Slow", "Slower"]]
    
    similarity_score = analyzer.cal_word_similarity(list_1, list_2)
    print(f"Semantic Similarity Score: {similarity_score}")
    ```

#### 2. Structured Lyric Generation & Evaluation

This module uses LLMs to generate structured lyrics based on a theme and a structural template, then evaluates the output based on structure, alignment, and rhyme.

-   **One-Shot Prompting**: The core of the generation is a detailed one-shot prompt that instructs the LLM on how to format the output.
-   **Objective Evaluation**: The quality of the generated lyrics is assessed using `glrc_obj_eval.py`. The evaluation considers multiple dimensions:
    -   **Overall Performance**: Gestalt pattern matching between the required and generated structure.
    -   **Section Structure**: Correctness of section names, order, and line counts.
    -   **Word Count Alignment**: How closely the word count of each line matches the template.
    -   **Rhyme**: Adherence to end-of-line rhyme requirements (`R` markers), based on the "中华新韵" rhyme scheme.

#### 3. RAG and CLAP-based Retrieval

Our Retrieval-Augmented Generation system uses CLAP to bridge the semantic gap.

-   **Data Pipeline**: The pipeline in `Code-for-Experiment/RAG/data_prep_muchin_to_clap_vectors/` processes amateur and professional descriptions into 10 different text formats per song, which are then converted into CLAP-based vector embeddings for retrieval.
-   **Vector Search Frontend**: The code in `Code-for-Experiment/RAG/clap_retrieval_system/` and `rag_search_frontend/` provides a user interface for the retrieval system. To run it:
    ```bash
    # First, create a user
    python Code-for-Experiment/RAG/clap_retrieval_system/adduser.py <username> <password>
    
    # Then, launch the server
    python Code-for-Experiment/RAG/clap_retrieval_system/server.py
    ```

#### 4. Targeted Training & Model Architectures

The repository contains code for training and evaluating various models. A key utility used is **NVIDIA Apex** for high-performance mixed-precision training.

-   **Example Training Command (Distributed Training with Apex)**:
    A typical command for launching a distributed training job (e.g., for ImageNet, adapted for our models) looks like this:
    ```bash
    python -m torch.distributed.launch --nproc_per_node=2 main_amp.py \
           -a resnet50 --b 224 --workers 4 --opt-level O1 ./
    ```
    This demonstrates the use of `torch.distributed.launch` and Apex's automatic mixed precision (`--opt-level O1`).

---

### Dataset & Available Resources

#### MuChin Dataset Releases

The dataset is available in two main versions. **Note:** Some data files contain Chinese characters. Please ensure you open them with UTF-8 encoding to prevent garbled text.

-   **MuChin 1000**
    This initial set contains 1,000 audio tracks with detailed text annotations.
    -   **Hugging Face:** `huggingface.co/datasets/karl-wang/MuChin1k`
    -   **Baidu Netdisk:** `pan.baidu.com/s/1D4xGQhYUwWbpaHyAS71dfw` (Password: `1234`)

-   **MuChin 6066 (Full Dataset)**
    The second, larger release contains 6,066 unique songs and all associated annotations.
    -   **Audio + Annotations (`.tar.bz2`)**:
        -   **Hugging Face:** `huggingface.co/datasets/karl-wang/MuChin-v2-6066`
        -   **Baidu Netdisk:** `pan.baidu.com/s/1GP5p3Ip_j0MraAcA52gIuQ` (Password: `68c1`)
    -   **Database File (Metadata & Annotations, No Audio)**:
        -   **Hugging Face:** `huggingface.co/datasets/karl-wang/MuChin-v2-6066`

#### Important Usage Notes

1.  **Timestamp Discrepancy**: The primary focus of manual annotation was on descriptive and structural content, not timestamp correction. For the most accurate available timestamps, please use the `raw_lyric` files. The timestamps in `tknz_json` files are incorrect due to manual merging and splitting of lyric lines during annotation.

2.  **Duplicate Annotations**: Due to an operational error, 724 songs from the initial `MuChin 1000` release were annotated a second time. We recommend using the annotation results from the `MuChin 1000` directory for these specific songs. A list of these duplicates can be found in `muchin_5790_1000_overlap.jsonl`.

3.  **File Compression**: The dataset is packaged as a `.tar.bz2` archive to prevent extraction errors associated with large `.zip` files on some systems.

#### Component Models

To ensure the full reproducibility of our findings, we provide the fine-tuned model weights used in our intervention experiments (e.g., for the RAG system). These are available at our Hugging Face Model repository:
`huggingface.co/karl-wang/ama-prof-divi`

---

### License

This project is licensed under the MIT License. See the `LICENSE` file for details.
