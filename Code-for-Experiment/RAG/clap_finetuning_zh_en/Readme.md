Scripts and resources for fine-tuning the CLAP model on a Chinese-to-English description task, starting from pre-trained open-source weights.

 Description Processing and CLAP Vectorization Pipeline
This document outlines the pipeline for processing the amateur (amat_desc_json) and professional (prof_desc_json) descriptions from the MuChin dataset to generate CLAP vector embeddings for model training.

Pipeline Overview

The core objective is to generate 10 distinct text-based representations for each song from both amateur and professional descriptions, and then convert these text representations into CLAP vector embeddings. This multi-representation approach aims to provide a rich and varied input for training models.

The entire process is encapsulated into a reusable module for future use, especially for when a Chinese-language CLAP model becomes available.

Stage 1: Text Variant Generation

For each song, we process both the amateur and professional description files to generate 10 different text variants.
Process Amateur and Professional Descriptions (amat_desc_json, prof_desc_json):
For each of the two description types, create the following five text formats:
Short Form: A concise summary or key-phrase extraction.
Medium Form: A standard-length description.
Long Form: A detailed, elaborate description.
Combined Labels: A single string containing all annotated label values concatenated together.
Combined Sentences: A single string containing all descriptive sentences concatenated together.
This process results in (3 + 2) * 2 = 10 distinct text files for each song.

Stage 2: CLAP Vectorization

Convert English Text to Vectors:
The English text variants generated in the previous stage are processed using a pre-trained CLAP model to generate corresponding vector embeddings.
Future Work: Chinese Text:
The Chinese text variants are preprocessed and saved, but vectorization is pending the availability of a suitable, high-performance Chinese-language CLAP model. The pipeline is designed to easily incorporate this step in the future.

Stage 3: Data Storage

Store Vectors in a Dictionary:
For each song, the 10 generated CLAP vector tensors are stored in a dictionary.
The keys for this dictionary are clap0, clap1, clap2, ..., clap9.
This dictionary structure allows for easy access and organization of the multiple vector representations for each song.

Stage 4: Usage for Model Training

Randomized Vector Selection:
During the model training process, for each song in a batch, one of the 10 available CLAP vectors is to be randomly selected as the input condition.
This strategy introduces significant data augmentation, exposing the model to a wide variety of descriptive nuances for the same musical piece, which is intended to improve robustness and generalization.