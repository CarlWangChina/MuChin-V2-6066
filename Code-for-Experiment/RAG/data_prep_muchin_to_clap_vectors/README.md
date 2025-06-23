# Pipeline for processing the MuChin dataset to support Amateur-Professional Semantic Divide research. It generates 10 types of descriptive texts for each song and converts them into a CLAP-based vector dictionary for retrieval.

## Processing Pipeline

(1) Process the amateur and professional description JSON files (`amat_desc_json` and `prof_desc_json`) using the current Chinese test set processing code, generating short, medium, and long formats, plus all label value combinations and all answer sentence combinations - 2 formats, resulting in (3+2)*2=10 types of Chinese texts.

(2) Process English texts using CLAP to generate vectors. Chinese texts will be processed later when Chinese CLAP is available.

This way, each song corresponds to 10 different CLAP vectors, stored in a dictionary with keys named clap0, clap1...clap9, containing CLAP vector tensors.

After processing vectors for 7k songs, write code to randomly select one of the CLAP vectors when processing each song.

Finally, encapsulate the text processing + CLAP vector conversion + dictionary storage as a single function, so that subsequent Chinese data can also be processed into vectors and dictionaries for training.

English processing into CLAP vectors will be provided to model training.

Chinese doesn't have ready-made CLAP available, so we'll process it when the new CLAP is ready.

This pipeline is part of the MuChin RAG system designed to bridge the amateur-professional semantic gap in music understanding.