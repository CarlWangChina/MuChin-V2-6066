# A service designed for batch processing of large-scale audio datasets, using CLAP for embedding and Whisper for ASR to support the Amateur-Professional Semantic Divide research.

## Startup Instructions

```bash
cd asr_server
ray start --head
https_proxy=127.0.0.1:7890 serve deploy config.yaml
```

## Shutdown Instructions

```bash
ray stop
```

This service is part of the MuChin RAG system for processing large-scale audio datasets in the context of amateur-professional semantic analysis.
