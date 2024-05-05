## It's Never Too Late: Fusing Acoustic Information into Large Language Models for Automatic Speech Recognition

[[Paper]](https://openreview.net/pdf?id=QqjFHyQwtF)


## Conda Environment Configuration

Our code is built based on [lit-llama](https://github.com/Lightning-AI/lit-llama), please refer to official tutorial to build a conda environment. Then, please install the required packages using following command:
```bash
pip install -r requirements.txt
```

## Code
The core implement is in generate_uadf.py

Before UADF fusion, you need: 
1) Train your own ASR model that has same tokenizer with LLM.
2) Train a LLaMA-based GER (generative error correction) model, please refer to [RobustGER](https://github.com/YUCHEN005/RobustGER) 
                       
With advancement of large speech langauge models, (1) and (2) can be implemented in a unified model by two separated lora-adapters, and then directly perform fusion during decoding.  

## Dataset
The dataset with N-best list can be found at [HuggingFace](https://huggingface.co/PeacefulData/HyPoradise-v0).

## References
```bib
@inproceedings{chen2024uadf,
  title={It's Never Too Late: Fusing Acoustic Information into Large Language Models for Automatic Speech Recognition},
  author={Chen, Chen and Li, Ruizhe and Hu, Yuchen and Siniscalchi, Sabato Marco and Chen, Pin-Yu and Chng, Eng Siong and Yang, Chao-Han Huck},
  booktitle={International Conference on Learning Representations},
  year={2024}
}

@inproceedings{chen2023hyporadise,
  title={HyPoradise: An Open Baseline for Generative Speech Recognition with Large Language Models},
  author={Chen, Chen and Hu, Yuchen and Yang, Chao-Han Huck and Siniscalchi, Sabato Marco and Chen, Pin-Yu and Chng, Eng Siong},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```
