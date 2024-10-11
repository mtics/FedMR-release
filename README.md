# FedMR

> This project is the code and the supplementary of "**Personalized Item Embeddings in Federated Multimodal Recommendation**"

## Requirements

1. The code is implemented with `Python ~= 3.9` and `torch~=2.4.0+cu118`;
2. Other requirements can be installed by `pip install -r requirements.txt`.

## Quick Start

1. Put datasets into the path `./datasets/`;

2. For quick start, please run:
    ``````
    python main.py --model MMFedAvg --dataset KU
    ``````

3. if you want to use the notice function `mail_notice`, please set your own keys.

## Thanks

In the implementation of this project, we drew upon the following resources: [MMRec](https://github.com/enoche/MMRec), [RecBole](https://github.com/RUCAIBox/RecBole) and [Tenrec](https://github.com/yuangh-x/2022-NIPS-Tenrec?tab=readme-ov-file). 
We sincerely appreciate their open-source contributions!

## Contact

- This project is free for academic usage. You can run it at your own risk.
