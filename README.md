## Instructions to run the code

1. Run `pip install -r requirements.txt`

2. Data should be organized like this

```
data/
├── train/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── valid/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── test/
    ├── image1.png
    ├── image2.png
    └── ...
```

## What the notebooks contain

`run_train` is an example of running the `train.py` script.

`results` plots training loss and t-SNE plots for different hyperparameter configurations.

`inference` contains the function `retrieve_top_k`, which, given a model and query image, plots the k images having embeddings closest to the query embedding.
