## Instructions to run the code

1. Install dependencies `pip install -r requirements.txt`
2. To crop individual symbols from the images in the Roboflow dataset, run `crop.py`. An example is shown below. See also the notebook `run_crop.ipynb`.

```bash
python crop.py --json_path="data/original_data/train/_annotations.coco.json" --images_dir="data/original_data/train" --out_dir="data/train"
```

## What the notebooks contain

`run_train` is an example of running the `train.py` script.

`results` plots training loss and t-SNE plots for different hyperparameter configurations.

`inference` contains the function `retrieve_top_k`, which, given a model and query image, plots the k images having embeddings closest to the query embedding.
