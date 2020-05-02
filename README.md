# Prerequisites

- Python 3.x
- Fire
- nltk

To install these, run:

```bash
pip install fire nltk
```

## NLTK Data

After installation, NLTK need to download some data to be useful. Invoke the python interpreter and run the following code block:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

# Downloading the dataset

To download the dataset used in this model, simply run:

```bash
python download_dataset.py
```

# Running the model

To run the model in evaluation mode, just run:

```bash
python main.py evaluate <options>
```

To run the model to classify texts, just run:

```bash
python main.py classify <text> <options>
```

## Model options

- `retrain` - forces the model to relearn all of the parameters even if the pickle file that contains the serialized parameters already exists. Default is `False`.
- `stopwords` - tells the model to use stopwords during training and prediction phase. Default is `True`.
- `lemmatize` - tells the model to use lemmatization during training and prediction phase. Default is `True`.