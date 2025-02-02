## Evo samples

The samples are structured by feature, then by language. For example, the `data-converters/python` folder contains Python
examples for using the Data Converters library.

### Seequent Evo APIs

Once you have obtained an Evo access token and found your organisation ID and Evo hub URL, you can start to explore the API samples in:

- `auth-and-evo-discovery` - code samples relating to authentication and service discovery.
- `data-converters` - code samples demonstrating how to use the Evo Data Converters library.

## Python samples

The following instructions relate to the Python samples code found in this directory.

The Python samples are provided as Jupyter notebooks. 

In order to run the Jupyter notebook the Evo samples virtual env can be used as per the [setup instructions](../README.md#setting-up-environment) with some additions to run the sample code.

### Installing additional packages

Different samples may require additional packages to be installed.   Check for a requirements.txt file in the sample sub directory and if required run `pip install` on it.  e.g. for the data converters samples:

```
pip install -r samples/data-converters/python/requirements.txt
```

### Setting up and running Jupyter notebooks

Notebooks can be run in your tool of choice (e.g. VS Code). To use Jupyter (the default):

```
pip install jupyter
```

Then in the directory of the notebook(s) you want to run, type:

```
jupyter notebook
```

It should open a browser where you can open the notebooks for the current directory.