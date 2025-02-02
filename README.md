# Evo samples and data converters

## Overview

This repository provides sample code along with the source code for Evo-specific data converters.

When running a converter, data is imported from a supported file format, converted into geoscience objects, and then published to the Seequent Evo API.

The existing data converters can be used as is, or you can use them as a template for your own integration.

## Repository structure

The top level sections for the repository are as follows.

- [Samples](samples/README.md) - Jupyter notebooks and code samples for importing data with the existing data converters
- [Data Converters](src/evo/data_converters/README.md) - Source code for the data converters module (`evo.data_converters`)
- Scripts - helper scripts for working with different types of data files
- Tests - unit tests for the data converter module

### Evo authorisation and discovery

Whether using either the Jupyter samples or undertaking development work on the data converter module itself, integration with Evo will require that you are granted access as an Evo Partner or Customer, along with access to a specific Evo Workspace. Access is granted via a token. For more information on getting started, see the [Seequent Evo Developer Portal](https://developer.seequent.com/).

Refer to the [auth-and-evo-discovery](samples/auth-and-evo-discovery/python/README.md) documentation to learn how to create an Evo access token and how to perform an Evo Discovery request.

## Setting up environment

Whether using the code samples or working on development of the converters the following initial setup instructions apply.

### Requirements

* Python 3.10

Python 3.10 is explicitly required to maintain compatibility with upstream dependencies that are not compatible with earlier or later versions of Python - specifically `resqpy`.

### Setting up Python environment

Firstly create the Python environment.

Install `virtualenv` and initialize a virtual environment in the root directory.

You must ensure you use Python 3.10 as the interpreter.  You may need to install this specifically for your environment and refer to it when creating the virtualenv.

```shell
pip install virtualenv
virtualenv -p python3.10  example_virtual_env
```

Activate the virtual environment from the root directory

```shell
.\example_virtual_env\Scripts\activate
```

For Mac and Linux users, run `source example_virtual_env/bin/activate` instead.

### Evo Artifactory dependencies

The following Evo dependencies are not yet publicly available on PyPI and need to be installed from Seequent's Artifactory package repository.

* `evo-client-common`
* `evo-object-client`
* `seequent-geoscience-object-schemas`

The Artifactory index for these has been added to the `requirements.txt` file,  but to successfully install you must have valid `ARTIFACTORY_USER` and `ARTIFACTORY_TOKEN` variables set in your environment before running `pip install`.

To obtain these values you must already have a valid login for Artifactory provided to you by Seequent and follow these steps:

1. Log in to Artifactory https://seequent.jfrog.io/ui/login/
1. Navigate to the user profile page
1. Click Generate an Identity Token
1. Name the token (perhaps something like "development token") and press Next
1. Copy the token value (Don't worry if the token doesn't appear listed under the 'Identity Tokens' header, this is a bug which is being resolved - the token that was generated IS valid)
1. Add a user environment variable to your computer named `ARTIFACTORY_USER` with the value as your Bentley email address (i.e. first.last@bentley.com)
1. Add a user environment variable to your computer named `ARTIFACTORY_TOKEN` with the copied token as the value

### Install Python dependencies

Ensure your pip version is up-to-date for this version of Python.

```
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
```

Install the project dependencies

```
pip install -r requirements.txt
```

Note that additional dependencies may be required to run individual Jupyter notebooks.  Refer to the [samples documentation](samples/README.md) for more information.

## License

The code in this repository is released under the [MIT license](LICENSE).