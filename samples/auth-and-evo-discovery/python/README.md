## Getting started

Create a Python environment and install the package requirements, eg.

```pip install -r requirements.txt```

## Create an Evo access token

Evo access tokens can be created using:
- Native apps (auth code + PKCE flow)
- Service apps (client credentials flow)
- Web apps (authorization code flow)
- SPA apps (auth code + PKCE flow)

You must choose the app type that suits your environment.
For more information, consult the page **Apps & Tokens** on the Evo documentation website.

- Follow `native-app-token.ipynb` for native apps.
- Follow `service-app-token.ipynb` for service apps.
- Web apps and SPA apps work in a similar way to native apps.

## Perform Evo Discovery

After you have obtained an access token you must find your `organisation ID` and your assigned `Evo hub URL`.

Follow the notebook `evo-discovery.ipynb` to obtain these values.
