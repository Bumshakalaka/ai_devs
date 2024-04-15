import logging
import os
import requests
from typing import Dict

from dotenv import load_dotenv, find_dotenv
from requests.auth import AuthBase

logger = logging.getLogger(__name__)


class TokenAuth(AuthBase):
    """Implements a token authentication scheme."""

    def __init__(self, token):
        self.token = token

    def __call__(self, request):
        """Attach an API token to the Authorization header."""
        request.headers["X-API-KEY"] = self.token
        return request


class RenderFormApi:
    def __init__(self):
        self._base_url = "https://api.renderform.io/api/v2"
        self.api_key = os.environ["RENDERFORM_API_KEY"]

    def render(self, template: str, data: Dict):
        url = f"{self._base_url}/render"
        response = requests.post(
            url, json=dict(template=template, data=data), auth=TokenAuth(self.api_key)
        )
        logger.info(response.json())
        return response.json()["href"]


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    loggerFormat = "%(asctime)s:\t%(message)s"
    loggerFormatter = logging.Formatter(loggerFormat)
    loggerLevel = logging.DEBUG
    logging.basicConfig(format=loggerFormat, level=loggerLevel)

    api = RenderFormApi()
    data = {
        "TEXT.text": "Gdy koledzy z pracy mówią, że ta cała automatyzacja to tylko chwilowa moda, a Ty właśnie zastąpiłeś ich jednym, prostym skryptem",
        "PIC.src": "https://tasks.aidevs.pl/data/monkey.png",
    }
    print(api.render("sedate-wasps-dive-warmly-1825", data))
