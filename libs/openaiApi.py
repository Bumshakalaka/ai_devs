import os

import requests
import logging

from dotenv import load_dotenv, find_dotenv
from requests.auth import AuthBase

logger = logging.getLogger(__name__)


class TokenAuth(AuthBase):
    """Implements a token authentication scheme."""

    def __init__(self, token):
        self.token = token

    def __call__(self, request):
        """Attach an API token to the Authorization header."""
        request.headers["Authorization"] = f"Bearer {self.token}"
        return request


class OpenApi:
    def __init__(self):
        self._base_url = "https://api.openai.com/v1"
        self.api_key = os.environ["OPENAI_API_KEY"]

    def moderations(self, input):
        url = f"{self._base_url}/moderations"
        response = requests.post(
            url, json=dict(input=input), auth=TokenAuth(self.api_key)
        )
        logger.info(response.json())
        return response.json()["results"][0]["flagged"]


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    loggerFormat = "%(asctime)s:\t%(message)s"
    loggerFormatter = logging.Formatter(loggerFormat)
    loggerLevel = logging.DEBUG
    logging.basicConfig(format=loggerFormat, level=loggerLevel)

    api = OpenApi()
    print(api.moderations("azjaci są fajni"))
