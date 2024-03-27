import logging
import os
from typing import Dict

import requests
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

logger = logging.getLogger(__name__)


class TaskApi:
    def __init__(self, task):
        self._base_url = "https://tasks.aidevs.pl"
        self.api_key = os.getenv("TASK_API_KEY")
        self._token = None
        self.auth(task)

    def auth(self, task):
        url = f"{self._base_url}/token/{task}"
        response = requests.post(url, json={"apikey": self.api_key})
        logger.info(response.json())
        if response.status_code == 200 and response.json()["code"] == 0:
            self._token = response.json()["token"]
        else:
            logger.error(response.json())
            raise Exception("Cannot get token")

    def task(self) -> Dict:
        if self._token is None:
            logger.error("Do auth first")
        url = f"{self._base_url}/task/{self._token}"
        response = requests.get(url)
        logger.info(response.json())
        return response.json()

    def task_question(self, question) -> Dict:
        if self._token is None:
            logger.error("Do auth first")
        url = f"{self._base_url}/task/{self._token}"
        response = requests.post(url, data=dict(question=question))
        logger.info(response.json())
        return response.json()

    def answer(self, answer) -> bool:
        if self._token is None:
            logger.error("Do auth first")
        url = f"{self._base_url}/answer/{self._token}"
        response = requests.post(url, json={"answer": answer})
        logger.info(response.json())
        if (
            response.status_code == 200
            and response.json()["code"] == 0
            and response.json()["note"] == "CORRECT"
        ):
            return True
        else:
            return False
