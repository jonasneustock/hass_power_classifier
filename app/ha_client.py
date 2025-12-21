import logging
import requests


class HAClient:
    def __init__(self, base_url, token, timeout=30):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._log_token()

    def _log_token(self):
        if not self.token:
            logging.warning("HA token is not set")
            return
        token_len = len(self.token)
        if token_len <= 8:
            masked = "*" * token_len
        else:
            masked = f"{self.token[:4]}{'*' * (token_len - 8)}{self.token[-4:]}"
        logging.info("HA token loaded: %s", masked)

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def get_state(self, entity_id):
        url = f"{self.base_url}/api/states/{entity_id}"
        response = requests.get(url, headers=self._headers(), timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def set_state(self, entity_id, state, attributes=None):
        url = f"{self.base_url}/api/states/{entity_id}"
        payload = {"state": state}
        if attributes:
            payload["attributes"] = attributes
        response = requests.post(
            url, headers=self._headers(), json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
