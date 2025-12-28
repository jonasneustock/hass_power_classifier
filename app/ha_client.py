import logging
import asyncio

import httpx
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

    async def async_get_states(self, entity_ids):
        results = {}

        async def fetch(client, entity_id):
            try:
                resp = await client.get(f"/api/states/{entity_id}")
                resp.raise_for_status()
                results[entity_id] = resp.json()
            except Exception as exc:
                logging.warning("Async read failed for %s: %s", entity_id, exc)

        async with httpx.AsyncClient(
            base_url=self.base_url, headers=self._headers(), timeout=self.timeout
        ) as client:
            tasks = [fetch(client, eid) for eid in entity_ids]
            if tasks:
                await asyncio.gather(*tasks)
        return results

    def get_states_parallel(self, entity_ids):
        if not entity_ids:
            return {}
        try:
            return asyncio.run(self.async_get_states(entity_ids))
        except RuntimeError:
            # if already in an event loop, fallback to serial
            return {eid: self.get_state(eid) for eid in entity_ids}

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
