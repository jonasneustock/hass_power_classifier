import logging
from datetime import datetime
from pathlib import Path

from fastapi.templating import Jinja2Templates

from app.classifier import ClassifierService, RegressionService
from app.config import load_config
from app.data_store import DataStore
from app.ha_client import HAClient
from app.logging_utils import log_event
from app.mqtt_client import MqttPublisher
from app.poller import PowerPoller
from app.training import TrainingManager


def format_ts(ts):
    try:
        return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


config = load_config()
logging.basicConfig(level=logging.INFO)

base_dir = Path(__file__).resolve().parent
data_dir = Path(config["data_dir"])
data_dir.mkdir(parents=True, exist_ok=True)

store = DataStore(str(data_dir / "power_classifier.sqlite"))
ha_client = HAClient(config["ha_base_url"], config["ha_token"])
classifier = ClassifierService(str(data_dir / "model.pkl"))
regression_service = RegressionService()
mqtt_publisher = None
if config["mqtt_enabled"]:
    mqtt_publisher = MqttPublisher(
        host=config["mqtt_host"],
        port=config["mqtt_port"],
        username=config["mqtt_username"],
        password=config["mqtt_password"],
        client_id=config["mqtt_client_id"],
    )

training_manager = TrainingManager(store, classifier, regression_service, config, data_dir)
poller = PowerPoller(
    store, ha_client, classifier, regression_service, config, mqtt_publisher
)

templates = Jinja2Templates(directory=str(base_dir / "templates"))
templates.env.filters["format_ts"] = format_ts

