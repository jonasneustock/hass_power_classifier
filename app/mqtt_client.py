import json
import logging
import threading

import paho.mqtt.client as mqtt


class MqttPublisher:
    def __init__(self, host, port, username, password, client_id):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.client_id = client_id
        self.client = None
        self.connected = threading.Event()

    def connect(self):
        self.client = mqtt.Client(client_id=self.client_id, protocol=mqtt.MQTTv311)
        if self.username:
            self.client.username_pw_set(self.username, self.password)

        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                logging.info("MQTT connected to %s:%s", self.host, self.port)
                self.connected.set()
            else:
                logging.warning("MQTT connect failed with code %s", rc)

        def on_disconnect(client, userdata, rc):
            self.connected.clear()
            if rc != 0:
                logging.warning("MQTT disconnected with code %s", rc)

        self.client.on_connect = on_connect
        self.client.on_disconnect = on_disconnect
        self.client.connect(self.host, self.port, keepalive=30)
        self.client.loop_start()

    def publish_json(self, topic, payload, retain=True):
        if not self.client:
            logging.warning("MQTT publish skipped, client not connected")
            return False
        try:
            payload_str = json.dumps(payload)
        except Exception as exc:
            logging.warning("MQTT payload serialization failed: %s", exc)
            return False
        result = self.client.publish(topic, payload_str, retain=retain)
        return result.rc == mqtt.MQTT_ERR_SUCCESS

    def publish_value(self, topic, value, retain=True):
        if not self.client:
            logging.warning("MQTT publish skipped, client not connected")
            return False
        result = self.client.publish(topic, value, retain=retain)
        return result.rc == mqtt.MQTT_ERR_SUCCESS

    def close(self):
        if not self.client:
            return
        self.client.loop_stop()
        self.client.disconnect()
        self.client = None
