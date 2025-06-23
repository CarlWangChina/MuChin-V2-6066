import pika
import sunoapi.config_loader

MQCP_HOST = sunoapi.config_loader.config.mq.MQCP_HOST
MQCP_PORT = sunoapi.config_loader.config.mq.MQCP_PORT
MQCP_VHOST = sunoapi.config_loader.config.mq.MQCP_VHOST
MQCP_USER = sunoapi.config_loader.config.mq.MQCP_USER
MQCP_PASS = sunoapi.config_loader.config.mq.MQCP_PASS
SUBQ_NAME = sunoapi.config_loader.config.mq.SUBQ_NAME
PUBQ_NAME = sunoapi.config_loader.config.mq.PUBQ_NAME

def get_mq_cnx():
    credentials = pika.PlainCredentials(
        username=MQCP_USER,
        password=MQCP_PASS,
        erase_on_connect=True
    )
    parameters = pika.ConnectionParameters(
        host=MQCP_HOST,
        port=MQCP_PORT,
        virtual_host=MQCP_VHOST,
        credentials=credentials
    )
    return pika.BlockingConnection(parameters)