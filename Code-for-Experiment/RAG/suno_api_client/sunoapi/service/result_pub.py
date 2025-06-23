import json
import pika
from sunoapi.service.mqc_helper import PUBQ_NAME, get_mq_cnx

def send_push(msg: dict, qname: str = PUBQ_NAME):
    connection = get_mq_cnx()
    chan_op = connection.channel()
    chan_op.queue_declare(queue=qname, durable=True)
    chan_op.basic_publish(
        exchange='',
        routing_key=qname,
        body=json.dumps(msg, ensure_ascii=False, separators=(',', ':')),
        properties=pika.BasicProperties(
            delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE
        ),
    )
    chan_op.close()
    connection.close()