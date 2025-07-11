import logging
logging.basicConfig(
    format='%(asctime)s|%(levelname)s|'
           '%(process)d-%(thread)d|'
           '%(name)s: %(message)s',
    level=logging.INFO)
_logger = logging.getLogger(__name__)
import json
import time
import pika
from pika.adapters.blocking_connection import BlockingChannel

MQCP_HOST = 'rabbitmq-serverless-cn-jeo3og3a00f.cn-zhangjiakou.amqp-2.net.mq.amqp.aliyuncs.com'
MQCP_PORT = 5672
MQCP_VHOST = 'ama-prof-divi'
MQCP_USER = 'MjpyYWJiaXRtcS1zZXJ2ZXJsZXNzLWNuLWplbzNvZzNhMDBmOkxUQUk1dFFtZnVnOFJKUlV2aTdmUE5XZw=='
MQCP_PASS = 'RUY1NTQ2NUM3MDc0RUVBRERGOTEzQkI3RUNCNjZDOTYyODlGQkQ4NDoxNzExOTg1ODA3Mjk5'
SUBQ_NAME = 'create_music_tasktwo_cccr'
PUBQ_NAME = 'create_music_tasktwo_cccr_resp'

_cnx4ip: pika.BlockingConnection = None

def _get_mq_cnx4ip(force_new=False):
    global _cnx4ip
    if not force_new and _cnx4ip and \
            _cnx4ip.is_open and not _cnx4ip.is_closed:
        return _cnx4ip
    _cnx4ip = None
    credentials = pika.PlainCredentials(
        username=MQCP_USER,
        password=MQCP_PASS,
        erase_on_connect=True)
    cparameters = pika.ConnectionParameters(
        host=MQCP_HOST,
        port=MQCP_PORT,
        virtual_host=MQCP_VHOST,
        connection_attempts=2,
        credentials=credentials)
    _cnx4ip = pika.BlockingConnection(cparameters)
    return _cnx4ip

def _get_input_channel(force_new=False, qname=SUBQ_NAME):
    _chan_ip = _get_mq_cnx4ip(force_new).channel()
    _chan_ip.queue_declare(queue=qname, durable=True)
    _chan_ip.basic_qos(prefetch_count=1)
    return _chan_ip

def consume(callback, qname=SUBQ_NAME):
    cnx_rec_max_t = 8
    cnx_rec_times = 0
    while cnx_rec_times <= cnx_rec_max_t:
        try:
            chan_ip = _get_input_channel(cnx_rec_times > 0)
            chan_ip.basic_consume(
                queue=qname,
                auto_ack=True,
                on_message_callback=callback)
            chan_ip.start_consuming()
        except pika.exceptions.ConnectionClosedByBroker:
            _logger.exception("Un-recover sub-error_0.")
            break
        except pika.exceptions.AMQPChannelError:
            _logger.exception("Un-recover sub-error_1.")
            break
        except pika.exceptions.AMQPConnectionError:
            _logger.exception("Recoverable sub-error.")
            cnx_rec_times += 1
            time.sleep(pow(2, cnx_rec_times))
            continue
        except:
            _logger.exception("Un-recover sub-error_999.")
            break

_cnx4op: pika.BlockingConnection = None

def _get_mq_cnx4op(force_new=False):
    global _cnx4op
    if not force_new and _cnx4op and \
            _cnx4op.is_open and not _cnx4op.is_closed:
        return _cnx4op
    _cnx4op = None
    credentials = pika.PlainCredentials(
        username=MQCP_USER,
        password=MQCP_PASS,
        erase_on_connect=True)
    cparameters = pika.ConnectionParameters(
        host=MQCP_HOST,
        port=MQCP_PORT,
        virtual_host=MQCP_VHOST,
        connection_attempts=2,
        credentials=credentials)
    _cnx4op = pika.BlockingConnection(cparameters)
    return _cnx4op

_chan_op: BlockingChannel = None

def _get_ouput_channel(force_new=False, qname=PUBQ_NAME):
    global _chan_op
    if not force_new and _chan_op and \
            _chan_op.is_open and not _chan_op.is_closed:
        return _chan_op
    _chan_op = None
    _chan_op = _get_mq_cnx4op(True).channel()
    _chan_op.queue_declare(queue=qname, durable=True)
    return _chan_op

def produce(msg: dict, qname: str=PUBQ_NAME):
    tried_times = 0
    max_retry_t = 4
    success = False
    need_reconnect = False
    while not success and tried_times <= max_retry_t:
        try:
            chan_op = _get_ouput_channel(need_reconnect)
            chan_op.basic_publish(
                exchange='',
                routing_key=qname,
                body=json.dumps(msg, ensure_ascii=False,
                                 separators=(',', ':')),
                properties=pika.BasicProperties(
                    delivery_mode=pika.DeliveryMode.Persistent))
            success = True
        except pika.exceptions.AMQPConnectionError:
            need_reconnect = True
            _logger.exception(f'Failed to send message. t-{tried_times}')
        except:
            _logger.critical(f'Sending msg:\n{msg}', exc_info=1)
            break
        finally:
            tried_times += 1