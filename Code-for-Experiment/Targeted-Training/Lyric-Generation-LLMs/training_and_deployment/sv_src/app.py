import logging
logging.basicConfig(format='%(asctime)s|%(levelname)s|%(process)d-%(thread)d|%(name)s: %(message)s', level=logging.INFO)
_logger = logging.getLogger(__name__)
import json
import multiprocessing
import os
from compose_poem2 import compose_poem, init_all_models
from mqc_helper import consume, produce

def assemble_ok_dicts(body_jd: "list[dict]"):
    ok_msgs = [{"mid": each["mid"], "oss": "", "server": "ama-prof-divi-libai"} for each in body_jd]
    return {"type": "ok", "msg": ok_msgs}

def core_inferring(body_jd: "list[dict]"):
    su_dict = {"type": "succ", "msg": []}
    for each in body_jd:
        mid = each["mid"]
        msg_op = {"mid": mid, "server": "ama-prof-divi-libai"}
        al, og = compose_poem(mid, each['prompt'], each['oss'])
        _logger.info(f"mid-{mid} | Final aligned lyric:\n{al}")
        msg_op['oss'] = al
        msg_op['wholeoss'] = og
        su_dict['msg'].append(msg_op)
    return su_dict

def callback(_0, _1, _2, body: bytes):
    print(f" [x] Received message body: {body.decode()}")
    jd_body = list(json.loads(body))
    print(f" [x] ATTN: Sending the ok msg immediately!")
    produce(assemble_ok_dicts(jd_body))
    print(f" [x] Start doing inference works...")
    produce(core_inferring(jd_body))
    print(f" [x] Done! succ type msg has been sent.")

def subs_and_push():
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    init_all_models()
    consume(callback)

def main():
    workers_num = 4
    mpp = multiprocessing.Pool(processes=workers_num)
    for _ in range(workers_num):
        mpp.apply_async(subs_and_push)
    mpp.close()
    mpp.join()

if __name__ == "__main__":
    main()