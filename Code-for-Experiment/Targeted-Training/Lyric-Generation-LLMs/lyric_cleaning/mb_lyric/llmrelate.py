import logging
import os
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

load_dotenv(find_dotenv(), override=True)
oai_askey = os.getenv('OPENAI_API_KEY')
oai_mname = os.getenv('OPENAI_MODEL_N')
oai_mname = oai_mname if oai_mname else 'gpt-3.5-turbo'
oaic = OpenAI()
_req_logger = logging.getLogger(__name__ + '.gpt_req')

def gpt_req(sysct, usrct, mname=oai_mname, jm=False, mtemp=1.0):
    rf_type = "json_object" if jm else "text"
    response = oaic.chat.completions.create(
        model=mname,
        response_format={"type": rf_type},
        messages=[
            {"role": "system", "content": sysct},
            {"role": "user", "content": usrct}
        ],
        temperature=mtemp,
    )
    _req_logger.debug(f'raw GPT response:\n{response}')
    return response.choices[0].message.content

def gpt_req_wu(sysct, usrct, mname=oai_mname, jm=False, mtemp=1.0):
    rf_type = "json_object" if jm else "text"
    response = oaic.chat.completions.create(
        model=mname,
        response_format={"type": rf_type},
        messages=[
            {"role": "system", "content": sysct},
            {"role": "user", "content": usrct}
        ],
        temperature=mtemp,
    )
    _req_logger.info(f'raw GPT response:\n{response}')
    mc = response.choices[0].message.content
    pt: int = getattr(response.usage, 'prompt_tokens', 0)
    ct: int = getattr(response.usage, 'completion_tokens', 0)
    tt: int = getattr(response.usage, 'total_tokens', 0)
    return (mc, (pt, ct, tt))

oai_px_burl = os.getenv('OAI_PX_API_URL')
oai_px_skey = os.getenv('OAI_PX_API_KEY')
oai_pxc = OpenAI(api_key=oai_px_skey, base_url=oai_px_burl)

def pxg_req_wu(sysct, usrct, mname=oai_mname, jm=False, mtemp=1.0):
    rf_type = "json_object" if jm else "text"
    response = oai_pxc.chat.completions.create(
        model=mname,
        response_format={"type": rf_type},
        messages=[
            {"role": "system", "content": sysct},
            {"role": "user", "content": usrct}
        ],
        temperature=mtemp,
    )
    _req_logger.debug(f'raw GPT-PROXY response:\n{response}')
    mc = response.choices[0].message.content
    pt: int = getattr(response.usage, 'prompt_tokens', 0)
    ct: int = getattr(response.usage, 'completion_tokens', 0)
    tt: int = getattr(response.usage, 'total_tokens', 0)
    return (mc, (pt, ct, tt))