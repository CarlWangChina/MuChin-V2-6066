import sys
import re
import os
import base64
import requests
from urllib.parse import quote

def parse_args():
    if len(sys.argv) < 5:
        return "", "", "", "", "", "", ""
    inpute_sqlOrDir = sys.argv[1]
    output_table = sys.argv[2]
    partition = sys.argv[3]
    taskName = sys.argv[4]
    inputeTable = sys.argv[5]
    environment = sys.argv[6]
    shardingid = sys.argv[7]
    return inpute_sqlOrDir, output_table, partition, taskName, inputeTable, environment, shardingid

def parse_env():
    s3_enpoint = os.getenv('S3A_ENDPOINT')
    keyId = os.getenv('AWS_ACCESS_KEY_ID')
    access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    return s3_enpoint, keyId, access_key

def parseSql_AddShardingid(inpute_sql, shardingid):
    if (inpute_sql.find("limit") != -1):
        return inpute_sql
    else:
        return f"{inpute_sql} and shardingid={shardingid}"

def parse_file_To_bucket(inpute_Dir):
    dirs = inpute_Dir.split(os.sep)
    pattern = rf'^\/{dirs[1]}\/'
    newFilePath = re.sub(pattern, "", inpute_Dir)
    return dirs[1], newFilePath

def parse_filepath_To_SourDir(filePath, dirPrefix):
    pattern = rf'^{dirPrefix}'
    newFilePath = re.sub(pattern, "", filePath)
    dirs = newFilePath.split(os.sep)
    if (len(dirs) <= 2):
        return ""
    if (dirs[0] != ""):
        return dirs[0]
    return dirs[1]

def parse_filepath_To_utf8(filePath):
    dirs = filePath.split(os.sep)
    newDirs = []
    for dir in dirs:
        if (dir == ""):
            newDirs.append(dir)
            continue
        isBase64 = is_base64(dir)
        if isBase64:
            newDirs.append(base64.b64decode(dir).decode('utf-8'))
        else:
            newDirs.append(dir)
    newFilePath = os.sep.join(newDirs)
    return newFilePath

def is_base64(s):
    try:
        decoded = base64.b64decode(s, validate=True)
        decoded.decode('utf-8')
        return True
    except Exception:
        return False

def table_exists(spark, table_str):
    try:
        spark.catalog.refreshTable(f"{table_str}")
        return True
    except:
        return False

def execute_shardingid(taskName, shardingid):
    taskName = quote(taskName)