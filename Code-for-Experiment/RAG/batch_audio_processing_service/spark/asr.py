import os
import sys
import gc

basePath = os.path.dirname(os.path.realpath(__file__))
pathArr = basePath.split('add_FileToIceberg_ForBase64')
print(pathArr)
if pathArr[0] not in sys.path:
    sys.path.insert(0, pathArr[0])

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, lit, base64
from pyspark.sql.types import StructType, StructField, StringType, BinaryType
from minio import Minio
import Argvparse
from pydub import AudioSegment
import soundfile as sf
import pyloudnorm as pyln
import Code_for_Experiment.Metrics.music_understanding_model.tensorboardX.tests.test_numpy
import torchaudio
import traceback
import librosa
import requests
import io
import re
import base64
from ama_prof_divi_codec import ama_prof_diviCodec
import pyarrow

spark = SparkSession.builder.appName("OSS Example").config("spark.sql.parquet.enableVectorizedReader", "false").getOrCreate()
spark.sparkContext.setLogLevel("INFO")
print("beigin config")
print(pyarrow.__version__)

def asr_processor_func(mp3File, endpoint, accessKey, secretKey):
    print("[asr_processor_func]start")
    url = 'http://42.81.227.59:8000/ASR'
    try:
        print("[asr_processor_func]process", mp3File)
        dirs = mp3File.split(os.sep)
        pattern = rf'^\/{dirs[1]}\/'
        newFilePath = re.sub(pattern, "", mp3File)
        bucket = dirs[1]
        minio_client = Minio(
            endpoint=endpoint,
            access_key=accessKey,
            secret_key=secretKey,
            secure=False
        )
        response = minio_client.get_object(bucket, newFilePath)
        content = response.read()
        response.close()
        response.release_conn()
        response = requests.post(url, data=content)
        if response.status_code == 200:
            return response.text
        else:
            print('[asr_processor_func]request error:', response.status_code)
            return None
    except Exception as e:
        print(f"[asr_processor_func]Error processing file: {e}")
        traceback.print_exc()
        return None

def table_create_if_not_exists(spark, out_table_path):
    if not Argvparse.table_exists(spark=spark, table_str=out_table_path):
        try:
            spark.sql(f"CREATE TABLE {out_table_path} USING iceberg LOCATION '{out_table_path}'")
        except Exception as e:
            print(f"Error in CREATE TABLE: {e}")
            return
        print(f"Table {out_table_path} created.")
    else:
        print(f"Table {out_table_path} already exists.")
        return

def main(spark):
    print("begin run")
    inputSql, outTable, partition, taskName, inputeTable, environment, shardingid = Argvparse.parse_args()
    endpoint, accessKey, secretKey = Argvparse.parse_env()
    if inputSql == "":
        print(f"{taskName} args error {inputSql} {outTable} {partition}")
        return
    extract_features_udf = udf(asr_processor_func, StringType())
    table_create_if_not_exists(spark, outTable)
    endSharding = 0
    try:
        sql = f"SELECT shardingid FROM {inputeTable} where tagpartition='{partition}' order by shardingid desc limit 1"
        count_df = spark.sql(sql)
        print(sql)
        endSharding = int(count_df.first()["shardingid"])
        print(f"endSharding is {endSharding}")
        if (endSharding > 0):
            endSharding = endSharding - 1
    except Exception as e:
        print(f"Error in count max shardingid operation: {e}")
        traceback.print_exc()
        spark.stop()
        return

    nowShardingid = int(shardingid)
    if (nowShardingid != 0):
        nowShardingid = 0
    while endSharding >= nowShardingid:
        if (environment == "TEST" and nowShardingid >= 1):
            break
        nowShardingid = nowShardingid + 1
        sql = Argvparse.parseSql_AddShardingid(inputSql, nowShardingid)
        if (environment == "TEST"):
            sql = inputSql
        try:
            df = spark.sql(sql)
            if df.isEmpty():
                print(f"{inputSql} df is empyt compute end")
                continue
        except Exception as e:
            print(f"{inputSql} df is error{e}")
            traceback.print_exc()
            spark.stop()
            return

        try:
            print(f"begin withColumn:{nowShardingid}")
            print(f"sql_content:{sql}")
            newdf = df.withColumn("content", extract_features_udf(df["filename"], lit(endpoint), lit(accessKey), lit(secretKey))) \
                .select("obj_id", "tagpartition", "shardingid", "content").filter("content IS NOT NULL")
            newdf.writeTo(outTable).overwritePartitions()
            print(f"end withColumn:{nowShardingid}")
            Argvparse.execute_shardingid(taskName, shardingid)
        except Exception as e:
            spark.stop()
            traceback.print_exc()
            print(f"Error in withColumn operation: {e}")
            return

        print(f"now don shardingid:{nowShardingid} max:{endSharding}")
    spark.stop()

if __name__ == "__main__":
    main(spark)