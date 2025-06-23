from ama_prof_divi_codec import ama_prof_diviCodec
import os
import sys
import ssl
import torchaudio
import requests
import soundfile as sf
import pyloudnorm as pyln
import Code_for_Experiment.Metrics.music_understanding_model.tensorboardX.tests.test_numpy
basePath = os.path.dirname(os.path.realpath(__file__))
pathArr = basePath.split('add_FileToIceberg_ForBase64')
print(pathArr)
if pathArr[0] not in sys.path:
    sys.path.insert(0, pathArr[0])
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, lit
from pyspark.sql.types import StructType, StructField, StringType, BinaryType
import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.amp.utils
from minio import Minio
from minio.error import S3Error
from urllib3 import PoolManager
from urllib3.util.ssl_ import create_urllib3_context
def clap_processor_func(mp3File, obj_id, out_bucket, out_table, out_column_name, endpoint, accessKey, secretKey):
    print(f"endpoint:{endpoint} mp3File:{mp3File}\n")
    codeName = "clap_processor_func"
    url = 'http://42.81.227.59:8000/CLAP'
    try:
        http_client = None
        secure = False
        if "shandong" in endpoint:
            ca_certs_path = "/usr/local/share/ca-certificates/dntq-ca.crt"
            ssl_context = ssl.create_default_context(cafile=ca_certs_path)
            http_client = PoolManager(ssl_context=ssl_context)
            secure = True
        minio_client = Minio(
            endpoint,
            access_key=accessKey,
            secret_key=secretKey,
            secure=secure,
            http_client=http_client
        )
        fileContent = datahouseUtil.read_oss_file(minio_client, mp3File)
        if fileContent is None:
            print(f"{codeName} Error reading file {mp3File}\n")
            return None
        response = requests.post(url, data=fileContent)
        if response.status_code == 200:
            outFile = datahouseUtil.table_name_to_path(out_bucket, out_table, out_column_name, obj_id, ".bin")
            if outFile == "":
                print(f"{codeName} Error make_out_filedir {out_bucket} {out_table} {out_column_name}")
                return None
            res = datahouseUtil.write_oss_file(minio_client, outFile, response.content)
            if res == False:
                print(f"{codeName} Error write_oss_file {outFile}")
                return None
            print(f"{codeName}_outFile:{outFile}")
            return outFile
        else:
            print('[clap_processor_func]request error:', response.status_code)
            return None
    except Exception as e:
        print(f"{codeName}_outFile Error processing file {mp3File} {out_bucket}: {e}")
        return None
def table_create_if_not_exists(spark, out_table_path):
    if not datahouseUtil.table_exists(spark=spark, table_str=out_table_path):
        try:
            spark.sql(f"CREATE TABLE {out_table_path} USING DELTA LOCATION '{out_table_path}'")
        except Exception as e:
            print(f"Error in CREATE TABLE: {e}")
            return
        print(f"Table {out_table_path} created.")
    else:
        print(f"Table {out_table_path} already exists.")
        return
def main():
    print("begin run")
    inputSql, outTable, partition, taskName, inputeTable, environment, shardingid, out_bucket = datahouseUtil.parse_args()
    print(f"{inputSql}, {outTable}, {partition}, {taskName}, {inputeTable}, {environment}, {shardingid}, {out_bucket} ")
    endpoint, accessKey, secretKey = datahouseUtil.parse_env()
    if inputSql == "":
        print(f"{taskName} args error {inputSql} {outTable} {partition}")
        return
    spark = SparkSession.builder.appName("OSS Example").getOrCreate()
    spark.sparkContext.setLogLevel("INFO")
    extract_features_udf = udf(clap_processor_func, StringType())
    table_create_if_not_exists(spark, outTable)
    df = spark.sql(inputSql)
    if df.isEmpty():
        print(f"{inputSql} df is empyt compute end")
        return
    print(f"begin withColumn")
    newdf = df.withColumn("outfile", extract_features_udf(df["filename"], df["obj_id"], lit(out_bucket), lit(outTable), lit("outfile"), lit(endpoint), lit(accessKey), lit(secretKey))) \
        .select("obj_id", "tagpartition", "outfile").filter("outfile IS NOT NULL")
    newdf.writeTo(outTable) \
        .overwritePartitions()
    print(f"end withColumn")
    spark.stop()
if __name__ == "__main__":
    main()