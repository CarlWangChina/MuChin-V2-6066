import mysql.connector
import os
import boto3
import random

bucket_name = "ama-prof-divi-changba"
region_name = 'us-west-1'
aws_access_key_id = ''
aws_secret_access_key = ''

dbconfig = {
    'user': 'music_data',
    'password': 'JR4',
    'host': '3.101.37.186',
    'database': 'mysong',
    'port': 3301,
}

def get_full_rows():
    try:
        connection = mysql.connector.connect(**dbconfig)
        cursor = connection.cursor()
        query = "SELECT * FROM songs"
        cursor.execute(query)
        results = cursor.fetchall()
        return results
    except mysql.connector.Error as error:
        print(f"Error while connecting to MySQL: {error}")
        return None
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

def download_file_from_s3(local_path, key):
    s3 = boto3.client(
        's3',
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    try:
        s3.download_file(bucket_name, key, local_path)
    except Exception as e:
        print(f"File {key} download failed: {e}")

def download_files(download_root):
    data = get_full_rows()
    for row in data:
        download_dir = f"{download_root}/{row[0]}"
        os.makedirs(download_dir, exist_ok=True)
        local_path_lrc = f"{download_dir}/lrc.txt"
        local_path_audio = f"{download_dir}/full.mp3"
        download_file_from_s3(local_path_audio, row[1])
        with open(local_path_lrc, 'w', encoding='utf-8') as f:
            f.write(row[3])
        print(f"Downloaded {row[0]}")

download_files("./download/")