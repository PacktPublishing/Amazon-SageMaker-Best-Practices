from __future__ import print_function
from __future__ import unicode_literals

import argparse
import csv
import os
import shutil
import sys
import time
import logging
import boto3

import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import (
    StringIndexer,
    VectorAssembler,
    VectorIndexer,
    StandardScaler,
    OneHotEncoder
)
from pyspark.sql.functions import *
from pyspark.sql.functions import round as round_
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
    BooleanType,
    IntegerType
)

def get_tables():
    
    tables = ['0d18fb9e-857d-4380-bbac-ffbb60b07ae2']
    #tables = ['0d18fb9e-857d-4380-bbac-ffbb60b07ae2',
    #                       '1645e09b-0919-439a-b07f-cd7532069f10',
    #                       '80d785aa-6da8-4b37-8632-7386b1d535f3',
    #                       '8375c185-ea3e-44b7-b61c-68534e33ddf7',
    #                       '9a6e24db-cffc-42de-a77f-7ab96c487022',
    #                       'bc981da3-bc9d-435a-8bf2-4107a8fb2676',
    #                       'cf5cf814-c9e3-4a2b-8811-e4fc6481a1fe',
    #                       'd3b8f1ab-f3e5-4fc9-84ab-8568edd8a03d']
    

    return tables


def isBadAir(v, p):
    if p == 'pm10':
        if v > 50:
            return 1
        else:
            return 0
    elif p == 'pm25':
        if v > 25:
            return 1
        else:
            return 0
    elif p == 'so2':
        if v > 20:
            return 1
        else:
            return 0
    elif p == 'no2':
        if v > 200:
            return 1
        else:
            return 0
    elif p == 'o3':
        if v > 100:
            return 1
        else:
            return 0
    else:
        return 0

def extract(row):
    return (row.value, row.ismobile, row.year, row.month, row.quarter, row.day, row.isBadAir, 
            row.indexed_location, row.indexed_city, row.indexed_country, row.indexed_sourcename, 
            row.indexed_sourcetype)

"""
Schema on disk:

 |-- date_utc: string (nullable = true)
 |-- date_local: string (nullable = true)
 |-- location: string (nullable = true)
 |-- country: string (nullable = true)
 |-- value: float (nullable = true)
 |-- unit: string (nullable = true)
 |-- city: string (nullable = true)
 |-- attribution: array (nullable = true)
 |    |-- element: struct (containsNull = true)
 |    |    |-- name: string (nullable = true)
 |    |    |-- url: string (nullable = true)
 |-- averagingperiod: struct (nullable = true)
 |    |-- unit: string (nullable = true)
 |    |-- value: float (nullable = true)
 |-- coordinates: struct (nullable = true)
 |    |-- latitude: float (nullable = true)
 |    |-- longitude: float (nullable = true)
 |-- sourcename: string (nullable = true)
 |-- sourcetype: string (nullable = true)
 |-- mobile: string (nullable = true)
 |-- parameter: string (nullable = true)
 
Example output:

date_utc='2015-10-31T07:00:00.000Z'
date_local='2015-10-31T04:00:00-03:00'
location='Quintero Centro'
country='CL'
value=19.81999969482422
unit='µg/m³'
city='Quintero'
attribution=[Row(name='SINCA', url='http://sinca.mma.gob.cl/'), Row(name='CENTRO QUINTERO', url=None)]
averagingperiod=None
coordinates=Row(latitude=-32.786170959472656, longitude=-71.53143310546875)
sourcename='Chile - SINCA'
sourcetype=None
mobile=None
parameter='o3'

Transformations:

* Featurize date_utc
* Drop date_local
* Encode location
* Encode country
* Scale value
* Drop unit
* Encode city
* Drop attribution
* Drop averaging period
* Drop coordinates
* Encode source name
* Encode source type
* Convert mobile to integer
* Encode parameter

* Add label for good/bad air quality

"""
def main():
    parser = argparse.ArgumentParser(description="Preprocessing configuration")
    parser.add_argument("--s3_input_bucket", type=str, help="s3 input bucket")
    parser.add_argument("--s3_input_key_prefix", type=str, help="s3 input key prefix")
    parser.add_argument("--s3_output_bucket", type=str, help="s3 output bucket")
    parser.add_argument("--s3_output_key_prefix", type=str, help="s3 output key prefix")
    parser.add_argument("--parameter", type=str, help="parameter filter")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('Preprocess')

    spark = SparkSession.builder.appName("Preprocessor").getOrCreate()

    logger.info("Reading data set")
    tables = get_tables()
    df = spark.read.parquet(f"s3://{args.s3_input_bucket}/{args.s3_input_key_prefix}/{tables[0]}/")
    for t in tables[1:]:
        df_new = spark.read.parquet(f"s3://{args.s3_input_bucket}/{args.s3_input_key_prefix}/{t}/")
        df = df.union(df_new)
        
    # Filter on parameter
    df = df.filter(df.parameter == args.parameter)
    
    # Drop columns
    logger.info("Dropping columns")
    df = df.drop('date_local').drop('unit').drop('attribution').drop('averagingperiod').drop('coordinates')

    # Mobile field to int
    logger.info("Casting mobile field to int")
    df = df.withColumn("ismobile",col("mobile").cast(IntegerType())).drop('mobile')

    # scale value
    logger.info("Scaling value")
    value_assembler = VectorAssembler(inputCols=["value"], outputCol="value_vec")
    value_scaler = StandardScaler(inputCol="value_vec", outputCol="value_scaled")
    value_pipeline = Pipeline(stages=[value_assembler, value_scaler])
    value_model = value_pipeline.fit(df)
    xform_df = value_model.transform(df)

    # featurize date
    logger.info("Featurizing date")
    xform_df = xform_df.withColumn('aggdt', 
                   to_date(unix_timestamp(col('date_utc'), "yyyy-MM-dd'T'HH:mm:ss.SSSX").cast("timestamp")))
    xform_df = xform_df.withColumn('year',year(xform_df.aggdt)) \
        .withColumn('month',month(xform_df.aggdt)) \
        .withColumn('quarter',quarter(xform_df.aggdt))
    xform_df = xform_df.withColumn("day", date_format(col("aggdt"), "d"))

    # Automatically assign good/bad labels
    logger.info("Simulating good/bad air labels")
    isBadAirUdf = udf(isBadAir, IntegerType())
    xform_df = xform_df.withColumn('isBadAir', isBadAirUdf('value', 'parameter'))
    xform_df = xform_df.drop('parameter')

    # Categorical encodings.  
    logger.info("Categorical encoding")
    #parameter_indexer = StringIndexer(inputCol="parameter", outputCol="indexed_parameter", handleInvalid='keep')
    location_indexer = StringIndexer(inputCol="location", outputCol="indexed_location", handleInvalid='keep')
    city_indexer = StringIndexer(inputCol="city", outputCol="indexed_city", handleInvalid='keep')
    country_indexer = StringIndexer(inputCol="country", outputCol="indexed_country", handleInvalid='keep')
    sourcename_indexer = StringIndexer(inputCol="sourcename", outputCol="indexed_sourcename", handleInvalid='keep')
    sourcetype_indexer = StringIndexer(inputCol="sourcetype", outputCol="indexed_sourcetype", handleInvalid='keep')
    #enc_est = OneHotEncoder(inputCols=["indexed_parameter"], outputCols=["vec_parameter"])
    enc_pipeline = Pipeline(stages=[location_indexer, 
        city_indexer, country_indexer, sourcename_indexer, 
        sourcetype_indexer])
    enc_model = enc_pipeline.fit(xform_df)
    enc_df = enc_model.transform(xform_df)
    #param_cols = enc_df.schema.fields[17].metadata['ml_attr']['vals']

    # Clean up data set
    logger.info("Final cleanup")
    final_df = enc_df.drop('location') \
        .drop('city').drop('country').drop('sourcename') \
        .drop('sourcetype').drop('date_utc') \
        .drop('value_vec').drop('aggdt')
    firstelement=udf(lambda v:str(v[0]),StringType())
    final_df = final_df.withColumn('value_str', firstelement('value_scaled'))
    final_df = final_df.withColumn("value",final_df.value_str.cast(DoubleType())).drop('value_str').drop('value_scaled')
    schema = StructType([
        StructField("value", DoubleType(), True),
        StructField("ismobile", StringType(), True),
        StructField("year", StringType(), True),
        StructField("month", StringType(), True),
        StructField("quarter", StringType(), True),
        StructField("day", StringType(), True),
        StructField("isBadAir", StringType(), True),
        StructField("location", StringType(), True),
        StructField("city", StringType(), True),
        StructField("country", StringType(), True),
        StructField("sourcename", StringType(), True),
        StructField("sourcetype", StringType(), True)
                    ])
    final_df = final_df.rdd.map(extract).toDF(schema=schema)
    
    # Replace missing values
    final_df = final_df.na.fill("0")
    
    # Round the value
    final_df = final_df.withColumn("value", round_(final_df["value"], 4))

    # Split sets
    logger.info("Splitting data set")
    (train_df, validation_df, test_df) = final_df.randomSplit([0.7, 0.2, 0.1])
    
    # Drop value from test set
    test_df = test_df.drop('value')

    # Save to S3
    logger.info("Saving to S3")
    train_df.write.option("header",False).csv('s3://' + os.path.join(args.s3_output_bucket, 
        args.s3_output_key_prefix, 'train/'))
    validation_df.write.option("header",False).csv('s3://' + os.path.join(args.s3_output_bucket, 
        args.s3_output_key_prefix, 'validation/'))
    test_df.write.option("header",False).csv('s3://' + os.path.join(args.s3_output_bucket, 
        args.s3_output_key_prefix, 'test/'))

if __name__ == "__main__":
    main()