from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.types import StringType, FloatType, StructType, StructField
from pyspark.mllib.classification import LogisticRegressionModel

header = 'FL_DATE,UNIQUE_CARRIER,AIRLINE_ID,CARRIER,FL_NUM,ORIGIN_AIRPORT_ID,ORIGIN_AIRPORT_SEQ_ID,ORIGIN_CITY_MARKET_ID,ORIGIN,DEST_AIRPORT_ID,DEST_AIRPORT_SEQ_ID,DEST_CITY_MARKET_ID,DEST,CRS_DEP_TIME,DEP_TIME,DEP_DELAY,TAXI_OUT,WHEELS_OFF,WHEELS_ON,TAXI_IN,CRS_ARR_TIME,ARR_TIME,ARR_DELAY,CANCELLED,CANCELLATION_CODE,DIVERTED,DISTANCE,DEP_AIRPORT_LAT,DEP_AIRPORT_LON,DEP_AIRPORT_TZOFFSET,ARR_AIRPORT_LAT,ARR_AIRPORT_LON,ARR_AIRPORT_TZOFFSET,EVENT,NOTIFY_TIME'


def get_structfield(colname):
   if colname in ['ARR_DELAY', 'DEP_DELAY', 'DISTANCE', 'TAXI_OUT']:
      return StructField(colname, FloatType(), True)
   else:
      return StructField(colname, StringType(), True)


def to_example(raw_data_point):
  return LabeledPoint(\
              float(raw_data_point['ARR_DELAY'] < 15),
              [ \
                  raw_data_point['DEP_DELAY'], \
                  raw_data_point['TAXI_OUT'], \
                  raw_data_point['DISTANCE'], \
              ])


schema = StructType([get_structfield(colname) for colname in header.split(',')])


def main():
    BUCKET=os.environ['BUCKET']
    traindays = spark.read \
        .option("header", "true") \
        .csv('gs://{}/flights/trainday.csv'.format(BUCKET))
    traindays.createOrReplaceTempView('traindays')
    spark.sql("SELECT * from traindays ORDER BY FL_DATE LIMIT 5").show()
    inputs = 'gs://{}/flights/tzcorr/all_flights-00004-*'.format(BUCKET)
    flights = spark.read \
        .schema(schema) \
        .csv(inputs)
    flights.createOrReplaceTempView('flights')
    trainquery = """
    SELECT
      F.DEP_DELAY,F.TAXI_OUT,f.ARR_DELAY,F.DISTANCE
    FROM flights f
    JOIN traindays t
    ON f.FL_DATE == t.FL_DATE
    WHERE
      t.is_train_day == 'True'
    """
    traindata = spark.sql(trainquery)
    trainquery = """
    SELECT
      DEP_DELAY, TAXI_OUT, ARR_DELAY, DISTANCE
    FROM flights f
    JOIN traindays t
    ON f.FL_DATE == t.FL_DATE
    WHERE
      t.is_train_day == 'True' AND
      f.CANCELLED == '0.00' AND
      f.DIVERTED == '0.00'
    """
    traindata = spark.sql(trainquery)
    traindata.describe().show()
    examples = traindata.rdd.map(to_example)
    lrmodel = LogisticRegressionWithLBFGS.train(examples, intercept=True)
    print(lrmodel.weights, lrmodel.intercept)
    MODEL_FILE = 'gs://' + BUCKET + '/flights/sparkmloutput/model'
    os.system('gsutil -m rm -r ' + MODEL_FILE)
    lrmodel.save(sc, MODEL_FILE)
    print('{} saved'.format(MODEL_FILE))
    lrmodel = LogisticRegressionModel.load(sc, MODEL_FILE)
    lrmodel.setThreshold(0.7)
    print(lrmodel.weights)