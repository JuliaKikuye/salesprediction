from pyspark.sql.functions import dayofweek, hour, col, date_format, datediff, lit
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler

def preprocessing_data(spark_df):
    # Extrai a parte do tempo corretamente da coluna `transaction_time`
    spark_df = spark_df.withColumn('transaction_time', date_format(col('transaction_time'), 'HH:mm:ss'))
    
    # Cria colunas day_of_week e hour_of_day
    spark_df = spark_df.withColumn('day_of_week', dayofweek(col('transaction_date')))
    spark_df = spark_df.withColumn('hour_of_day', hour(col('transaction_time')))

    # Cria coluna hour_day_interaction
    spark_df = spark_df.withColumn('hour_day_interaction', col('hour_of_day') * col('day_of_week'))

    return spark_df

def create_features(spark_df, categorical_cols, num_cols):
    
    original_cols = spark_df.columns
    
    # Codificação das variáveis categóricas
    indexers = [StringIndexer(inputCol=column, outputCol=column+"_index") for column in categorical_cols]

    for indexer in indexers:
        spark_df = indexer.fit(spark_df).transform(spark_df)

    input_cols_encoder = [column+"_index" for column in categorical_cols]
    output_cols_encoder = [column+"_vec" for column in categorical_cols]
    encoder = OneHotEncoder(
        inputCols=input_cols_encoder, 
        outputCols=output_cols_encoder,
    )
    spark_df = encoder.fit(spark_df).transform(spark_df)

    # Montar o vetor de features
    input_cols_assembler = output_cols_encoder + num_cols
    assembler = VectorAssembler(inputCols=input_cols_assembler, outputCol='features')
    spark_df = assembler.transform(spark_df)
    
    return spark_df.select(original_cols + ['features'])