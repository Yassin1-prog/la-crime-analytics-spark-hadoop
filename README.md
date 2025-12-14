# LA Crime Analytics - Spark & Hadoop

Semester project for "Advanced Database Systems" (NTUA, 2025).

## Overview

This project implements data processing and geospatial analytics over Los Angeles crime datasets using Apache Spark (DataFrame, SQL, RDD APIs), Hadoop, and Apache Sedona. It includes performance evaluation under different execution configurations.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/la-crime-analytics-spark-hadoop.git
cd la-crime-analytics-spark-hadoop

# verify you are using python 3.10 and jdk 11
# otherwise the code might not work as expected
python --version
java -version

# Set up environment
# It is assumed you are on a sagemaker notebook on ntua aws
# with all the required dependencies set up
pip install apache-sedona
```

> it is expected that the datasets are available in AWS S3 cloud in //initial-notebook-data-bucket-dblab-905418150721/

## Usage

> Below are the commands to run all queries with configuration
> as specified in the project specifications

All queries are executed from the project root directory. Below are the commands to run each query with configurations as specified in the project specifications.

---

### Query 1: Aggravated Assault Victims by Age Group

**Purpose**: Rank age groups of victims in incidents involving any form of "aggravated assault" (Children <18, Young adults 18-24, Adults 25-64, Seniors >64).

**Configuration**: 4 executors, 1 core, 2GB memory

**DataFrame API** (default):

```bash
python queries/query1.py --mode df
```

**DataFrame API with UDF**:

```bash
python queries/query1.py --mode df_udf
```

**RDD API**:

```bash
python queries/query1.py --mode rdd
```

---

### Query 2: Top 3 Victim Descent Groups per Year

**Purpose**: Find the 3 racial/ethnic groups with the most crime victims per year, showing count and percentage of total victims.

**Configuration**: 4 executors, 1 core, 2GB memory

**DataFrame API** (default):

```bash
python queries/query2.py --mode df
```

**SQL API**:

```bash
python queries/query2.py --mode sql
```

---

### Query 3: Crime Methods (MO Codes) Frequency Analysis

**Purpose**: Rank crime methods (modus operandi) by frequency of occurrence, mapping codes to their descriptions.

**Configuration**: 4 executors, 1 core, 2GB memory

**DataFrame API with optimizer choice** (default):

```bash
python queries/query3.py --mode df
```

**DataFrame API with BROADCAST join**:

```bash
python queries/query3.py --mode df --join-strategy BROADCAST
```

**DataFrame API with MERGE join**:

```bash
python queries/query3.py --mode df --join-strategy MERGE
```

**DataFrame API with SHUFFLE_HASH join**:

```bash
python queries/query3.py --mode df --join-strategy SHUFFLE_HASH
```

**DataFrame API with SHUFFLE_REPLICATE_NL join**:

```bash
python queries/query3.py --mode df --join-strategy SHUFFLE_REPLICATE_NL
```

**RDD API**:

```bash
python queries/query3.py --mode rdd
```

---

### Query 4: Crimes per Police Station with Distance Analysis

**Purpose**: Calculate the number of crimes closest to each police station and the average distance from the station to those crime locations using geospatial analysis.

**Scaling Configurations** (2 executors):

**Configuration 1** - 1 core, 2GB memory:

```bash
python queries/query4.py --executors 2 --cores 1 --memory 2g
```

**Configuration 2** - 2 cores, 4GB memory:

```bash
python queries/query4.py --executors 2 --cores 2 --memory 4g
```

**Configuration 3** - 4 cores, 8GB memory:

```bash
python queries/query4.py --executors 2 --cores 4 --memory 8g
```

---

### Query 5: Income vs Crime Rate Correlation Analysis

**Purpose**: Calculate the correlation between median household income per capita and crime rate per capita (2020-2021) for LA communities. Repeat for top 10 and bottom 10 income communities.

**Total Resources**: 8 cores, 16GB memory

**Configuration 1** - 2 executors × 4 cores, 8GB memory:

```bash
python queries/query5.py --executors 2 --cores 4 --memory 8g
```

**Configuration 2** - 4 executors × 2 cores, 4GB memory:

```bash
python queries/query5.py --executors 4 --cores 2 --memory 4g
```

**Configuration 3** - 8 executors × 1 core, 2GB memory:

```bash
python queries/query5.py --executors 8 --cores 1 --memory 2g
```

---

### Implementation Notes

- **Query 1**: Compares performance between DataFrame (with/without UDF) and RDD implementations
- **Query 2**: Compares DataFrame and SQL API performance for the same analytical task
- **Query 3**: Tests different join strategies (BROADCAST, MERGE, SHUFFLE_HASH, SHUFFLE_REPLICATE_NL) to analyze optimizer choices
- **Query 4**: Evaluates resource scaling effects (cores and memory) on geospatial queries
- **Query 5**: Tests executor count impact on geospatial join performance with fixed total resources

All queries automatically print execution time and query plans (where applicable) using the [`run_and_time`](utils/timing.py) utility.
