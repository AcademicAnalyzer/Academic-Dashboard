import time
import plotly.express as px
import streamlit as st
from geopy.geocoders import Nominatim
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, when, array, lit
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType


# Function to fetch country coordinates using geopy
def fetch_country_coordinates(countries_df):
    geolocator = Nominatim(user_agent="AcademicArticlesExplorer")
    country_coordinates = {}

    for country in countries_df['country'].unique():
        try:
            # Delay to avoid rate-limiting
            time.sleep(1)

            # Fetch coordinates
            location = geolocator.geocode(country)
            if location:
                country_coordinates[country] = (location.latitude, location.longitude)
            else:
                # Fallback if geocode fails
                country_coordinates[country] = (0, 0)
        except Exception as e:
            print(f"Error fetching coordinates for {country}: {str(e)}")
            country_coordinates[country] = (0, 0)

    return country_coordinates

# Custom Schema Definition
def define_mongodb_schema():
    return StructType([
        StructField("title", StringType(), True),
        StructField("year", IntegerType(), True),
        StructField("journal", StringType(), True),
        StructField("issn", StringType(), True),
        StructField("countries", ArrayType(StringType()), True),
        StructField("authors", ArrayType(StringType()), True),
        StructField("keywords", ArrayType(StringType()), True),
        StructField("universeties", ArrayType(StringType()), True)
    ])


# Initialize Spark Session with additional configurations
def init_spark_session():
    spark = SparkSession.builder \
        .appName("AcademicArticlesAnalysis") \
        .config("spark.driver.host", "localhost") \
        .config("spark.mongodb.input.uri",
                "mongodb+srv://Bakr:luvI58Ggd16G5t4y@myatlasclusteredu.hgzqemt.mongodb.net/AcademicAnalyzer.Articles") \
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
        .config("spark.hadoop.fs.defaultFS", "file:///") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    return spark


# Main Streamlit App
def main():
    st.set_page_config(layout="wide", page_title="Academic Articles Explorer")

    # Fetch country coordinates using geopy

    # Initialize Spark Session
    spark = init_spark_session()

    # Load data from MongoDB with explicit schema
    try:
        # Read with explicit schema to avoid type inference issues
        df = spark.read.format("mongo") \
            .option("spark.mongodb.input.database", "AcademicAnalyzer") \
            .option("spark.mongodb.input.collection", "Articles") \
            .schema(define_mongodb_schema()) \
            .load()
                                                        
        # Preprocess data
        df = df.withColumn("countries",
                           when(col("countries").isNull(), array(lit("Unknown")))
                           .otherwise(col("countries"))
                           )

        # Flatten and preprocess countries
        df = df.withColumn("country", explode(col("countries")))

        # Convert the 'country' column to a Pandas DataFrame before fetching coordinates
        countries_df = df.select("country").distinct().toPandas()
        COUNTRY_COORDINATES = fetch_country_coordinates(countries_df)

        # Convert to Pandas for Plotly (with sampling for large datasets)
        pandas_df = df.limit(10000).toPandas()  # Limit to prevent memory issues

        # Title and styling
        st.markdown("""
        <style>
        .main-title {
            font-size: 3rem;
            color: #2C3E50;
            text-align: center;
            margin-bottom: 30px;
        }
        .stApp {
            background-color: #F0F4F8;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<h1 class="main-title">🌍 Academic Articles Global Explorer</h1>', unsafe_allow_html=True)

        # Sidebar Filters
        st.sidebar.header("📊 Analysis Filters")

        # Process and get available years and countries
        available_years = sorted(pandas_df['year'].dropna().unique().astype(int))
        available_countries = sorted(pandas_df['country'].unique())

        selected_years = st.sidebar.multiselect(
            "Select Years",
            options=available_years,
            default=available_years
        )

        selected_countries = st.sidebar.multiselect(
            "Select Countries",
            options=available_countries,
            default=available_countries[:5] if len(available_countries) > 5 else available_countries
        )

        # Filter data using Spark SQL
        spark_filtered_df = df.filter(
            (col("year").isin(selected_years)) &
            (col("country").isin(selected_countries))
        )

        # Convert back to Pandas for visualization (with sampling)
        filtered_pandas_df = spark_filtered_df.limit(10000).toPandas()

        # Prepare data for geographical visualization
        geo_data = filtered_pandas_df.groupby('country').size().reset_index(name='publications')
        geo_data['lat'] = geo_data['country'].map(lambda x: COUNTRY_COORDINATES.get(x, [0, 0])[0])
        geo_data['lon'] = geo_data['country'].map(lambda x: COUNTRY_COORDINATES.get(x, [0, 0])[1])

        # Create a grid of visualizations
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)

        # Geographical Visualization
        with row1_col1:
            st.subheader("🌐 Global Publications Heatmap")
            try:
                # Create geographical scatter plot
                fig_geo = px.scatter_geo(geo_data,
                                         lat='lat',
                                         lon='lon',
                                         size='publications',
                                         color='publications',
                                         hover_name='country',
                                         projection='natural earth')
                fig_geo.update_layout(height=400)
                st.plotly_chart(fig_geo, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating geographical visualization: {str(e)}")

        # Journal Distribution
        with row1_col2:
            st.subheader("📚 Journal Distribution")
            try:
                journal_counts = filtered_pandas_df['journal'].value_counts().head(10)
                fig_journal = px.pie(
                    values=journal_counts.values,
                    names=journal_counts.index,
                    title='Top 10 Journals by Publication Count'
                )
                fig_journal.update_layout(height=400)
                st.plotly_chart(fig_journal, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating journal distribution chart: {str(e)}")

        # Publications Over Time
        with row2_col1:
            st.subheader("📈 Publications Trend")
            try:
                # Time series of publications
                publications_over_time = filtered_pandas_df.groupby('year').size()
                fig_time = px.line(
                    x=publications_over_time.index,
                    y=publications_over_time.values,
                    labels={'x': 'Year', 'y': 'Number of Publications'}
                )
                fig_time.update_layout(height=400)
                st.plotly_chart(fig_time, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating publications trend chart: {str(e)}")

        # Country-wise Publication Distribution
        with row2_col2:
            st.subheader("🌍 Country Publication Distribution")
            try:
                country_counts = filtered_pandas_df['country'].value_counts()
                fig_country = px.bar(
                    x=country_counts.index,
                    y=country_counts.values,
                    labels={'x': 'Country', 'y': 'Number of Publications'}
                )
                fig_country.update_layout(height=400)
                st.plotly_chart(fig_country, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating country distribution chart: {str(e)}")

        # Summary Statistics
        st.subheader("🔢 Key Statistics")
        num_publications = len(filtered_pandas_df)
        num_authors = len(filtered_pandas_df['authors'].dropna().explode().unique())
        num_journals = len(filtered_pandas_df['journal'].dropna().unique())
        num_countries = len(filtered_pandas_df['country'].dropna().unique())

        st.write(f"Total Publications: {num_publications}")
        st.write(f"Total Authors: {num_authors}")
        st.write(f"Total Journals: {num_journals}")
        st.write(f"Total Countries: {num_countries}")

    except Exception as e:
        st.error(f"Error loading MongoDB data: {str(e)}")


if __name__ == "__main__":
    main()
