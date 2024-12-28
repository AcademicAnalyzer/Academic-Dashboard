import time

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from geopy.geocoders import Nominatim
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, when, array, lit
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType
from wordcloud import WordCloud
import numpy as np


def fetch_country_coordinates(countries_df):
    geolocator = Nominatim(user_agent="AcademicArticlesExplorer")
    country_coordinates = {}

    for country in countries_df['country'].unique():
        try:
            time.sleep(1)  # Delay to avoid rate-limiting
            location = geolocator.geocode(country)
            if location:
                country_coordinates[country] = (location.latitude, location.longitude)
            else:
                country_coordinates[country] = (0, 0)
        except Exception as e:
            print(f"Error fetching coordinates for {country}: {str(e)}")
            country_coordinates[country] = (0, 0)

    return country_coordinates


def define_mongodb_schema():
    return StructType([
        StructField("title", StringType(), True),
        StructField("year", IntegerType(), True),
        StructField("journal", StringType(), True),
        StructField("issn", StringType(), True),
        StructField("countries", ArrayType(StringType()), True),
        StructField("authors", ArrayType(StringType()), True),
        StructField("keywords", ArrayType(StringType()), True),
        StructField("universities", ArrayType(StringType()), True)
    ])


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


def generate_wordcloud(keywords):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(keywords))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig


def create_collaboration_network(df, min_collaborations=2):
    """Create a network visualization of author collaborations."""
    import networkx as nx

    # Create collaboration pairs
    collaborations = []
    for authors in df['authors']:
        if isinstance(authors, list) and len(authors) > 1:
            for i in range(len(authors)):
                for j in range(i + 1, len(authors)):
                    collaborations.append(tuple(sorted([authors[i], authors[j]])))

    # Count collaborations
    collab_counts = pd.Series(collaborations).value_counts()
    significant_collabs = collab_counts[collab_counts >= min_collaborations]

    # Create network
    G = nx.Graph()
    for (author1, author2), count in significant_collabs.items():
        G.add_edge(author1, author2, weight=count)

    # Get positions
    pos = nx.spring_layout(G)

    # Create plot
    edge_trace = go.Scatter(
        x=[], y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)

    node_trace = go.Scatter(
        x=[], y=[],
        text=[],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            size=10,
            color='#1f77b4',
        ))

    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        node_trace['text'] += (node,)

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig

def main():
    st.set_page_config(layout="wide", page_title="Academic Articles Explorer")

    # Styling
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

    spark = init_spark_session()

    try:
        # Load data from MongoDB
        df = spark.read.format("mongo") \
            .option("spark.mongodb.input.database", "AcademicAnalyzer") \
            .option("spark.mongodb.input.collection", "Articles") \
            .schema(define_mongodb_schema()) \
            .load()
        # Preprocess data
        df = df.withColumn("countries",
                           when(col("countries").isNull(), array(lit("Unknown")))
                           .otherwise(col("countries")))
        df = df.withColumn("country", explode(col("countries")))

        # Get country coordinates
        countries_df = df.select("country").distinct().toPandas()
        COUNTRY_COORDINATES = fetch_country_coordinates(countries_df)

        # Convert to Pandas for visualization
        pandas_df = df.toPandas()

        st.markdown('<h1 class="main-title">🌍 Academic Articles Global Explorer</h1>', unsafe_allow_html=True)
        st.sidebar.header("📊 Analysis Filters")

        # Sidebar Filters
        available_years = sorted(pandas_df['year'].dropna().unique().astype(int))
        selected_year_range = st.sidebar.slider(
            "Select Year Range",
            min_value=min(available_years),
            max_value=max(available_years),
            value=(min(available_years), max(available_years))
        )

        available_countries = sorted(pandas_df['country'].unique())
        selected_countries = st.sidebar.multiselect(
            "Select Countries",
            options=available_countries,
            default=available_countries[:5] if len(available_countries) > 5 else available_countries
        )

        # Filter data
        filtered_df = pandas_df[
            (pandas_df['year'] >= selected_year_range[0]) &
            (pandas_df['year'] <= selected_year_range[1]) &
            (pandas_df['country'].isin(selected_countries))
            ]

        tab1, tab2, tab3 = st.tabs(["📊 Main Metrics", "🔍 Detailed Analysis", "🤝 Collaboration Insights"])

        with tab1:
            row1_col1, row1_col2 = st.columns(2)
            row2_col1, row2_col2 = st.columns(2)

            # Geographical Visualization
            with row1_col1:
                st.subheader("🌐 Global Publications Heatmap")
                geo_data = filtered_df.groupby('country').size().reset_index(name='publications')
                geo_data['lat'] = geo_data['country'].map(lambda x: COUNTRY_COORDINATES.get(x, [0, 0])[0])
                geo_data['lon'] = geo_data['country'].map(lambda x: COUNTRY_COORDINATES.get(x, [0, 0])[1])

                fig_geo = px.scatter_geo(
                    geo_data,
                    lat='lat',
                    lon='lon',
                    size='publications',
                    color='publications',
                    hover_name='country',
                    projection='natural earth'
                )
                fig_geo.update_layout(height=400)
                st.plotly_chart(fig_geo, use_container_width=True)

            # Keyword Word Cloud
            with row1_col2:
                st.subheader("🌟 Keyword Word Cloud")
                if len(filtered_df) > 0:
                    keywords = filtered_df['keywords'].dropna().explode().tolist()
                    fig_wordcloud = generate_wordcloud(keywords)
                    st.pyplot(fig_wordcloud)

            # Publications Trend
            with row2_col1:
                st.subheader("📈 Publications Trend")
                publications_over_time = filtered_df.groupby('year').size()
                fig_time = px.line(
                    x=publications_over_time.index,
                    y=publications_over_time.values,
                    labels={'x': 'Year', 'y': 'Number of Publications'}
                )
                fig_time.update_layout(height=400)
                st.plotly_chart(fig_time, use_container_width=True)

            # Journal Distribution
            with row2_col2:
                st.subheader("📚 Journal Distribution")
                journal_counts = filtered_df['journal'].value_counts().head(10)
                fig_journal = px.pie(
                    values=journal_counts.values,
                    names=journal_counts.index,
                    title='Top 10 Journals by Publication Count'
                )
                fig_journal.update_layout(height=400)
                st.plotly_chart(fig_journal, use_container_width=True)

        with tab2:
            # Author Publication Count
            st.subheader("👥 Top Authors by Publication Count")
            author_counts = filtered_df['authors'].explode().value_counts().head(15)
            fig_authors = px.bar(
                x=author_counts.index,
                y=author_counts.values,
                labels={'x': 'Author', 'y': 'Number of Publications'},
                title='Top 15 Authors by Publication Count'
            )
            fig_authors.update_layout(height=400)
            st.plotly_chart(fig_authors, use_container_width=True)

            # Publication Types Distribution
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📅 Publications by Month")
                # Assuming we can extract month from the data
                monthly_dist = filtered_df.groupby('year').size().reset_index()
                monthly_dist.columns = ['Year', 'Count']
                fig_monthly = px.bar(monthly_dist, x='Year', y='Count',
                                     title='Publication Distribution by Year')
                st.plotly_chart(fig_monthly, use_container_width=True)

            with col2:
                st.subheader("🏢 University Distribution")
                uni_counts = filtered_df['universities'].explode().value_counts().head(10)
                fig_unis = px.pie(values=uni_counts.values, names=uni_counts.index,
                                  title='Top 10 Universities by Publication Count')
                st.plotly_chart(fig_unis, use_container_width=True)

            # Keywords Co-occurrence
            st.subheader("🔤 Keyword Co-occurrence Matrix")
            all_keywords = filtered_df['keywords'].explode().value_counts().head(10).index
            keyword_matrix = np.zeros((len(all_keywords), len(all_keywords)))

            for keywords in filtered_df['keywords']:
                if isinstance(keywords, list):
                    for i, kw1 in enumerate(all_keywords):
                        for j, kw2 in enumerate(all_keywords):
                            if kw1 in keywords and kw2 in keywords:
                                keyword_matrix[i][j] += 1

            fig_heatmap = go.Figure(data=go.Heatmap(
                z=keyword_matrix,
                x=all_keywords,
                y=all_keywords,
                colorscale='Viridis'))
            fig_heatmap.update_layout(height=500)
            st.plotly_chart(fig_heatmap, use_container_width=True)

        with tab3:
            st.subheader("🤝 Author Collaboration Network")
            # Create and display collaboration network
            collab_fig = create_collaboration_network(filtered_df)
            st.plotly_chart(collab_fig, use_container_width=True)

            # Cross-country Collaboration Analysis
            st.subheader("🌍 International Collaboration Metrics")
            col1, col2 = st.columns(2)

            with col1:
                # Country collaboration matrix
                country_pairs = []
                for countries in filtered_df['countries']:
                    if isinstance(countries, list) and len(countries) > 1:
                        for i in range(len(countries)):
                            for j in range(i + 1, len(countries)):
                                country_pairs.append(tuple(sorted([countries[i], countries[j]])))

                collab_counts = pd.Series(country_pairs).value_counts().head(10)
                fig_country_collab = px.bar(
                    x=collab_counts.index.map(lambda x: f"{x[0]} - {x[1]}"),
                    y=collab_counts.values,
                    title='Top 10 Country Collaborations',
                    labels={'x': 'Country Pair', 'y': 'Number of Collaborations'}
                )
                st.plotly_chart(fig_country_collab, use_container_width=True)

            with col2:
                # Average authors per paper by country
                avg_authors = filtered_df.groupby('country').agg(
                    {'authors': lambda x: np.mean([len(i) if isinstance(i, list) else 0 for i in x])}
                ).reset_index()
                avg_authors.columns = ['Country', 'Avg Authors']
                fig_avg_authors = px.bar(
                    avg_authors.sort_values('Avg Authors', ascending=False).head(10),
                    x='Country',
                    y='Avg Authors',
                    title='Average Number of Authors per Paper by Country'
                )
                st.plotly_chart(fig_avg_authors, use_container_width=True)

        # Summary Statistics
        st.subheader("🔢 Key Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Publications", len(filtered_df))
        with col2:
            st.metric("Total Authors", len(filtered_df['authors'].dropna().explode().unique()))
        with col3:
            st.metric("Total Journals", len(filtered_df['journal'].dropna().unique()))
        with col4:
            st.metric("Total Countries", len(filtered_df['country'].dropna().unique()))

        # Download Filtered Data
        st.sidebar.download_button(
            "Download Data as CSV",
            filtered_df.to_csv(index=False),
            file_name="academic_articles_data.csv"
        )

    except Exception as e:
        st.error(f"Error loading MongoDB data: {str(e)}")


if __name__ == "__main__":
    main()