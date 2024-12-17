#Importing Libraries
import streamlit as st
import numpy as np
import pandas as pd
import base64
import plotly.express as px
import seaborn as sns
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import joblib
import os

# Recomend 5 wines based on selected wine
def recommend_wines_kmeans(X,wine_name, num_recommendations=5):
    wine_index = kmeans_data[kmeans_data['Winename'] == wine_name].index[0]
    cluster_labels = kmeans_model.predict(X)
    X['Cluster_Kmeans'] = cluster_labels
    wine_cluster = X.loc[wine_index, 'Cluster_Kmeans']
    cluster_wines = kmeans_data[X['Cluster_Kmeans'] == wine_cluster]
    cluster_wines = cluster_wines[cluster_wines.index != wine_index]
    recommended_wines = cluster_wines['Winename'].head(num_recommendations)
    return recommended_wines.tolist()

# Display the wine name along with wine details 
def display_wine(wine_details, selected_wine):
    if not wine_details[wine_details['WineName'] == selected_wine].empty:
        wine_info = wine_details[wine_details['WineName'] == selected_wine].iloc[0]
        col1, col2 = st.columns([2, 4])  # Adjust column widths if necessary
        
        with col2:
            # Wine details in a single markdown
            st.markdown(f"""
            **Wine Name:** {wine_info['WineName']}  
            **Type of Wine:** {wine_info['Type']}  
            **Grapes Content:** {wine_info['Grapes']}  
            **Food Suggestion:** {wine_info['Harmonize']}  
            **Alcohol Content:** {wine_info['ABV']}%  
            **Country:** {wine_info['Country']}  
            **Acidity:** {wine_info['Acidity']}  
            **Website:** [Visit Website]({wine_info['Website']})
            """)
        
        with col1:
            # Check if the image file exists
            image_path = f"Data/XWines_Test_100_labels/{wine_info['WineID']}.jpeg"
            if os.path.exists(image_path):
                # Display the image if it exists
                st.image(
                    image_path, 
                    width=200, 
                    caption=wine_info['WineName']
                )
            else:
                # Display "No Image Available" message if the image is missing
                st.markdown("**No Image Available**")
    else:
        st.write("Wine details not found.")


# Reading Dataframe
df_exp= pd.read_csv('Data/EDA.csv')
validation= pd.read_csv('Data/Validation.csv')
kmeans_data= pd.read_csv('Model/kmeans_dataset.csv')
wine_details= pd.read_csv('Data/XWines_Test_100_wines.csv')


# Set Streamlit app config for a wider layout and light theme
st.set_page_config(layout="wide", page_title="Wine Recommendation App", initial_sidebar_state="expanded")

# Set background image using HTML and CSS
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        base64_image = base64.b64encode(image_data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/webp;base64,{base64_image}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
 
# Background image
pic = 'pics/wine-gone-bad.jpg'
set_background(pic)

# Page navigation state
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a Section", ["Introduction", "IDA", "EDA", "Feature Engineering", "Modeling", "Recomendation"])

# Function to load different sections
if options == "Introduction":
    st.markdown(
        "<h1 style='color: red;'>Introduction</h1>", 
        unsafe_allow_html=True
    )
    

    # First Text Block: Introduction to Wine
    st.markdown("""
    ### Let’s savor the timeless pour of the perfect red

    Wine is an alcoholic beverage made from fermented grapes or other fruits. Its origins trace back to around **6000 BC** in regions of modern-day Georgia, Iran, and Armenia. Early wine production spread to ancient Egypt and Mesopotamia, where it became a staple in religious ceremonies and royal banquets.

    The **Greeks and Romans** played key roles in advancing winemaking techniques and expanding wine culture. The Romans, in particular, cultivated vineyards across their empire, spreading wine throughout Europe, including present-day France, Spain, and Italy, regions that later became famous for their wine production.

    As European explorers and settlers ventured to the New World, they introduced wine to regions like **South Africa, Australia, and the Americas**. Wine took root in **California, Argentina, and Chile**, now major global wine producers. Today, wine is produced and enjoyed worldwide, with each region adding its own unique flavors, traditions, and styles to the global wine landscape.
    """)

    # Embed the YouTube video
    st.markdown("#### Learn More About Wine")
    components.iframe("https://www.youtube.com/embed/hGJWUg4wx78", width=560, height=315)

    # Second Text Block: Regional and Vintage Differences in Wine
    st.markdown("""
    ### Regional and Vintage Differences in Wine

    Wines originating from the same grape variety but different regions and vintages can vary significantly in taste due to several factors:

    **Terroir (Region-Specific Factors):**

    - **Climate:** Warmer regions produce wines with riper, bolder flavors (e.g., more fruit-forward), while cooler regions create more acidic and lighter wines. Soil type, altitude, and sunlight also influence flavor nuances.
    - **Soil Composition:** Different minerals in the soil can impart subtle flavors, such as earthy, mineral, or even floral notes.

    **Vintage (Year of Harvest):**

    - **Weather Conditions:** The growing season's temperature, rainfall, and sunshine affect the grapes' ripeness and sugar levels. A sunny year can produce a rich, full-bodied wine, while a cooler, wetter year might result in higher acidity and lighter body.
    - **Aging Potential:** Different vintages age differently, leading to changes in tannin structure, acidity, and overall complexity over time.

    **Winemaking Practices:**

    The winemaker’s approach, including fermentation techniques, use of oak barrels, and aging methods, can also introduce variations in flavor, even within the same variety.

    Ultimately, regional and vintage differences can create a broad spectrum of flavors, from fruity and bold to subtle, nuanced, and earthy, even in wines made from the same type of grape.
    """)

    # IDA page
elif options == "IDA":
    st.markdown(
        "<h1 style='color: red;'>Initial Data Analysis (IDA)</h1>", 
        unsafe_allow_html=True
    )

    st.markdown("""
    ### About the Data

    #### Source([Link](https://github.com/rogerioxavier/X-Wines/tree/main))
    - **WX-Wines**: X-Wines is a world wine dataset with 5-stars user ratings for recommender systems, machine learning, and other research purpose.
    It is published under free license for wider use.
    [Paper](https://www.mdpi.com/2504-2289/7/1/20) must be mandatory cited in publications that make use of this dataset, where you can also find additional information.

    #### Tables
    - **XWines_Test_100_wines**: Contains descriptions of different types of wines worldwide.
    - **XWines_Test_1K_ratings**: Contains the wine ratings by different users.
    - **Key_map**: Contains the mapping of ratings with user id and wine id.
    """)


    
    # Dropdown filter for selecting dataset
    st.markdown(""" ### Wine Data Summary""")
    option = st.selectbox(
        "Select a dataset to view details:",
        ("XWines_Test_100_wines", "XWines_Test_1K_ratings", "Key_map")
    )

    # Data description for XWines_Test_100_wines
    XWines_Test_100_wines = pd.DataFrame({
        "Column Name": ['WineID', 'WineName', 'Type', 'Elaborate', 'Grapes', 'Harmonize', 'ABV',
                        'Body', 'Acidity', 'Code', 'Country', 'RegionID', 'RegionName',
                        'WineryID', 'WineryName', 'Website', 'Vintages'],
        "Description": [
            "A unique identifier for each wine entry in the dataset.",
            "The name of the wine",
            "The category or type of wine (e.g., Red, White, Rosé, Sparkling, etc.)",
            "Descriptive notes or additional details about the wine's characteristics, production method, or story",
            "The primary grape or blend of grapes used to produce the wine (e.g., Merlot, Cabernet Sauvignon, etc.)",
            "Represents the kind of food that generally go with the wine",
            "Alcohol by Volume, representing the percentage of alcohol in the wine",
            "A measure of the weight or fullness of the wine on the palate (e.g., Light, Medium, Full)",
            "The level of acidity in the wine, impacting its freshness, tartness, and overall taste balance",
            "A code or identifier that may represent a batch, wine series, or classification",
            "The country of origin where the wine is produced",
            "A unique identifier for the wine's specific production region",
            "The name of the region where the wine is produced",
            "A unique identifier for the winery that produced the wine",
            "The name of the winery that produced the wine",
            "The website URL of the winery, where additional details about the wine may be found",
            "Information about the different vintages (years of production) available for the wine"
        ]
    })

    # Data description for XWines_Test_1K_ratings
    XWines_Test_1K_ratings = pd.DataFrame({
        "Column Name": ['RatingID', 'Vintage', 'Rating', 'Date'],
        "Description": [
            "A unique identifier for the specific rating entry in the dataset",
            "The year the wine was produced, which often indicates its quality and characteristics based on growing conditions",
            "A score given by a user or expert to indicate the quality or preference of the wine",
            "The date when the wine was rated, reviewed, or recorded in the dataset"
        ]
    })

    # Data description for Key_map
    Key_map = pd.DataFrame({
        "Column Name": ['RatingID', 'UserID', 'WineID'],
        "Description": [
            "A unique identifier for the specific rating entry in the dataset",
            "A unique identifier for the specific user entry in the dataset",
            "A unique identifier for the specific wine entry in the dataset"
        ]
    })

    # Conditional rendering based on the selected option
    if option == "XWines_Test_100_wines":
        st.markdown("##### **XWines_Test_100_wines**")
        st.table(XWines_Test_100_wines)
        st.image("pics/Heatmaps/XWines_Test_100_wines.png", caption="Heatmap of Missing Data", use_container_width=True)
    elif option == "XWines_Test_1K_ratings":
        st.markdown("##### **XWines_Test_1K_ratings**")
        st.table(XWines_Test_1K_ratings)
        st.image("pics/Heatmaps/XWines_Test_1K_ratings.png", caption="Heatmap of Missing Data", use_container_width=True)
    elif option == "Key_map":
        st.markdown("##### **Key_map**")
        st.table(Key_map)
        st.image("pics/Heatmaps/Key_map.png", caption="Heatmap of Missing Data", use_container_width=True)
    
    st.markdown(""" 
                ### Data Cleaning and Merging

                - Key_map: It has very few missing values, dropped the rows with missing values
                - XWines_Test_100_wines: Applied label encoder to the dataset and **imputed** the missing values using KNN  """)
    # Set up two columns for side-by-side heatmaps
    col1, col2 = st.columns(2)
    with col1:
        st.image("pics/Heatmaps/XWines_Test_100_wines_imputed.png", caption="KNN Imputed XWines_Test_100_wines", use_container_width=True)
    with col2:
        st.image("pics/Heatmaps/Key_map_imputed.png", caption="Null Dropped Key_map", use_container_width=True)

    st.markdown("""
    The following data was merged using the WineID and the Rating ID \n
                XWines_Test_100_wines_imputed  <---(**Wine ID**)--->  Key_map  <---(**RatingID**)--->  XWines_Test_1K_ratings
     """)
    

    # EDA page
elif options == "EDA":
    st.markdown(
        "<h1 style='color: red;'>Exploratory Data Analysis (EDA)</h1>", 
        unsafe_allow_html=True
    )
    st.markdown(
    """
    Let's explore the world of wine and gain some interesting insights on it.

    #### Wine Around the World:

    Hover over the map to see which wine country produces which kind of wine. 
    """
    )

    
    required_columns = ['Country', 'WineName', 'Rating']  

    # Aggregate data: Calculate the average rating and compile a list of wines for each country
    country_data = df_exp.groupby('Country').agg(
        Avg_Rating=('Rating', 'mean'),  
        Wine_List=('WineName', lambda x: ', '.join(x.unique()))  
    ).reset_index()

    # Check if the dataset has any data to display
    if country_data.empty:
        st.warning("No data available for visualization.")
    else:
        # Create a choropleth map
        fig = px.choropleth(
            country_data,
            locations='Country',
            locationmode='country names',  
            color='Avg_Rating',
            hover_name='Country',
            hover_data={
                'Avg_Rating': True,
                'Wine_List': True  
            },
            #title='World Map of Wine Ratings',
            color_continuous_scale='reds'
        )

        # Show the interactive map
        st.plotly_chart(fig)
        st.markdown("""#### Mutual Information between various categorical columns:""")
        st.image("pics/EDAs/Mutual_Information.png", caption="Mutual Informtion", use_container_width=True)
        st.markdown("""
        ##### Insights from the Mutual Information Plot

        - **WineName** shows a strong connection with **RegionName**, **Code**, and **Country**, indicating that wine names are heavily influenced by regional and country-specific factors.  
        - **Wine characteristics**, such as **Body** and **Type**, exhibit moderate relationships with **RegionName**, suggesting that regional factors play a significant role in shaping these attributes.  
        - **Acidity** and **Rating** have an almost negligible relationship, implying that acidity has minimal influence on wine ratings.
        """)

        st.markdown("""#### Data Distribution:""")
        # Define tabs
        tabs = st.tabs(["Type", "Country", "Region", "Acidity", "Code", "Alcohol Level", "Rating"])

        # Type of wine
        with tabs[0]:
           st.image("pics/EDAs/Distribution/Type.png", caption="Type of wine", use_container_width=True)
           st.markdown("Conclusion : Red wine is the most popular among all types of wine")

        # Country of Origin
        with tabs[1]:
            st.image("pics/EDAs/Distribution/Country.png", caption="Country of Origin", use_container_width=True)
            st.markdown("Conclusion : Most of the wine are from France, Italy US and Portugal ")

        # Region of Origin
        with tabs[2]:
            st.image("pics/EDAs/Distribution/RegionName.png", caption="Region of Origin", use_container_width=True)
        
        # Acidity Level
        with tabs[3]:
            st.image("pics/EDAs/Distribution/Acidity.png", caption="Acidity Level", use_container_width=True)
            st.markdown("Conclusion : Most wines have a medium level of acidity ")

        # Code
        with tabs[4]:
            st.image("pics/EDAs/Distribution/Code.png", caption="Code", use_container_width=True)

        # Alcohol Level
        with tabs[5]:
            st.image("pics/EDAs/Distribution/ABV.png", caption="Alcohol Level", use_container_width=True)
            st.markdown("Conclusion : Alcohol level of wine mostly lies between 10 to 20 percent ")

        # Rating
        with tabs[6]:
            st.image("pics/EDAs/Distribution/Rating.png", caption="Rating", use_container_width=True)
            st.markdown("Most of the rating lies between 3 to 4")

        
        
        st.markdown("""#### Outlier:""")
        # Define tabs
        tabs = st.tabs(["Type", "Country", "Region", "Acidity", "Code"])

        # Type of wine
        with tabs[0]:
           st.image("pics/EDAs/Box_Plot/Type.png", caption="Type of wine", use_container_width=True)
           st.markdown("""The boxplot shows the distribution of ratings for different wine types. 
                       Most wine types have a median rating between 3.5 and 4.0, indicating generally high ratings. 
                       The variability in ratings is similar for most types, but White and Rosé wines show slightly more variation. 
                       Some outliers, especially with low ratings, are present in all categories, suggesting a few wines were rated significantly lower than the rest. 
                       Overall, the ratings are concentrated around the higher end, with some differences in consistency across wine types.""")

        # Country of Origin
        with tabs[1]:
            st.image("pics/EDAs/Box_Plot/Country.png", caption="Country of Origin", use_container_width=True)
            st.markdown("""The boxplot shows the distribution of wine ratings by country. 
                        Most countries have median ratings between 3.5 and 4.0, indicating consistently good-quality wines across regions. 
                        Countries like Portugal, France, and New Zealand show higher medians and relatively narrower interquartile ranges, 
                        suggesting consistently high ratings with less variability. On the other hand, countries like South Africa and 
                        Canada exhibit broader ranges, indicating greater variability in wine quality. Outliers are present in most countries, 
                        representing a mix of exceptionally low or high-rated wines. Overall, while ratings are generally positive, 
                        the variability in wine quality differs significantly across countries.""")

        # Region of Origin
        with tabs[2]:
            st.image("pics/EDAs/Box_Plot/RegionName.png", caption="Region of Origin", use_container_width=True)
            st.markdown("""The boxplot shows wine ratings across different regions. Regions like Vale dos Vinhedos and Sauternes 
                        have high median ratings and relatively narrow interquartile ranges, indicating consistent and high-quality wines. 
                        In contrast, regions like Serra Gaúcha and Langhe exhibit wider ranges and lower medians, suggesting greater variability 
                        in wine quality. Some regions, such as Langhe and Mendoza, have noticeable outliers with very low ratings, reflecting 
                        occasional poor-quality wines. Overall, while certain regions demonstrate consistently high-quality ratings, others 
                        show a wider spectrum of wine quality, with some outliers significantly affecting their overall distribution.""")
        
        # Acidity Level
        with tabs[3]:
            st.image("pics/EDAs/Box_Plot/Acidity.png", caption="Acidity Level", use_container_width=True)
            st.markdown(""" The boxplot illustrates wine ratings across different acidity levels. 
                        Wines with medium and high acidity tend to receive higher ratings compared to those with low acidity. 
                        The variability in ratings is similar across all acidity levels, indicating consistent quality within each
                        category. However, some wines, particularly those with medium and high acidity, have outliers with 
                        significantly lower ratings, suggesting a few underperforming wines. Overall, ratings are predominantly 
                        concentrated at the higher end, highlighting a general preference for wines with moderate to high acidity.""")

        # Code
        with tabs[4]:
            st.image("pics/EDAs/Box_Plot/Code.png", caption="Code", use_container_width=True)


        # Streamlit app
        st.markdown(" #### Top Rated Wines")

        # Arrange the first three filters (Country, Type, Acidity) in a single row
        col1, col2, col3 = st.columns(3)

        with col1:
            country = st.selectbox(
                "Country",
                options=["All"] + sorted(df_exp["Country"].dropna().unique()),
                index=0
            )

        with col2:
            wine_type = st.selectbox(
                "Type",
                options=["All"] + sorted(df_exp["Type"].dropna().unique()),
                index=0
            )

        with col3:
            acidity = st.selectbox(
                "Acidity",
                options=["All"] + sorted(df_exp["Acidity"].dropna().unique()),
                index=0
            )

        # Place the ABV range slider below the row
        abv_range = st.slider(
            "ABV Range (%)",
            float(df_exp["ABV"].min()),
            float(df_exp["ABV"].max()),
            (float(df_exp["ABV"].min()), float(df_exp["ABV"].max()))
        )

        # Filter data based on user input
        filtered_data = df_exp.copy()

        if country != "All":
            filtered_data = filtered_data[filtered_data["Country"] == country]

        if wine_type != "All":
            filtered_data = filtered_data[filtered_data["Type"] == wine_type]

        if acidity != "All":
            filtered_data = filtered_data[filtered_data["Acidity"] == acidity]

        filtered_data = filtered_data[
            (filtered_data["ABV"] >= abv_range[0]) & (filtered_data["ABV"] <= abv_range[1])
        ]

        # Sort by rating and get the top 5
        top_wines = filtered_data.sort_values(by="Rating", ascending=False).head(5)

        # Plot bar chart
        if not top_wines.empty:
            plt.figure(figsize=(10, 6))
            plt.bar(top_wines["WineName"], top_wines["Rating"], color="red")
            plt.xlabel("Wine Name")
            plt.ylabel("Rating")
            plt.title("Top Rated Wines")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(plt.gcf())  # Use plt.gcf() to render the current figure
        else:
            st.write("No wines match the selected criteria.")

        # Feature Engineering
elif options == "Feature Engineering":
    st.markdown(
        "<h1 style='color: red;'>Feature Engineering</h1>", 
        unsafe_allow_html=True
    )
    # User Rating
    st.markdown("""#### User Rating Distribution:""")
    st.image("pics/Feature_Engineering/User_Rating.png", caption="User Rating", use_container_width=True)
    st.markdown("""Different users have diverse taste preferences when it comes to wine, making it essential to adjust and scale ratings based on individual user preferences. 
                This approach ensures that the ratings reflect a more personalized and accurate representation of each user's unique palate. """)
    
    # Word Cloud
    st.markdown("""#### Word Cloud:""")
    st.markdown("""##### Grapes:""")
    st.image("pics/Feature_Engineering/Word_Cloud_Grapes.png", caption="Word Cloud Grapes", use_container_width=True)
    st.markdown("""Grapes are the fundamental ingredient in wine production, making it crucial to analyze their characteristics for providing accurate wine recommendations.""")
    st.markdown("""##### Food:""")
    st.image("pics/Feature_Engineering/Word_Cloud_Harmonize.png", caption="Word Cloud Food", use_container_width=True)
    st.markdown("""Wine is often enjoyed alongside food, and selecting the right wine to complement a specific dish is essential for enhancing the overall dining experience.""")

    # Engineered Feature
    st.markdown("""#### Features Created:""")
    Feature_Engineering = pd.DataFrame({
        "Column Name": ['Normalised_User_Rating', 'Normalised_Wine_Rating', 'Grape_Count', 'Harmonize_Count', 'Grapes', 'Harmonize' ],
        "Description": [
            "User wise normalized rating",
            "Wine wise normalized rating",
            "Count of total grapes used in the wine",
            "Count of total food item that can harmonize with the wine",
            "Explode the grape list into different columns putting a flag 1 if the grape is present in the wine",
            "Explode the Harmonize list into different columns putting a flag 1 if the item goes with the wine"
        ]
    })
    st.table(Feature_Engineering)

    # PCA
    st.markdown("""#### PCA:""")
    st.image("pics/Feature_Engineering/Explained_Variance_vs_Components.png", caption="Explained Variance vs Components", use_container_width=True)
    st.image("pics/Feature_Engineering/Scree_Plot.png", caption="Scree Plot", use_container_width=True)
    st.markdown("""Number of principal component considered: 70 """)

    # Modeling
elif options == "Modeling":
    st.markdown(
        "<h1 style='color: red;'>Modeling</h1>", 
        unsafe_allow_html=True
    )
    # PCA
    st.markdown("""#### Final Modeling Dataset:""")
    st.dataframe(pd.read_csv('Data/Final_Model.csv').iloc[:, 1:], height=400)
    st.markdown("""Selected top 20 important features using Recursive Feature Elimination method """)
    
    # Selecting no of clusters
    st.markdown("""#### Selecting no of clusters:""")
    st.image("pics/Model/Elbow.png", caption="Elbow Plot", use_container_width=True)
    st.markdown("""Number of clusters we are working with is 4 """)

    # Models Applied
    st.markdown("""#### Models Applied:""")

    # Validation
    st.dataframe(validation)

    st.markdown("""##### Silhouette Scores Comparison""")
    st.write("This bar plot compares the Silhouette Scores for different clustering models.")
    st.image("pics/Model/silhoutte_Score.png", caption="Silhoutte_Score Plot", use_container_width=True)

    st.markdown("""##### DB Index Comparison""")
    st.write("This bar plot compares the DB Index for different clustering models.")
    st.image("pics/Model/DB Index.png", caption="DB Index Plot", use_container_width=True)

    # Selection
    st.markdown("""
         ### Recommendation:
    Based on the **Silhouette Score**, **KNN (Euclidean)** or **KNN (Minkowski)** are promising models for a wine recommendation system.

    However, **K-Means (Euclidean)** achieves a significantly better **DB Index**, indicating more compact clusters, and has the added benefit of lower computational cost compared to KNN.

    #### Final Choice:
    - **K-Means (Euclidean)** is the best choice for wine recommendations due to its:
    1. Low **DB Index** (best among all).
    2. Acceptable **Silhouette Score** (though not the highest).
    3. Lower computational cost compared to KNN, making it efficient for larger datasets.
    """)

    # Selection
    st.markdown("""
         ### Kmeans Performance:""")
    st.markdown("#####Correspondence Analysis")
    st.image("pics/Model/Corrospondence Analysis.png", caption="Corrospondence Analysis", use_container_width=True)
    st.markdown("""
         In the given plot, Correspondence Analysis effectively separates the data into four distinct clusters, 
                demonstrating the variables' strong discriminative power to identify meaningful groupings. 
                The green and yellow clusters appear compact, indicating that the data points within these 
                groups are closely related and exhibit similar features. In contrast, the purple cluster 
                is more dispersed, suggesting greater internal variation among its points. The blue cluster, 
                concentrated near the center, indicates moderate variation and a closer relationship to the dataset's 
                central tendencies. Overall, the plot visually captures the underlying patterns and associations within 
                the data, highlighting both the similarities and differences across the identified clusters.
    """)

    st.markdown("##### T-SNE Analysis")
    st.image("pics/Model/T-SNE.png", caption="T-SNE Analysis", use_container_width=True)
    st.markdown("""
         The t-SNE Visualization plot displays the distribution of data points across two components, revealing four distinct clusters. The clusters, differentiated by color, suggest that the data has been effectively grouped based on underlying similarities:

        - **The purple cluster**: Positioned in the lower-left region and appears compact, indicating closely related data points.  
        - **The blue cluster**: Located in the upper region and shows moderate spread, reflecting slight variation within the group.  
        - **The green cluster**: Concentrated on the right side and displays a well-defined structure.  
        - **The yellow cluster**: Positioned in the lower-right region and also appears compact with a clear separation from the other groups.  

        Overall, the plot highlights strong clustering patterns, indicating that the variables used for clustering provide meaningful distinctions among the data points.
     """)


    # Recomendation
elif options == "Recomendation":
    st.markdown(
        "<h1 style='color: red;'>Recomendation</h1>", 
        unsafe_allow_html=True
    )
    st.markdown("""Select your most prefered wine from the list to get your recomendation""")

    # Streamlit dropdown for wine selection
    wine_names = kmeans_data['Winename'].tolist()
    selected_wine = st.selectbox("Select a Wine", wine_names, index=wine_names.index('Origem Merlot'))

    # Load Model
    kmeans_model = joblib.load('Model/kmeans_model.pkl')
    X = kmeans_data.drop(['Winename'], axis=1)  # Features
    y = kmeans_data['Winename'] 
    recommendations_kmeans = recommend_wines_kmeans(X,selected_wine)
    
    # Display selected wine details
    st.markdown("""### Your Choice:""")
    display_wine(wine_details, selected_wine)

    
    # Display recommendations in Streamlit
    st.markdown(f"### Recommendations for {selected_wine}:")
    for wine in recommendations_kmeans:
        display_wine(wine_details, wine)







# Footer
st.sidebar.markdown("---")
st.sidebar.write("Created by Roshni Bhowmik")