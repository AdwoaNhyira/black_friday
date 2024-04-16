import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import requests
from bs4 import BeautifulSoup
from PIL import Image
from wordcloud import STOPWORDS

# load data
sentiment_analysis = pd.read_csv('data/SentimentAnalysis.csv')

# unique product categories
product_categories = sentiment_analysis['product_category_name'].unique()

# main filter
filter_option = st.sidebar.selectbox("Filter by:", ["Best Reviews", "Product Category"])

if filter_option == "Best Reviews":
    # product categories in order from best reviewed to least reviewed
    st.subheader("Select Product Category:")
    sorted_categories = sentiment_analysis.groupby('product_category_name')['average_review_score'].mean().sort_values(ascending=False).index
    selected_category = st.selectbox("Select Product Category:", sorted_categories)

    # filter data for selected product category
    filtered_data = sentiment_analysis[sentiment_analysis['product_category_name'] == selected_category]

    # display picture, word cloud, line chart, sentiment count, and average sentiment for the selected category
    st.subheader(f"{selected_category}")

    # get the  first Google image of selected category
    query = f"{selected_category} product"
    search_url = f"https://www.google.com/search?q={query}&tbm=isch"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    img_tags = soup.find_all('img')
    img_url = img_tags[1]['src']

    # display the image
    image = Image.open(requests.get(img_url, stream=True).raw)
    st.image(image, caption=f"First Google Image for '{selected_category}' Product")

    # display wordcloud for selected product category
    st.subheader("Word Cloud of Review Comments")
    wordcloud_text = ' '.join(filtered_data['review_comment_message_en'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wordcloud_text)
    st.image(wordcloud.to_array(), caption="Word Cloud", use_column_width=True)

    # display line chart of product rating distribution
    st.subheader(f"Line Chart of Product Rating Distribution for {selected_category}")
    rating_counts = filtered_data['review_score'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rating_counts.index, rating_counts.values, marker='o')
    ax.set_xlabel('Rating')
    ax.set_ylabel('Count')
    ax.set_title(f'Product Rating Distribution for {selected_category}')
    st.pyplot(fig)

    # count sentiment for each product category
    sentiment_counts_per_category = filtered_data.groupby('product_category_name')['sentiment'].value_counts().unstack(
        fill_value=0)

    # bar chart of sentiment count
    st.subheader("Sentiment Count for Selected Product Categories")
    fig_bar, ax_bar = plt.subplots(figsize=(10, 6))

    # define colors for positive, neutral, and negative sentiments
    colors = {'positive': 'green', 'neutral': 'blue', 'negative': 'red'}

    # plot bars for each sentiment with respective colors
    for sentiment in sentiment_counts_per_category.columns:
        ax_bar.bar(sentiment_counts_per_category.index, sentiment_counts_per_category[sentiment],
                   color=colors[sentiment], label=sentiment)

    ax_bar.set_xlabel('Product Category')
    ax_bar.set_ylabel('Count')
    ax_bar.set_title('Sentiment Count Distribution for Selected Product Categories')
    ax_bar.legend()
    fig_bar.tight_layout()

    # show plot in dashboard
    st.pyplot(fig_bar)

    # display average sentiment for selected product category
    st.subheader("Average Sentiment")
    average_sentiment_text = filtered_data['average_sentiment'].iloc[0]

    # set color based on sentiment
    color = "green" if "positive" in average_sentiment_text.lower() else (
        "red" if "negative" in average_sentiment_text.lower() else "blue")

    # format as bold and use color codes
    st.markdown(f"<font color='{color}' size='5'><b>{average_sentiment_text}</b></font>", unsafe_allow_html=True)

else:
    # filter by product category
    st.sidebar.header("Select Product Category")

    # buttons for "Select All" and "See data for all Categories"
    select_all = st.sidebar.button("Select All")
    unselect_all = st.sidebar.button("See data for all Categories")

    # filter data based on selected product categories
    selected_product_categories = []
    if select_all:
        selected_product_categories = list(product_categories)
    elif unselect_all:
        selected_product_categories = []
    else:
        selected_category = st.sidebar.selectbox("Select Product Category:", product_categories)
        selected_product_categories.append(selected_category)

    if selected_product_categories:
        # filtered data for selected product categories
        filtered_data = sentiment_analysis[sentiment_analysis['product_category_name'].isin(selected_product_categories)]

        for category in selected_product_categories:
            st.subheader(f"{category}")

            # get the URL of the first Google image
            query = f"{category} product"
            search_url = f"https://www.google.com/search?q={query}&tbm=isch"
            response = requests.get(search_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            img_tags = soup.find_all('img')
            img_url = img_tags[1]['src']

            # display the image
            image = Image.open(requests.get(img_url, stream=True).raw)
            st.image(image, caption=f"First Google Image for '{category}' Product")

    else:
        # use entire dataset if no product category is selected
        filtered_data = sentiment_analysis

    # stopwords to make our wordcloud more accurate
    custom_stopwords = set(STOPWORDS)
    custom_stopwords.update(["and", "or", "if", "I", "it", "the", "a"])

    # function to preprocess text and remove stopwords
    def preprocess_text(text):
        # convert text to lowercase
        text = text.lower()
        # split text into words
        words = text.split()
        # remove stopwords
        words = [word for word in words if word not in custom_stopwords]
        # join words back into a single string
        cleaned_text = ' '.join(words)
        return cleaned_text

    # display word cloud for all products
    st.subheader("Word Cloud of Review Comments for All Products")
    wordcloud_text_all = ' '.join(filtered_data['review_comment_message_en'].dropna().apply(preprocess_text))
    wordcloud_all = WordCloud(width=800, height=400, background_color='white', stopwords=custom_stopwords).generate(wordcloud_text_all)
    fig_all, ax_all = plt.subplots()
    ax_all.imshow(wordcloud_all, interpolation='bilinear')
    ax_all.axis('off')
    st.pyplot(fig_all)

    # display line chart of product rating distribution
    st.subheader(f"Line Chart of Product Rating Distribution for Selected Product Categories")
    rating_counts = filtered_data['review_score'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rating_counts.index, rating_counts.values, marker='o')
    ax.set_xlabel('Rating')
    ax.set_ylabel('Count')
    ax.set_title(f'Product Rating Distribution for Selected Product Categories')
    fig.tight_layout()
    st.pyplot(fig)

    # count sentiment for each product category
    sentiment_counts_per_category = filtered_data.groupby('product_category_name')['sentiment'].value_counts().unstack(
        fill_value=0)

    # bar chart of sentiment count
    st.subheader("Sentiment Count for Selected Product Categories")
    fig_bar, ax_bar = plt.subplots(figsize=(10, 6))

    # define colors for positive, neutral, and negative sentiments
    colors = {'positive': 'green', 'neutral': 'blue', 'negative': 'red'}

    # plot bars for each sentiment with respective colors
    for sentiment in sentiment_counts_per_category.columns:
        ax_bar.bar(sentiment_counts_per_category.index, sentiment_counts_per_category[sentiment],
                   color=colors[sentiment], label=sentiment)

    ax_bar.set_xlabel('Product Category')
    ax_bar.set_ylabel('Count')
    ax_bar.set_title('Sentiment Count Distribution for Selected Product Categories')
    ax_bar.legend()
    fig_bar.tight_layout()

    # show plot in Streamlit
    st.pyplot(fig_bar)

    # display average sentiment for selected product category
    st.subheader("Average Sentiment")
    average_sentiment_text = filtered_data['average_sentiment'].iloc[0]

    # set color based on sentiment
    color = "green" if "positive" in average_sentiment_text.lower() else (
        "red" if "negative" in average_sentiment_text.lower() else "blue")

    # format as bold and use HTML color codes
    st.markdown(f"<font color='{color}' size='5'><b>{average_sentiment_text}</b></font>", unsafe_allow_html=True)
