import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#from scipy.optimize import curve_fit
from wordcloud import WordCloud
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load data (CSV)
def load_data():
    try:
        # Assuming 'historical_data.csv' exists and contains the required columns
        historical_df = pd.read_csv('historical_data.csv')
        return historical_df
    except FileNotFoundError:
        st.error("The file 'historical_data.csv' was not found.")
        return None

# Task 1: Task-specific content for task 1
def task1(historical_df):
    st.subheader("Task 1: Average Based Analysis")
     # Loading data
    historical_df = load_data()
    if historical_df is None:
        return

    # Check if required columns exist in the data
    required_columns = ['quiz.topic', 'score', 'accuracy', 'speed', 'final_score', 'total_questions']
    if not all(col in historical_df.columns for col in required_columns):
        st.error("CSV file is missing required columns.")
        return

    # Grouping by topic and calculating the mean of numerical columns
    topic_performance = historical_df.groupby('quiz.topic').agg({
        'score': 'mean',
        'accuracy': 'mean',
        'speed': 'mean',
        'final_score': 'mean',
        'total_questions': 'mean'
    }).reset_index()

    # Sorting data by accuracy
    topic_performance_sorted = topic_performance.sort_values(by='accuracy', ascending=False)

    # Plotting the graphs
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    sns.barplot(x='score', y='quiz.topic', data=topic_performance_sorted, ax=axes[0], palette='Blues_d')
    axes[0].set_title('Average Score by Topic')

    sns.barplot(x='accuracy', y='quiz.topic', data=topic_performance_sorted, ax=axes[1], palette='Greens_d')
    axes[1].set_title('Average Accuracy by Topic')

    sns.barplot(x='speed', y='quiz.topic', data=topic_performance_sorted, ax=axes[2], palette='Reds_d')
    axes[2].set_title('Average Speed by Topic')

    plt.tight_layout()
    st.pyplot(fig)

    st.write("Top 2 Topics with Highest Accuracy:")
    st.write(topic_performance_sorted.head(2))

    st.write("Bottom 2 Topics with Lowest Accuracy:")
    st.write(topic_performance_sorted.tail(2))
    # You can add your logic for task 1 here
    st.write("End of page.")


# Task 2: Task-specific content for task 2 (Weak Areas and Improvement Trends)
def task2(historical_df):
    st.subheader("Task 2: Weak Areas and Improvement Trends")
    
    # Weak Areas (Accuracy < 60%)
    weak_areas = historical_df[historical_df['accuracy'] < 60]

    # Plotting Weak Areas
    plt.figure(figsize=(10, 6))
    plt.barh(weak_areas['quiz.topic'], weak_areas['accuracy'], color='red')
    plt.xlabel('Accuracy')
    plt.title('Weak Areas (Accuracy < 60%)')
    plt.gca().invert_yaxis()
    st.pyplot(plt)

    # Task 2.2: Improvement Trends
    def exp_func(x, a, b, c):
        return a * np.exp(b * x) + c

    # Plotting Improvement Trends
    fig, ax = plt.subplots(figsize=(12, 6))

    topics = historical_df['quiz.topic'].unique()
    for topic in topics:
        topic_data = historical_df[historical_df['quiz.topic'] == topic].reset_index()
        x_data = np.arange(len(topic_data))
        y_data = topic_data['accuracy']

        if len(x_data) > 3:
            try:
                popt, _ = curve_fit(exp_func, x_data, y_data, p0=(1, 0.01, 1))
                ax.plot(topic_data['quiz.time'], exp_func(x_data, *popt), linestyle='--', label=f'{topic} (Fit)')
            except RuntimeError:
                print(f"Could not fit exponential curve for topic: {topic}")

        ax.plot(topic_data['quiz.time'], y_data, marker='o', label=f'{topic} (Data)')

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title('Improvement Trends (Accuracy Over Time for Each Topic)', fontsize=14)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    st.write("End of page.")

# Task 3: Recommended vs Alternative Topics (K-Means Clustering)
def task3(historical_df):
    st.subheader("Task 3: Recommended vs Alternative Topics for Improvement")
    
    # Step 1: Remove duplicates based on topic
    historical_df['quiz.topic'] = historical_df['quiz.topic'].str.strip()  # Strip spaces if any
    unique_topics_df = historical_df.drop_duplicates(subset=['quiz.topic'])
    
    # Step 2: Group by 'quiz.topic' and calculate mean score and accuracy
    topic_performance = unique_topics_df.groupby('quiz.topic')[['score', 'accuracy']].mean().reset_index()
    
    # Step 3: Normalize the data for clustering
    scaler = StandardScaler()
    topic_features = scaler.fit_transform(topic_performance[['score', 'accuracy']])
    
    # Step 4: Apply K-Means Clustering
    kmeans = KMeans(n_clusters=4, random_state=42)  # Change n_clusters if needed
    topic_performance['cluster'] = kmeans.fit_predict(topic_features)
    
    # Step 5: Add cluster labels for better understanding
    st.write("Cluster Centers (Scaled):")
    st.write(pd.DataFrame(kmeans.cluster_centers_, columns=['Score', 'Accuracy']))
    
    # Step 6: Visualize Clusters
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x='score', y='accuracy', hue='cluster', data=topic_performance, palette='viridis', s=100, ax=ax
    )
    ax.set_title('K-Means Clustering of Topics Based on Score and Accuracy')
    ax.set_xlabel('Score')
    ax.set_ylabel('Accuracy')
    st.pyplot(fig)
    
    # Step 7: Recommendations
    st.write("### Recommendations:")
    for cluster_id in sorted(topic_performance['cluster'].unique()):
        st.write(f"**Cluster {cluster_id}** Topics: ")
        cluster_topics = topic_performance[topic_performance['cluster'] == cluster_id]['quiz.topic'].tolist()
        st.write(", ".join(cluster_topics))

    # Step 8: Highlight Improvement Areas
    lowest_accuracy_cluster_id = topic_performance.groupby('cluster')['accuracy'].mean().idxmin()
    improvement_topics = topic_performance[topic_performance['cluster'] == lowest_accuracy_cluster_id]

    # Ensure topics are displayed properly
    improvement_topics_list = improvement_topics['quiz.topic'].tolist()
    st.write("### Focus on improving these topics:")
    st.write(", ".join(improvement_topics_list))


    st.write("End of page.")

# Generate insights based on dynamic thresholds (high/low performance)
def generate_insights_dynamic(topic_performance, high_threshold, low_threshold):
    # High-performing topics
    high_performers = topic_performance[topic_performance >= high_threshold]
    # Low-performing topics
    low_performers = topic_performance[topic_performance <= low_threshold]
    
    # Strengths and recommendations
    strength_msg = "Strengths (High Performance):"
    if len(high_performers) > 0:
        strength_msg += f"\n- Topics: {', '.join(high_performers.index)}"
        strength_msg += "\n  Recommendation: Reinforce your knowledge in these topics. Continue practicing to maintain your high performance."
    else:
        strength_msg += "\nNo strong topics identified. Focus on improving various topics."
    
    # Weaknesses and recommendations
    weakness_msg = "\nWeaknesses (Low Performance):"
    if len(low_performers) > 0:
        weakness_msg += f"\n- Topics: {', '.join(low_performers.index)}"
        weakness_msg += "\n  Recommendation: Focus on improving these topics. Consider reviewing the foundational concepts and doing extra practice."
    else:
        weakness_msg += "\nNo weak topics identified. Keep up the great work!"
    
    return strength_msg, weakness_msg

# Create word cloud for low-performing topics
def create_wordcloud(low_performance_topics):
    wordcloud_data = low_performance_topics.to_dict()  # Convert to dictionary (topic: accuracy)
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate_from_frequencies(wordcloud_data)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Hide axes
    plt.title('WordCloud of Low-Performing Topics (Focus Areas)', fontsize=16)
    plt.tight_layout()
    st.pyplot(plt)

# Student Persona Section with WordCloud for weak topics
def student_persona(historical_df):
    st.subheader("Student Persona: Focus Areas (WordCloud)")

    # Calculate percentiles for dynamic thresholds
    high_threshold = historical_df['accuracy'].quantile(0.75)  # 75th percentile
    low_threshold = historical_df['accuracy'].quantile(0.25)   # 25th percentile

    # Calculate average accuracy per topic for all students
    topic_performance = historical_df.groupby('quiz.topic')['accuracy'].mean()

    # Categorize topics based on the calculated thresholds
    low_performance_topics = topic_performance[topic_performance <= low_threshold]

    # Generate insights based on performance
    strength_msg, weakness_msg = generate_insights_dynamic(topic_performance, high_threshold, low_threshold)

    # Display insights
    st.write(strength_msg)
    st.write(weakness_msg)

    # Create and display word cloud for low-performing topics
    create_wordcloud(low_performance_topics)
    st.write("End of page.")

# Main function to display everything in the Streamlit app
def main():
    st.title("Student Performance Dashboard")

    # Load data
    historical_df = load_data()
    if historical_df is None:
        return
    
    # Check if required columns exist in the data
    required_columns = ['quiz.topic', 'accuracy']
    if not all(col in historical_df.columns for col in required_columns):
        st.error("CSV file is missing required columns.")
        return
    
    # Sidebar with clickable buttons for Task 1, Task 2, Task 3, and Student Persona
    task = st.sidebar.selectbox("Select from dropdown", ["Welcome", "Averages on various aspects", "Weak Areas & Improvement Trends", "K-means based recommendations", "Student Persona"])

    if task == "Welcome":
        st.write("Welcome to the Student Performance Dashboard! Select a task from the sidebar to explore the performance details.")
        # Add a cartoon image (make sure to replace 'cartoon.png' with the correct path to your image)
        st.image('cartoon.png', caption='Welcome!', use_column_width=True)
    elif task == "Averages on various aspects":
        task1(historical_df)
    elif task == "Weak Areas & Improvement Trends":
        task2(historical_df)
    elif task == "K-means based recommendations":
        task3(historical_df)
    elif task == "Student Persona":
        student_persona(historical_df)

if __name__ == "__main__":
    main()
