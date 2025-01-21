import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity

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

    st.write("Top 5 Topics with Highest Accuracy:")
    st.write(topic_performance_sorted.head())

    st.write("Bottom 5 Topics with Lowest Accuracy:")
    st.write(topic_performance_sorted.tail())
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

# Task 3: Recommended vs Alternative Topics (Cosine Similarity)
def task3(historical_df):
    st.subheader("Task 3: Recommended vs Alternative Topics for Improvement")

   # Step 1: Remove duplicates based on topic and accuracy
    historical_df['quiz.topic'] = historical_df['quiz.topic'].str.strip()  # Strip spaces if any
    unique_topics_df = historical_df.drop_duplicates(subset=['quiz.topic'])

    # Step 2: Group by 'quiz.topic' and calculate mean accuracy
    topic_performance = unique_topics_df.groupby('quiz.topic')['accuracy'].mean().reset_index()

    # Step 3: Extract topics and their average accuracy
    topics = topic_performance['quiz.topic'].values

    # Step 4: Create feature matrix based on accuracy
    topic_features = topic_performance['accuracy'].values.reshape(-1, 1)  # Using accuracy for simplicity

    # Step 5: Calculate cosine similarity between topics
    cos_sim = cosine_similarity(topic_features)

    # Step 6: Select a student (example: student 0) for the similarity comparison
    student_index = 0
    similarities = cos_sim[student_index]  # Cosine similarity of student 0 with all topics

    # Step 7: Sort topics by their cosine similarity (highest first)
    similar_topics_indices = np.argsort(similarities)[::-1]  # Sort by most similar (highest first)

    # Step 8: Identify top 3 similar topics (good topics) and the bottom 3 (focus areas)
    main_topics = [topics[i] for i in similar_topics_indices[:4]]  # Top 3 most similar topics (good topics)
    alternative_topics = [topics[i] for i in similar_topics_indices[-3:]]  # Bottom 3 (need more focus)

    # Step 9: Scores based on cosine similarity (higher is better)
    main_scores = similarities[similar_topics_indices[:4]]
    alt_scores = similarities[similar_topics_indices[-3:]]

    # Step 10: Bar Chart plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Main Topics (Good Topics - Higher Similarity)
    ax.barh(main_topics, main_scores, color='salmon', label="Main Topics (Good)")

    # Alternative Topics (Topics for Improvement - Lower Similarity)
    ax.barh(alternative_topics, alt_scores, color='lightgreen', label="Alternative Topics (Improvement)")

    # Step 11: Add labels and title
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Topics')
    ax.set_title('Recommended vs Alternative Topics for Improvement')
    ax.legend(loc='upper left')

    # Step 12: Display Chart
    plt.tight_layout()
    st.pyplot(fig)
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
    task = st.sidebar.selectbox("Select from dropdown", ["Welcome", "Averages on various aspects", "Weak Areas & Improvement Trends", "Cosine Similarity based recommendations", "Student Persona"])

    if task == "Welcome":
        st.write("Welcome to the Student Performance Dashboard! Select a task from the sidebar to explore the performance details.")
        # Add a cartoon image (make sure to replace 'cartoon.png' with the correct path to your image)
        st.image('cartoon.png', caption='Welcome!', use_column_width=True)
    elif task == "Averages on various aspects":
        task1(historical_df)
    elif task == "Weak Areas & Improvement Trends":
        task2(historical_df)
    elif task == "Cosine Similarity based recommendations":
        task3(historical_df)
    elif task == "Student Persona":
        student_persona(historical_df)

if __name__ == "__main__":
    main()
