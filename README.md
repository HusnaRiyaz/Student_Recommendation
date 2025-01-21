# Student_Recommendation_System(General Analysis)

## Project Description

  Develop a Python-based solution to analyze quiz performance and provide students with personalized recommendations to improve their preparation.
  
## Tools Used

- Python
- Streamlit
- Pandas
- Other libraries or tools

## Steps:
1) Extracted the data from the given links below:
   quiz_endpoint = https://www.jsonkeeper.com/b/LLQT
   submission_data = https://api.jsonserve.com/rJvd7g
   historical_data = https://api.jsonserve.com/XgAgFJ
2) Preprocess the data(data cleaning)
   1) Covert the json data to Pandas dataframe
   2) Remove the duplicates, irrelavent and duplicate columns  
3) Analyse the data and find student performances by topics, accuracy etc and visualize with graphs to understand better
4) Highlight the weak topics and the performance gaps for the given student.
5) Recommend topics to be focused more to improve based on cosine similarity
6) Create a student persona highlighting the strong and weak topics, and generating wordcloud to highlight the topics to be focused more and suggestions to improve.
7) Finally used Streamlit app to showcase as student dashboard highlighting all the above mentioned aspects one by one.

## Results:
1) The student performed well in the below topics:
   1) Human Health and Disease
   2) Microbes in Human Welfare
  ## Suggestions:
    Reinforce your knowledge in these topics. Continue practicing to maintain your high performance.
    
2) The analysis reveals that the areas where the student needs to improve are:
   1) Reproducive Health
   2) Human Reproduction
   3) Principles of Inheritance and Variation
  ## Suggestions:
    Focus on improving these topics. Consider reviewing the foundational concepts and doing extra practice.
  
