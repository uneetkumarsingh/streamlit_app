# streamlit_app
This app takes only docx files and predicts if it belongs to either of the four categories:
1. Research Paper
2. Abstract
3. Response to Reviewer Comments 
4. Others

It also gives probability for each of the three categories(1-3) and an over all probability. 

Bug:
While deploying there were some issue linked to tkinter. Same was resolved using following thread
https://discuss.streamlit.io/t/tkinter-import-failed-during-app-deployment/23529
We added a packages.txt file for this purpose. 
