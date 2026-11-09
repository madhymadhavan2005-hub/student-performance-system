import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# 1. Page Configuration
st.set_page_config(
    page_title='Student Performance & Decision Support System',
    page_icon='🎓',
    layout='wide',
)

st.title(
    '🎓 Student Academic Performance Prediction & Early Intervention System'
)
st.markdown(
    'A comprehensive analytics and decision support dashboard powered by'
    ' Machine Learning and interactive visualizations.'
)
st.markdown('---')


# 2. Load Dataset
@st.cache_data
def load_data():
  try:
    df = pd.read_csv('large_student_performance_dataset.csv')
    return df
  except FileNotFoundError:
    st.error(
        '🚨 Dataset "large_student_performance_dataset.csv" not found in the'
        ' directory! Please generate and save it first.'
    )
    return None


df = load_data()

if df is not None:
  # Define Features and Target
  feature_cols = [
      'age',
      'study_hours',
      'attendance_rate',
      'previous_gpa',
      'midterm_score',
      'assignments_completed',
  ]
  target_col = 'passed'

  X = df[feature_cols]
  y = df[target_col]

  # Train-Test Split & Model Training
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.25, random_state=42
  )

  model = RandomForestClassifier(n_estimators=100, random_state=42)
  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)

  # Setup Tabs for Navigation
  tab1, tab2, tab3, tab4 = st.tabs([
      '📊 Dataset Overview',
      '📈 Visual Analytics Hub (10+ Graphs)',
      '🤖 Model Performance',
      '🎯 Student Prediction & Intervention',
  ])

  # --- TAB 1: OVERVIEW ---
  with tab1:
    st.header('Dataset Summary & Key Metrics')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
      st.metric(label='Total Students', value=len(df))
    with col2:
      st.metric(
          label='Overall Pass Rate',
          value=f'{(df[target_col].mean() * 100):.1f}%',
      )
    with col3:
      st.metric(
          label='Avg Study Hours / Week',
          value=f'{df["study_hours"].mean():.1f} hrs',
      )
    with col4:
      st.metric(
          label='Avg Attendance Rate',
          value=f'{df["attendance_rate"].mean():.1f}%',
      )

    st.markdown('### 📋 Raw Data Preview (First 10 Records)')
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown('### 📊 Statistical Summary')
    st.dataframe(df.describe(), use_container_width=True)

  # --- TAB 2: VISUAL ANALYTICS HUB (10+ GRAPHS) ---
  with tab2:
    st.header('Interactive Visual Analytics Hub (10 Graphs)')
    st.markdown(
        'Explore patterns, distributions, and feature relationships across your'
        ' student cohort.'
    )

    # Map status labels for cleaner visuals
    df['status_label'] = df['passed'].map(
        {1: 'Safe / On Track', 0: 'At-Risk'}
    )

    # Row 1: Graphs 1 & 2
    c1, c2 = st.columns(2)
    with c1:
      st.subheader('1. Study Hours vs. Mid-term Score')
      fig1 = px.scatter(
          df,
          x='study_hours',
          y='midterm_score',
          color='status_label',
          title='Study Hours vs Mid-term Score by Risk Status',
          color_discrete_map={
              'Safe / On Track': '#2ecc71',
              'At-Risk': '#e74c3c',
          },
      )
      st.plotly_chart(fig1, use_container_width=True)

    with c2:
      st.subheader('2. Attendance Rate Distribution')
      fig2 = px.box(
          df,
          x='status_label',
          y='attendance_rate',
          color='status_label',
          title='Attendance Rate Spread across Performance Status',
          color_discrete_map={
              'Safe / On Track': '#2ecc71',
              'At-Risk': '#e74c3c',
          },
      )
      st.plotly_chart(fig2, use_container_width=True)

    # Row 2: Graphs 3 & 4
    c3, c4 = st.columns(2)
    with c3:
      st.subheader('3. Previous GPA Distribution')
      fig3 = px.histogram(
          df,
          x='previous_gpa',
          color='status_label',
          marginal='violin',
          title='Distribution of Previous Semester GPA',
          color_discrete_map={
              'Safe / On Track': '#2ecc71',
              'At-Risk': '#e74c3c',
          },
      )
      st.plotly_chart(fig3, use_container_width=True)

    with c4:
      st.subheader('4. Random Forest Feature Importance')
      importances = model.feature_importances_
      feat_imp_df = pd.DataFrame(
          {'Feature': feature_cols, 'Importance': importances}
      ).sort_values(by='Importance', ascending=True)
      fig4 = px.bar(
          feat_imp_df,
          x='Importance',
          y='Feature',
          orientation='h',
          title='Which Features Drive Predictions?',
          color='Importance',
          color_continuous_scale='blues',
      )
      st.plotly_chart(fig4, use_container_width=True)

    # Row 3: Graphs 5 & 6
    c5, c6 = st.columns(2)
    with c5:
      st.subheader('5. Feature Correlation Heatmap')
      corr_matrix = df[feature_cols + [target_col]].corr()
      fig5 = px.imshow(
          corr_matrix,
          text_auto='.2f',
          aspect='auto',
          title='Correlation Matrix of Student Metrics',
          color_continuous_scale='RdBu_r',
      )
      st.plotly_chart(fig5, use_container_width=True)

    with c6:
      st.subheader('6. Pass vs At-Risk Proportion')
      status_counts = df['status_label'].value_counts().reset_index()
      status_counts.columns = ['Status', 'Count']
      fig6 = px.pie(
          status_counts,
          names='Status',
          values='Count',
          title='Overall Student Cohort Risk Breakdown',
          hole=0.4,
          color='Status',
          color_discrete_map={
              'Safe / On Track': '#2ecc71',
              'At-Risk': '#e74c3c',
          },
      )
      st.plotly_chart(fig6, use_container_width=True)

    # Row 4: Graphs 7 & 8
    c7, c8 = st.columns(2)
    with c7:
      st.subheader('7. Assignments Completed by Age')
      avg_assignments = (
          df.groupby('age')['assignments_completed'].mean().reset_index()
      )
      fig7 = px.bar(
          avg_assignments,
          x='age',
          y='assignments_completed',
          title='Average Assignments Completed by Student Age',
          color='assignments_completed',
          color_continuous_scale='viridis',
      )
      st.plotly_chart(fig7, use_container_width=True)

    with c8:
      st.subheader('8. Mid-term Scores Across Age Groups')
      fig8 = px.violin(
          df,
          x='age',
          y='midterm_score',
          box=True,
          points='all',
          title='Mid-term Score Spread Across Student Ages',
          color='age',
      )
      st.plotly_chart(fig8, use_container_width=True)

    # Row 5: Graphs 9 & 10
    c9, c10 = st.columns(2)
    with c9:
      st.subheader('9. Attendance vs. Previous GPA')
      fig9 = px.scatter(
          df,
          x='attendance_rate',
          y='previous_gpa',
          color='status_label',
          size='study_hours',
          title='Attendance Rate vs Previous GPA (Bubble = Study Hours)',
          color_discrete_map={
              'Safe / On Track': '#2ecc71',
              'At-Risk': '#e74c3c',
          },
      )
      st.plotly_chart(fig9, use_container_width=True)

    with c10:
      st.subheader('10. Average Study Hours by Risk Status')
      avg_study = df.groupby('status_label')['study_hours'].mean().reset_index()
      fig10 = px.bar(
          avg_study,
          x='status_label',
          y='study_hours',
          color='status_label',
          title='Comparison of Mean Study Hours by Performance Group',
          color_discrete_map={
              'Safe / On Track': '#2ecc71',
              'At-Risk': '#e74c3c',
          },
      )
      st.plotly_chart(fig10, use_container_width=True)

  # --- TAB 3: MODEL PERFORMANCE ---
  with tab3:
    st.header('Random Forest Model Evaluation')
    m_col1, m_col2 = st.columns(2)
    with m_col1:
      st.metric(
          label='Model Test Accuracy', value=f'{accuracy * 100:.2f}%'
      )
    with m_col2:
      st.info(
          f'Trained on **{len(X_train)}** student samples | Tested on'
          f' **{len(X_test)}** samples.'
      )

    st.markdown('### 📈 Classification Report')
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df, use_container_width=True)

    st.markdown('### 🔍 Confusion Matrix')
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        labels=dict(
            x='Predicted Outcome', y='Actual Outcome', color='Count'
        ),
        x=['At-Risk (0)', 'Safe (1)'],
        y=['At-Risk (0)', 'Safe (1)'],
        title='Confusion Matrix Heatmap',
        color_continuous_scale='Blues',
    )
    st.plotly_chart(fig_cm, use_container_width=True)

  # --- TAB 4: PREDICTION & INTERVENTION ---
  with tab4:
    st.header('🎯 Individual Student Prediction & Early Intervention Support')
    st.markdown(
        'Adjust the student parameters below to generate a real-time risk'
        ' evaluation and tailored faculty intervention plan.'
    )

    with st.form('prediction_form'):
      f_col1, f_col2, f_col3 = st.columns(3)
      with f_col1:
        age_in = st.slider(
            'Age',
            int(df['age'].min()),
            int(df['age'].max()),
            int(df['age'].mean()),
        )
        study_in = st.slider(
            'Study Hours / Week',
            float(df['study_hours'].min()),
            float(df['study_hours'].max()),
            float(df['study_hours'].mean()),
            0.5,
        )
      with f_col2:
        attendance_in = st.slider(
            'Attendance Rate (%)',
            int(df['attendance_rate'].min()),
            int(df['attendance_rate'].max()),
            int(df['attendance_rate'].mean()),
        )
        gpa_in = st.slider(
            'Previous GPA',
            float(df['previous_gpa'].min()),
            float(df['previous_gpa'].max()),
            float(df['previous_gpa'].mean()),
            0.1,
        )
      with f_col3:
        midterm_in = st.slider(
            'Midterm Score (%)',
            int(df['midterm_score'].min()),
            int(df['midterm_score'].max()),
            int(df['midterm_score'].mean()),
        )
        assignments_in = st.slider(
            'Assignments Completed',
            int(df['assignments_completed'].min()),
            int(df['assignments_completed'].max()),
            int(df['assignments_completed'].mean()),
        )

      submit_btn = st.form_submit_button(
          'Evaluate Student & Generate Intervention Plan', type='primary'
      )

    if submit_btn:
      single_input = pd.DataFrame([{
          'age': age_in,
          'study_hours': study_in,
          'attendance_rate': attendance_in,
          'previous_gpa': gpa_in,
          'midterm_score': midterm_in,
          'assignments_completed': assignments_in,
      }])

      pred = model.predict(single_input)[0]
      proba = model.predict_proba(single_input)[0]
      conf = proba[int(pred)] * 100

      st.markdown('---')
      st.subheader('📊 Evaluation Results')
      r_col1, r_col2 = st.columns(2)
      with r_col1:
        if pred == 1:
          st.success(
              f'### Status: Low Risk / On Track\n**Confidence:** {conf:.1f}%'
          )
        else:
          st.error(
              f'### Status: At-Risk / Intervention Required\n**Confidence:**'
              f' {conf:.1f}%'
          )
      with r_col2:
        st.metric(label='Prediction Confidence', value=f'{conf:.1f}%')

      st.markdown('### 💡 Automated Early Intervention Recommendations')
      if pred == 0:
        st.warning(
            '⚠️ **Targeted Faculty Action Plan Recommended:**'
        )
        if attendance_in < df['attendance_rate'].mean():
          st.markdown(
              f'- 📌 **Attendance Alert:** Attendance ({attendance_in}%) is'
              f' below cohort average ({df["attendance_rate"].mean():.1f}%).'
              ' Schedule a counseling check-in.'
          )
        if midterm_in < df['midterm_score'].mean():
          st.markdown(
              f'- 📌 **Academic Support:** Midterm score ({midterm_in}%) is'
              f' below cohort average ({df["midterm_score"].mean():.1f}%).'
              ' Enroll student in peer tutoring.'
          )
        if study_in < df['study_hours'].mean():
          st.markdown(
              '- 📌 **Time Management:** Weekly study hours are low. Provide'
              ' structured study planning resources.'
          )
      else:
        st.info(
            '✅ **On Track:** Student profile is healthy. Maintain regular'
            ' academic monitoring and positive reinforcement.'
        )
else:
  st.warning(
      'Please ensure "large_student_performance_dataset.csv" is in your project'
      ' folder.'
  )