import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv('mask_data copy.csv')

df = df.groupby('date')['faces_detected', 'masks', 'no_masks'].sum()
df['perc_with_mask'] = df['masks'] / df['faces_detected']
df['perc_without_mask'] = df['no_masks'] / df['faces_detected']

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['date'], y=df['perc_with_mask'],
                    mode='lines',
                    name='With Masks'))

fig.add_trace(go.Scatter(x=df['date'], y=df['perc_without_mask'],
                    mode='lines',
                    name='Without Masks'))

# Edit the layout
fig.update_layout(title='San Francisco, CA - Daily Mask Usage of Observed Pedestrians',
                   xaxis_title='Date',
                   yaxis_title='Observed Percentage',
                   yaxis_tickformat = '%',
                   xaxis_tickangle = 45)

fig.show()