import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

df = pd.read_csv("/Users/sahitisriupputuri/Desktop/Singular /Predictive_TrainingData.csv")

# Convert all to lowercase
df['category_group'] = df['category_group'].str.lower()


# Date processing
# -------------------------------
df['posted_month_year'] = pd.to_datetime(df['posted_month_year'], errors='coerce')
df = df.dropna(subset=['posted_month_year'])
df['posted_month_year'] = df['posted_month_year'].dt.to_period('M').dt.to_timestamp()

# Last 6 months
last_date = df['posted_month_year'].max()
cutoff_date = last_date - pd.DateOffset(months=5)
df = df[df['posted_month_year'] >= cutoff_date]

# Remove 'others'
df = df[df['category_group'] != 'others']

# -------------------------------
#  Aggregate top 30 per month
# -------------------------------
monthly_counts = df.groupby(['posted_month_year', 'category_group'], as_index=False)['total_count'].sum()

monthly_top30 = monthly_counts.groupby('posted_month_year').apply(
    lambda x: x.sort_values('total_count', ascending=False).head(30)
).reset_index(drop=True)

# -------------------------------
#  Create dashboard function
# -------------------------------
def create_dashboard(df_top30):
    pivot_df = df_top30.pivot(index='category_group', columns='posted_month_year', values='total_count')

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.35, 0.65],
        specs=[[{"type": "table"}, {"type": "scatter"}]],
        subplot_titles=["Top Roles for Selected Month", "Job Roles Trend"]
    )

    # Right panel: line chart (counts on y-axis)
    for cat in pivot_df.index:
        df_cat = df_top30[df_top30['category_group'] == cat].sort_values('posted_month_year')
        fig.add_trace(
            go.Scatter(
                x=df_cat['posted_month_year'],
                y=df_cat['total_count'],
                mode='lines+markers',
                name=cat,
                line=dict(width=2),
                marker=dict(size=6),
                hovertemplate='%{y} jobs in %{x}<extra>%{fullData.name}</extra>'
            ),
            row=1, col=2
        )

    # Left panel: table
    months = sorted(df_top30['posted_month_year'].unique())
    for month in months:
        df_month = df_top30[df_top30['posted_month_year'] == month].sort_values('total_count', ascending=False)
        fig.add_trace(
            go.Table(
                header=dict(values=["Job Role", "Count"], fill_color='lightgrey', align='left'),
                cells=dict(values=[df_month['category_group'], df_month['total_count']], fill_color='white', align='left'),
                visible=(month == months[0])
            ),
            row=1, col=1
        )

    # Dropdown for months
    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                x=0.25,
                y=1.15,
                xanchor='center',
                yanchor='top',
                buttons=[
                    dict(
                        label=str(pd.to_datetime(month).strftime('%Y-%m')),
                        method='update',
                        args=[
                            {"visible": [True]*len(pivot_df.index) + [m == month for m in months]},
                            {"title": f"Job Roles Trend + Top Roles for {pd.to_datetime(month).strftime('%Y-%m')}"}
                        ]
                    ) for month in months
                ]
            )
        ],
        template='plotly_white',
        height=900,
        width=1800,
        hovermode='x unified'
    )

    fig.update_yaxes(title="Job Count", row=1, col=2)
    fig.update_xaxes(title="Month", row=1, col=2)

    return fig

# -------------------------------
#  Generate dashboard
# -------------------------------
fig_all = create_dashboard(monthly_top30)

# -------------------------------
#  Show locally
# -------------------------------
fig_all.show()


    

