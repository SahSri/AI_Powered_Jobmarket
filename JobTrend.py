import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

df = pd.read_csv("/Users/sahitisriupputuri/Desktop/Singular /Predictive_TrainingData.csv")

# Convert all to lowercase
df['category_group'] = df['category_group'].str.lower()

# Normalize categories (your existing normalize_role)
def normalize_role(role):
    if '.net' in role or 'dot net' in role or 'net' in role:
        return '.net developer'
    elif 'java' in role or 'developers' in role:
        return 'java developer'
    elif 'full' in role:
        return 'full stack developer'
    elif 'data engineer' in role:
        return 'data engineer'
    elif 'business' in role or 'business system' in role:
        return 'business analyst'
    elif 'devops' in role:
        return 'devops engineer'
    elif 'aws' in role:
        return 'aws engineer'
    elif 'cloud' in role:
        return 'cloud engineer'
    elif 'sap' in role:
        return 'sap hana/ead consultant'
    elif 'azure' in role:
        return 'azure data/devops engineer'
    elif 'software' in role:
        return 'software development engineer'
    elif 'python' in role:
        return 'python developer'
    elif 'qa' in role or 'quality' in role:
        return 'qa engineer'
    elif 'system engineer' in role:
        return 'system engineer'
    elif 'web' in role:
        return 'web developer'
    elif 'procurement' in role:
        return 'procurement specialist manager'
    elif 'sourcing' in role or 'staffing' in role or 'category' in role:
        return 'sourcing/staffing category manager'
    elif 'technical' in role:
        return 'technical lead'
    elif 'supply chain' in role:
        return 'supply chain manager'
    elif 'project manager' in role:
        return 'project manager'
    elif 'data analyst' in role or 'analytics' in role:
        return 'data analyst'
    elif 'angular' in role:
        return 'angular developer'
    elif 'front end' in role or 'front-end' in role or 'reactjs' in role or 'node' in role:
        return 'front end developer'
    elif 'ios' in role:
        return 'ios developer'
    elif 'oracle' in role:
        return 'oracle developer'
    elif 'architect' in role or 'etl' in role:
        return 'data solution architect'
    elif 'php' in role:
        return 'php developer'
    elif 'warehouse' in role:
        return 'warehouse supervisor/manager'
    elif 'snowflake' in role:
        return 'snowflake developer'
    elif 'salesforce' in role:
        return 'salesforce developer'
    elif 'logistics' in role:
        return 'logistics coordinator/manager'
    elif 'cyber' in role:
        return 'cyber security analyst'
    elif 'microsoft' in role:
        return 'microsoft services power apps developer'
    elif 'cobol' in role:
        return 'cobol developer'
    elif 'inventory' in role:
        return 'inventory control specialist'
    elif 'scientist' in role or 'science' in role:
        return 'data scientist'
    elif 'share' in role:
        return 'share point developer'
    elif 'sdet' in role or 'test' in role:
        return 'software test engineer'
    elif 'power' in role or 'bi' in role:
        return 'power bi developer'
    elif 'kinaxsis' in role:
        return 'kinaxsis integration'
    return role
df['category_group'] = df['category_group'].apply(normalize_role)
# -------------------------------
# Step 3: Date processing
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
# Step 4: Aggregate top 30 per month
# -------------------------------
monthly_counts = df.groupby(['posted_month_year', 'category_group'], as_index=False)['total_count'].sum()

monthly_top30 = monthly_counts.groupby('posted_month_year').apply(
    lambda x: x.sort_values('total_count', ascending=False).head(30)
).reset_index(drop=True)

# -------------------------------
# Step 5: Create dashboard function
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
# Step 6: Generate dashboard
# -------------------------------
fig_all = create_dashboard(monthly_top30)

# -------------------------------
# Step 7: Show locally
# -------------------------------
fig_all.show()

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

df = pd.read_csv("/Users/sahitisriupputuri/Desktop/Singular /Predictive_TrainingData.csv")

# Convert all to lowercase
df['category_group'] = df['category_group'].str.lower()

# Normalize categories (your existing normalize_role)
def normalize_role(role):
    if '.net' in role or 'dot net' in role or 'net' in role:
        return '.net developer'
    elif 'java' in role or 'developers' in role:
        return 'java developer'
    elif 'full' in role:
        return 'full stack developer'
    elif 'data engineer' in role:
        return 'data engineer'
    elif 'business' in role or 'business system' in role:
        return 'business analyst'
    elif 'devops' in role:
        return 'devops engineer'
    elif 'aws' in role:
        return 'aws engineer'
    elif 'cloud' in role:
        return 'cloud engineer'
    elif 'sap' in role:
        return 'sap hana/ead consultant'
    elif 'azure' in role:
        return 'azure data/devops engineer'
    elif 'software' in role:
        return 'software development engineer'
    elif 'python' in role:
        return 'python developer'
    elif 'qa' in role or 'quality' in role:
        return 'qa engineer'
    elif 'system engineer' in role:
        return 'system engineer'
    elif 'web' in role:
        return 'web developer'
    elif 'procurement' in role:
        return 'procurement specialist manager'
    elif 'sourcing' in role or 'staffing' in role or 'category' in role:
        return 'sourcing/staffing category manager'
    elif 'technical' in role:
        return 'technical lead'
    elif 'supply chain' in role:
        return 'supply chain manager'
    elif 'project manager' in role:
        return 'project manager'
    elif 'data analyst' in role or 'analytics' in role:
        return 'data analyst'
    elif 'angular' in role:
        return 'angular developer'
    elif 'front end' in role or 'front-end' in role or 'reactjs' in role or 'node' in role:
        return 'front end developer'
    elif 'ios' in role:
        return 'ios developer'
    elif 'oracle' in role:
        return 'oracle developer'
    elif 'architect' in role or 'etl' in role:
        return 'data solution architect'
    elif 'php' in role:
        return 'php developer'
    elif 'warehouse' in role:
        return 'warehouse supervisor/manager'
    elif 'snowflake' in role:
        return 'snowflake developer'
    elif 'salesforce' in role:
        return 'salesforce developer'
    elif 'logistics' in role:
        return 'logistics coordinator/manager'
    elif 'cyber' in role:
        return 'cyber security analyst'
    elif 'microsoft' in role:
        return 'microsoft services power apps developer'
    elif 'cobol' in role:
        return 'cobol developer'
    elif 'inventory' in role:
        return 'inventory control specialist'
    elif 'scientist' in role or 'science' in role:
        return 'data scientist'
    elif 'share' in role:
        return 'share point developer'
    elif 'sdet' in role or 'test' in role:
        return 'software test engineer'
    elif 'power' in role or 'bi' in role:
        return 'power bi developer'
    elif 'kinaxsis' in role:
        return 'kinaxsis integration'
    return role
df['category_group'] = df['category_group'].apply(normalize_role)
# -------------------------------
# Step 3: Date processing
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
# Step 4: Aggregate top 30 per month
# -------------------------------
monthly_counts = df.groupby(['posted_month_year', 'category_group'], as_index=False)['total_count'].sum()

monthly_top30 = monthly_counts.groupby('posted_month_year').apply(
    lambda x: x.sort_values('total_count', ascending=False).head(30)
).reset_index(drop=True)

# -------------------------------
# Step 5: Create dashboard function
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
# Step 6: Generate dashboard
# -------------------------------
fig_all = create_dashboard(monthly_top30)

# -------------------------------
# Step 7: Show locally
# -------------------------------
fig_all.show()
# Save your dashboard as a standalone HTML
fig_all.write_html("/Users/sahitisriupputuri/Desktop/job_roles_trend_dashboard.html")

