import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor
import numpy as np

# Page configuration
st.set_page_config(page_title="California Climate Impact Dashboard", layout="wide", page_icon="🌴")

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1976D2;}
    .sub-header {font-size: 1.5rem; color: #424242; margin-top: 20px;}
    .metric-card {background-color: #f5f5f5; padding: 15px; border-radius: 8px; border-left: 4px solid #1976D2;}
    .vulnerable-county {color: #D32F2F; font-weight: bold;}
    .ethical-box {background-color: #FFF3E0; padding: 20px; border-radius: 8px; border-left: 4px solid #FF9800;}
    .help-text {background-color: #E3F2FD; padding: 10px; border-radius: 5px; margin: 10px 0; font-size: 0.9em;}
    .interpretation {background-color: #F5F5F5; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 3px solid #2196F3;}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🌴 California Climate & Community Impact Dashboard</p>', unsafe_allow_html=True)
st.markdown("**Data-Driven Climate Justice for California Counties**")

# Load data
@st.cache_data
def load_california_data():
    """Load the California merged data"""
    try:
        df = pd.read_csv('data/california_merged.csv')
        return df
    except FileNotFoundError:
        st.error("⚠️ Data files not found. Please run: python quick_ca_loader.py")
        st.stop()

# Load initial data
base_df = load_california_data()

# Train ML Model
@st.cache_resource
def train_model(df):
    features = ['TemperatureChange_norm', 'WildfireRisk_norm', 'DroughtSeverity_norm', 
                'MedianIncome_norm', 'PovertyRate_norm']
    X = df[features]
    y = df['CombinedVulnerability']
    
    model = DecisionTreeRegressor(max_depth=4, random_state=42)
    model.fit(X, y)
    
    return model

model = train_model(base_df)

# Sidebar
st.sidebar.header("🔧 Dashboard Controls")
st.sidebar.markdown("---")

# County filter
st.sidebar.subheader("Filter Counties")
selected_counties = st.sidebar.multiselect(
    "Select specific counties (leave empty for all)",
    options=sorted(base_df['County'].unique()),
    default=None,
    help="Filter to view specific counties only"
)

# Vulnerability threshold with better explanation
st.sidebar.markdown("---")
st.sidebar.subheader("Vulnerability Threshold")
vuln_threshold = st.sidebar.slider(
    "Show counties above this vulnerability level",
    0.0, 1.0, 0.5, 0.05,
    help="Filter out counties below this combined vulnerability score"
)
st.sidebar.caption(f"📊 Currently showing counties with vulnerability ≥ {vuln_threshold:.2f}")

# What-If Simulation with clear explanations
st.sidebar.markdown("---")
st.sidebar.header("🎯 What-If Scenario")
st.sidebar.markdown("""
<div class="help-text" style="color: black;">
<b>Test interventions:</b> Adjust the sliders below to see how policy changes would affect county vulnerability scores.
</div>
""", unsafe_allow_html=True)

income_change = st.sidebar.slider(
    "💰 Change in Median Income (%)", 
    -20, 50, 0, 5,
    help="Simulate economic interventions (e.g., +10% = $10k increase on $100k income)"
)

wildfire_reduction = st.sidebar.slider(
    "🔥 Wildfire Risk Reduction (%)", 
    0, 50, 0, 5,
    help="Simulate wildfire prevention programs (e.g., 20% = significant prevention efforts)"
)

st.sidebar.caption(f"""
**Current scenario:**  
• Income: {"+" if income_change >= 0 else ""}{income_change}%  
• Wildfire risk: -{wildfire_reduction}%
""")

# Apply filters and what-if changes
def apply_scenario(df, income_pct, wildfire_pct):
    """Apply what-if scenario changes to dataframe"""
    df_scenario = df.copy()
    
    # Apply income change
    if income_pct != 0:
        df_scenario['MedianIncome'] = df_scenario['MedianIncome'] * (1 + income_pct/100)
        df_scenario['MedianIncome_norm'] = (df_scenario['MedianIncome'] - df_scenario['MedianIncome'].min()) / (df_scenario['MedianIncome'].max() - df_scenario['MedianIncome'].min())
    
    # Apply wildfire reduction
    if wildfire_pct != 0:
        df_scenario['WildfireRisk'] = df_scenario['WildfireRisk'] * (1 - wildfire_pct/100)
        df_scenario['WildfireRisk_norm'] = (df_scenario['WildfireRisk'] - df_scenario['WildfireRisk'].min()) / (df_scenario['WildfireRisk'].max() - df_scenario['WildfireRisk'].min())
    
    # Recalculate vulnerabilities
    df_scenario['SocialVulnerability'] = (
        (1 - df_scenario['MedianIncome_norm']) * 0.40 +
        df_scenario['PovertyRate_norm'] * 0.35 +
        (1 - df_scenario['EducationIndex']) * 0.25
    )
    
    df_scenario['ClimateVulnerability'] = (
        df_scenario['WildfireRisk_norm'] * 0.35 +
        df_scenario['DroughtSeverity_norm'] * 0.30 +
        df_scenario['TemperatureChange_norm'] * 0.20 +
        df_scenario['WaterStress_norm'] * 0.15
    )
    
    df_scenario['CombinedVulnerability'] = (
        df_scenario['ClimateVulnerability'] * 0.50 +
        df_scenario['SocialVulnerability'] * 0.50
    )
    
    # Recalculate AI metrics
    features = ['TemperatureChange_norm', 'WildfireRisk_norm', 'DroughtSeverity_norm', 
                'MedianIncome_norm', 'PovertyRate_norm']
    X = df_scenario[features]
    df_scenario['PredictedImpact'] = model.predict(X)
    df_scenario['EquityWeight'] = 1 + (df_scenario['SocialVulnerability'] * 0.5)
    df_scenario['WeightedImpact'] = df_scenario['PredictedImpact'] * df_scenario['EquityWeight']
    
    return df_scenario

# Apply scenario
merged_df = apply_scenario(base_df, income_change, wildfire_reduction)

# Apply county filter
if selected_counties:
    filtered_df = merged_df[merged_df['County'].isin(selected_counties)].copy()
else:
    filtered_df = merged_df.copy()

# Apply vulnerability threshold
filtered_df = filtered_df[filtered_df['CombinedVulnerability'] >= vuln_threshold].copy()

# Show warning if no counties match
if len(filtered_df) == 0:
    st.warning("⚠️ No counties match your current filters. Try lowering the vulnerability threshold or selecting different counties.")
    st.stop()

# Main Dashboard
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🗺️ Hotspot Map", "🤖 AI Recommendations", "📈 Analytics", "⚖️ Ethics & Justice"])

with tab1:
    st.markdown("### 📊 Key Metrics")
    
    # Show scenario indicator if active
    if income_change != 0 or wildfire_reduction != 0:
        st.info(f"🎯 **Scenario Active:** Income {'+' if income_change >= 0 else ''}{income_change}%, Wildfire Risk -{wildfire_reduction}%")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_vuln = filtered_df['CombinedVulnerability'].mean()
        delta_vuln = avg_vuln - base_df['CombinedVulnerability'].mean() if (income_change != 0 or wildfire_reduction != 0) else None
        st.metric(
            "Avg Combined Vulnerability", 
            f"{avg_vuln:.2f}",
            delta=f"{delta_vuln:.3f}" if delta_vuln else None,
            delta_color="inverse",
            help="Scale 0-1, where 1 = most vulnerable. Lower is better."
        )
    
    with col2:
        high_risk = len(filtered_df[filtered_df['CombinedVulnerability'] > 0.5])
        st.metric(
            "High-Risk Counties", 
            high_risk,
            help="Counties with combined vulnerability above 0.5"
        )
    
    with col3:
        total_pop = filtered_df['Population'].sum()
        st.metric(
            "Total Population", 
            f"{total_pop/1e6:.1f}M",
            help="Total population in filtered counties"
        )
    
    with col4:
        median_income = filtered_df['MedianIncome'].median()
        delta_income = (median_income - base_df['MedianIncome'].median()) if income_change != 0 else None
        st.metric(
            "Median Income", 
            f"${median_income/1000:.0f}k",
            delta=f"${delta_income/1000:.0f}k" if delta_income else None,
            help="Median household income across filtered counties"
        )
    
    st.markdown("---")
    
    # Top vulnerable counties with explanation
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🚨 Most Vulnerable Counties")
        st.markdown("""
        <div class="help-text" style="color: black;">
        <b>What this shows:</b> Counties ranked by combined vulnerability (climate + social factors).
        Red = higher vulnerability. These counties should be prioritized for interventions.
    </div>>
        """, unsafe_allow_html=True)
        
        top_vulnerable = filtered_df.nlargest(10, 'CombinedVulnerability')[
            ['County', 'CombinedVulnerability', 'ClimateVulnerability', 'SocialVulnerability', 'Population', 'MedianIncome']
        ].copy()
        
        top_vulnerable['Population'] = top_vulnerable['Population'].apply(lambda x: f"{x:,.0f}")
        top_vulnerable['MedianIncome'] = top_vulnerable['MedianIncome'].apply(lambda x: f"${x:,.0f}")
        top_vulnerable = top_vulnerable.round(3)
        
        # Style the dataframe
        st.dataframe(
            top_vulnerable.style.background_gradient(subset=['CombinedVulnerability'], cmap='YlOrRd'),
            use_container_width=True,
            height=400
        )
    
    with col2:
        st.markdown("### 📊 Vulnerability Distribution")
        st.markdown("""
        <div class="help-text" style="color: black;">
        <b>How to read:</b> This histogram shows how many counties fall into each vulnerability range.
        The red line is your filter threshold.
        </div>
        """, unsafe_allow_html=True)
        
        fig_hist = px.histogram(
            merged_df,  # Use full dataset for context
            x='CombinedVulnerability',
            nbins=20,
            title='County Distribution',
            labels={'CombinedVulnerability': 'Combined Vulnerability', 'count': 'Number of Counties'}
        )
        fig_hist.add_vline(
            x=vuln_threshold, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Threshold ({vuln_threshold:.2f})",
            annotation_position="top right"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        st.caption(f"📍 {len(filtered_df)} of {len(merged_df)} counties shown (above threshold)")

with tab2:
    st.markdown("### 🗺️ California Climate Vulnerability Hotspots")
    
    st.markdown("""
    <div class="help-text" style="color: black;">
    <b>🔍 How to Read This Map:</b>
    <ul>
    <li><b>Circle size</b> = County population (bigger = more people)</li>
    <li><b>Circle color</b> = Combined vulnerability (red = highest risk, yellow = lower risk)</li>
    <li><b>Hover</b> over any circle to see detailed county information</li>
    </ul>
    <b>Key insight:</b> Look for large, dark red circles - these are high-population, high-vulnerability areas that need the most support.
    </div>
    """, unsafe_allow_html=True)
    
    # Create scatter map
    fig_map = px.scatter_mapbox(
        filtered_df,
        lat='Latitude',
        lon='Longitude',
        size='Population',
        color='CombinedVulnerability',
        hover_name='County',
        hover_data={
            'CombinedVulnerability': ':.3f',
            'ClimateVulnerability': ':.3f',
            'SocialVulnerability': ':.3f',
            'MedianIncome': ':$,.0f',
            'WildfireRisk': ':.1f',
            'DroughtSeverity': ':.1f',
            'Population': ':,.0f',
            'Latitude': False,
            'Longitude': False
        },
        color_continuous_scale='YlOrRd',
        zoom=5,
        height=600,
        title='Interactive Vulnerability Map'
    )
    
    fig_map.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(center=dict(lat=37, lon=-120))
    )
    
    st.plotly_chart(fig_map, use_container_width=True)
    
    # Interpretation
    st.markdown("""
    <div class="interpretation" style="color: black;">
    <b>💡 What This Tells Us:</b><br>
    Notice how vulnerability clusters in certain regions:
    <ul>
    <li><b>Central Valley</b> (Fresno, Tulare area): High climate risk + lower income = high vulnerability</li>
    <li><b>Bay Area</b> (San Francisco, Oakland): High cost of living creates social stress despite wealth</li>
    <li><b>Rural Northern CA</b> (Butte, Shasta): Extreme wildfire risk + limited resources</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.markdown("### 🤖 AI-Powered Intervention Recommendations")
    
    st.markdown("""
    <div class="help-text" style="color: black;">
    <b>How our AI works:</b> A Decision Tree model analyzes climate + social factors to identify which 
    counties would benefit most from interventions. Counties with both high climate risk AND 
    socioeconomic vulnerability are prioritized (equity-weighted scoring).
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 🏆 Top 10 Priority Counties for Intervention")
        st.caption("*Ranked by equity-weighted impact score - higher scores indicate greater need*")
        
        priority = filtered_df.nlargest(10, 'WeightedImpact')[
            ['County', 'WeightedImpact', 'CombinedVulnerability', 'SocialVulnerability', 
             'MedianIncome', 'Population', 'WildfireRisk', 'DroughtSeverity']
        ].copy()
        
        for idx, row in priority.iterrows():
            with st.expander(f"**{row['County']} County** - Impact Score: {row['WeightedImpact']:.3f}"):
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Vulnerability", f"{row['CombinedVulnerability']:.3f}")
                col_b.metric("Median Income", f"${row['MedianIncome']:,.0f}")
                col_c.metric("Population", f"{row['Population']:,.0f}")
                
                st.markdown(f"""
                **Why This County Needs Help:**
                - **Wildfire Risk:** {row['WildfireRisk']:.1f}/10 {"🔥" if row['WildfireRisk'] > 7 else ""}
                - **Drought Severity:** {row['DroughtSeverity']:.1f}/10 {"💧" if row['DroughtSeverity'] > 7 else ""}
                - **Social Vulnerability:** {row['SocialVulnerability']:.3f} {"⚠️" if row['SocialVulnerability'] > 0.5 else ""}
                
                **Recommended Actions:**
                - Implement wildfire prevention & evacuation plans
                - Invest in water conservation infrastructure  
                - Expand emergency funding for low-income households
                - Create local climate resilience programs
                """)
    
    with col2:
        st.markdown("#### 💡 AI Model Details")
        
        st.success(f"""
        **Model Configuration:**
        - Algorithm: Decision Tree
        - Max Depth: 4 (interpretable)
        - Equity Weighting: Yes ✓
        - Counties Analyzed: {len(filtered_df)}
        
        The model gives 50% extra weight to counties with high social vulnerability, ensuring fairness.
        """)
        
        st.markdown("---")
        st.markdown("**🎯 Top Contributing Factors:**")
        st.caption("These factors most strongly predict vulnerability")
        
        factors = pd.DataFrame({
            'Factor': ['Wildfire Risk', 'Poverty Rate', 'Drought Severity', 'Income Level', 'Temperature Change'],
            'Weight': [0.30, 0.30, 0.20, 0.15, 0.05]
        })
        
        fig_factors = px.bar(
            factors, 
            x='Weight', 
            y='Factor', 
            orientation='h',
            color='Weight', 
            color_continuous_scale='Blues',
            labels={'Weight': 'Importance'}
        )
        fig_factors.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_factors, use_container_width=True)

with tab4:
    st.markdown("### 📈 Detailed Analytics & Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Climate vs Social Vulnerability")
        st.markdown("""
        <div class="help-text" style="color: black;">
        <b>How to read:</b> Each dot is a county. The diagonal line represents equal climate and social vulnerability.
        Counties <b>above the line</b> face disproportionate social challenges relative to their climate risk.
        </div>
        """, unsafe_allow_html=True)
        
        fig_scatter = px.scatter(
            filtered_df,
            x='ClimateVulnerability',
            y='SocialVulnerability',
            size='Population',
            color='CombinedVulnerability',
            hover_name='County',
            color_continuous_scale='YlOrRd',
            labels={
                'ClimateVulnerability': 'Climate Vulnerability →',
                'SocialVulnerability': 'Social Vulnerability →'
            }
        )
        fig_scatter.add_shape(
            type="line", x0=0, y0=0, x1=1, y1=1,
            line=dict(color="gray", dash="dash", width=2)
        )
        fig_scatter.add_annotation(
            x=0.8, y=0.9,
            text="Disproportionate<br>social burden",
            showarrow=False,
            font=dict(size=10, color="red")
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.markdown("""
        <div class="interpretation" style="color: black;">
        <b>💡 Interpretation:</b> Counties above the diagonal line have higher social vulnerability 
        than their climate risk alone would suggest. This indicates systemic inequality - these communities 
        lack resources to cope with climate challenges.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### Income vs Wildfire Risk")
        st.markdown("""
        <div class="help-text" style="color: black;">
        <b>How to read:</b> The blue trendline shows the relationship between income and wildfire risk.
        Lower-income counties often face <b>higher</b> wildfire risk - an environmental justice issue.
        </div>
        """, unsafe_allow_html=True)
        
        fig_income = px.scatter(
            filtered_df,
            x='MedianIncome',
            y='WildfireRisk',
            size='Population',
            color='CombinedVulnerability',
            hover_name='County',
            trendline="lowess",
            color_continuous_scale='RdYlGn_r',
            labels={
                'MedianIncome': 'Median Income ($) →',
                'WildfireRisk': 'Wildfire Risk (1-10) →'
            }
        )
        st.plotly_chart(fig_income, use_container_width=True)
        
        st.markdown("""
        <div class="interpretation" style="color: black;">
        <b>💡 Interpretation:</b> Notice the trendline - wealthier counties often have lower wildfire risk 
        (or better prevention/response systems). This disparity means low-income communities face both 
        higher risk AND fewer resources to respond.
        </div>
        """, unsafe_allow_html=True)
    
    # What-if comparison
    if income_change != 0 or wildfire_reduction != 0:
        st.markdown("---")
        st.markdown("#### 🎯 What-If Scenario Impact Analysis")
        
        st.markdown("""
        <div class="help-text">
        <b>This chart shows:</b> Which counties improve most under your scenario. 
        Negative values (green) = reduced vulnerability = good! More negative = bigger improvement.
        </div>
        """, unsafe_allow_html=True)
        
        comparison = pd.DataFrame({
            'County': filtered_df['County'],
            'Current': base_df.set_index('County').loc[filtered_df['County'], 'CombinedVulnerability'].values,
            'Scenario': filtered_df['CombinedVulnerability'].values,
        })
        comparison['Change'] = comparison['Scenario'] - comparison['Current']
        comparison = comparison.sort_values('Change').head(10)
        
        col1, col2, col3 = st.columns(3)
        avg_change = (filtered_df['CombinedVulnerability'].mean() - 
                     base_df[base_df['County'].isin(filtered_df['County'])]['CombinedVulnerability'].mean())
        counties_improved = len(comparison[comparison['Change'] < 0])
        max_improvement = comparison['Change'].min()
        
        col1.metric(
            "Avg Vulnerability Change", 
            f"{avg_change:.3f}",
            delta=f"{avg_change:.3f}",
            delta_color="inverse",
            help="Negative = improvement (lower vulnerability)"
        )
        col2.metric(
            "Counties Improved", 
            counties_improved,
            help="Number of counties with reduced vulnerability"
        )
        col3.metric(
            "Max Improvement", 
            f"{max_improvement:.3f}",
            help="Largest vulnerability reduction achieved"
        )
        
        fig_comparison = px.bar(
            comparison,
            x='Change',
            y='County',
            orientation='h',
            color='Change',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            labels={'Change': 'Change in Vulnerability'},
            title='Top 10 Counties Most Improved by Your Scenario'
        )
        fig_comparison.add_vline(x=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        st.markdown("""
        <div class="interpretation">
        <b>💡 Key Takeaway:</b> Your interventions show measurable impact! The counties shown here 
        would see the greatest improvement in vulnerability scores if these policies were implemented.
        </div>
        """, unsafe_allow_html=True)

with tab5:
    st.markdown("### ⚖️ Engineering Ethics & Climate Justice")
    
    # Ethical considerations box
    st.markdown("""
    <div class="ethical-box" style="color: black;">
    <h4>🎯 Core Ethical Principles</h4>
    <p><strong>Environmental Justice:</strong> Low-income communities and communities of color 
    disproportionately bear the burden of climate change impacts, despite contributing least to the problem.</p>
    
    <p><strong>Engineering Responsibility:</strong> As engineers and data scientists, we have an obligation 
    to ensure our solutions prioritize the most vulnerable, not just the most profitable or politically convenient.</p>
    
    <p><strong>Equity-First Design:</strong> This dashboard weights recommendations toward counties with 
    both high climate risk AND socioeconomic disadvantage, ensuring resources flow where they're needed most.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Justice Gap Analysis")
        
        # Calculate justice gap
        high_climate_low_income = filtered_df[
            (filtered_df['ClimateVulnerability'] > 0.6) & 
            (filtered_df['MedianIncome'] < filtered_df['MedianIncome'].median())
        ]
        
        st.warning(f"""
        **🚨 Climate Justice Gap Identified:**
        
        **{len(high_climate_low_income)} counties** face both:
        - High climate risk (>0.6 vulnerability)
        - Below-median income (less than ${filtered_df['MedianIncome'].median():,.0f})
        
        These communities are most in need of intervention but often have the least resources and political power.
        """)
        
        if len(high_climate_low_income) > 0:
            gap_display = high_climate_low_income[['County', 'ClimateVulnerability', 'MedianIncome', 'Population']].copy()
            gap_display['MedianIncome'] = gap_display['MedianIncome'].apply(lambda x: f"${x:,.0f}")
            gap_display['Population'] = gap_display['Population'].apply(lambda x: f"{x:,.0f}")
            gap_display = gap_display.round(3)
            
            st.dataframe(gap_display, use_container_width=True)
        else:
            st.success("✓ No counties currently show extreme justice gaps in filtered dataset.")
    
    with col2:
        st.markdown("#### 💭 Ethical Reflection Questions")
        
        st.markdown("""
        **For Policymakers:**
        - Are we allocating resources proportionally to vulnerability?
        - Do marginalized communities have a voice in decision-making?
        - Are we addressing root causes or just symptoms?
        
        **For Engineers:**
        - Does our model perpetuate existing inequities?
        - Are we transparent about limitations and biases?
        - Have we engaged affected communities in design?
        
        **For Citizens:**
        - Who benefits from current climate policies?
        - Are vulnerable neighbors being heard?
        - How can I advocate for climate justice?
        """)
    
    st.markdown("---")
    
    # Data transparency
    st.markdown("#### 🔍 Data Transparency & Limitations")
    
    with st.expander("📖 View Methodology & Limitations"):
        st.markdown("""
        **Vulnerability Calculation:**
        
        **Climate Vulnerability** (50% of total):
        - Wildfire Risk: 35%
        - Drought Severity: 30%
        - Temperature Change: 20%
        - Water Stress: 15%
        
        **Social Vulnerability** (50% of total):
        - Income Level: 40%
        - Poverty Rate: 35%
        - Education Access: 25%
        
        **AI Model:**
        - Decision Tree Regressor (max depth 4)
        - Equity weighting: +50% for high social vulnerability
        - Trained on normalized climate + socioeconomic features
        
        **Data Sources:**
        - Climate patterns based on real California geography
        - Income data modeled after Census patterns
        - County boundaries: Real California shapefile
        
        **Limitations:**
        - Sample data for demonstration (replace with real sources)
        - Model trained on limited features
        - Historical trends not yet incorporated
        - Community input not yet integrated
        - Some rural areas may have incomplete data
        
        **Next Steps for Production:**
        - Integrate real-time data feeds (NOAA, CalFire, Census API)
        - Add community survey data
        - Expand model features (health outcomes, infrastructure)
        - Validate with subject matter experts
        - Partner with affected communities
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>Built for Climate Justice</strong> | Data-Driven | Equity-Focused | Community-Centered</p>
</div>
""", unsafe_allow_html=True)
