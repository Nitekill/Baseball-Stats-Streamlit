"""
InSeasonStats.py

A Streamlit application for analyzing MLB player statistics and fantasy baseball data.
This tool combines current season MLB statistics with Fantrax fantasy baseball league data
to provide insights for fantasy baseball player analysis and acquisition decisions.

Features:
- Display current season statistical leaders
- Compare two players across various statistical combinations
- Track available free agents in connected fantasy league
- Visualize player comparisons with quadrant analysis

Dependencies:
- streamlit
- pandas
- numpy
- pybaseball
- plotly
- requests

Environment Variables Required:
- FANTRAX_LEAGUE_ID: Your Fantrax league identifier

Usage:
    streamlit run InSeasonStats.py

Data Sources:
- MLB Statistics: pybaseball library
- Fantasy Data: Fantrax API

Author: Nate Killian
Created: 20-JAN-2025
Last Updated: 17-FEB-2025
Version: 1.0.0
"""


import os
import pandas as pd
import numpy as np
import pybaseball
import streamlit as st
import plotly
import plotly.graph_objects as go
import requests
import json
from datetime import datetime

# Constants
CURRENT_YEAR = datetime.now().year
START_YEAR = 2015  # Earliest year for historical analysis
# List of columns that need to be converted from decimal to percentage format (e.g., 0.300 to 30.0)
PERCENTAGE_COLUMNS = ['O-Swing%', 'SwStr%', 'BB%', 'K%', 'Z-Swing%', 'Contact%', 
                     'Z-Contact%', 'Pull%', 'Oppo%', 'GB%', 'FB%', 'LD%']

# Fantrax API Constants

PLAYER_IDS_URL = "https://www.fantrax.com/fxea/general/getPlayerIds"
LEAGUE_INFO_URL = "https://www.fantrax.com/fxea/general/getLeagueInfo"
FANTRAX_PARAMS = {
    'sport': 'MLB',
    'leagueId': os.getenv('FANTRAX_LEAGUE_ID')
}

# Statistical combinations for player comparison analysis
# Each combination defines:
#   - x_stat: metric for x-axis
#   - y_stat: metric for y-axis
#   - description: explanation of what the comparison shows
#   - multiply_x/y: whether to convert decimal to percentage (multiply by 100)
STAT_COMBINATIONS = {
    'Contact Quality vs Pull Tendency': {
        'x_stat': 'Pull%',
        'y_stat': 'HardHit%',
        'description': 'Evaluates how effectively a player pulls the ball with power. '
                      'Upper right quadrant indicates pull-power hitters, '
                      'while lower left shows opposite-field/contact profiles. '
                      'High HardHit% with low Pull% suggests all-fields power.',
        'multiply_x': False,
        'multiply_y': False
    },
    'Plate Discipline': {
        'x_stat': 'O-Swing%',
        'y_stat': 'SwStr%', 
        'description': 'Measures plate discipline and contact ability. '
                      'O-Swing% shows chase rate on pitches outside zone. '
                      'SwStr% indicates overall swing-and-miss tendency. '
                      'Lower left quadrant represents elite plate discipline. '
                      'Upper right suggests aggressive, high-risk approaches.',
        'multiply_x': False,
        'multiply_y': False
    },
    'Power Production': {
        'x_stat': 'Barrel%',
        'y_stat': 'HardHit%',
        'description': 'Evaluates quality of contact and power potential. '
                      'Barrel% represents optimal launch angle and exit velocity. '
                      'HardHit% shows consistency of hard contact. '
                      'Upper right quadrant indicates premium power hitters. '
                      'High HardHit% with low Barrel% suggests line-drive hitters.',
        'multiply_x': True,
        'multiply_y': False
    },
    'Expected Production': {
        'x_stat': 'xwOBA',
        'y_stat': 'BABIP',
        'description': 'Identifies potential regression candidates. '
                      'xwOBA represents expected weighted on-base average. '
                      'BABIP shows batting average on balls in play. '
                      'High BABIP with low xwOBA suggests likely regression. '
                      'Low BABIP with high xwOBA indicates potential breakout.',
        'multiply_x': False,
        'multiply_y': False
    },
    'Plate Discipline Advanced': {
        'x_stat': 'Z-Swing%',
        'y_stat': 'O-Swing%',
        'description': 'Advanced look at swing decisions. '
                      'Z-Swing% shows aggression on pitches in strike zone. '
                      'O-Swing% indicates chase rate outside zone. '
                      'Upper left quadrant (high Z-Swing%, low O-Swing%) '
                      'represents optimal plate discipline.',
        'multiply_x': False,
        'multiply_y': False
    },
    'Contact Skills': {
        'x_stat': 'Contact%',
        'y_stat': 'K%',
        'description': 'Evaluates contact ability and strikeout tendency. '
                      'Contact% shows how often player makes contact when swinging. '
                      'K% represents strikeout rate. '
                      'Upper left quadrant (high Contact%, low K%) indicates '
                      'elite contact hitters. Lower right suggests high-risk profiles.',
        'multiply_x': False,
        'multiply_y': False
    },
    'Batted Ball Profile': {
        'x_stat': 'GB%',
        'y_stat': 'FB%',
        'description': 'Shows batted ball tendencies and approach. '
                      'GB% represents ground ball rate. '
                      'FB% shows fly ball rate. '
                      'Position indicates hitting style - power (high FB%), '
                      'speed (high GB%), or balanced approach. '
                      'Line drive rate (LD%) makes up the remainder.',
        'multiply_x': False,
        'multiply_y': False
    },
    'Hitting Approach': {
        'x_stat': 'Pull%',
        'y_stat': 'Oppo%',
        'description': 'Displays directional hitting tendencies. '
                      'Pull% shows pull-side contact rate. '
                      'Oppo% indicates opposite field rate. '
                      'Position reveals hitting approach - pull heavy, '
                      'opposite field focused, or all-fields. '
                      'Center% makes up the remainder.',
        'multiply_x': False,
        'multiply_y': False
    },
    'Power and Patience': {
        'x_stat': 'BB%',
        'y_stat': 'ISO',
        'description': 'Combines plate discipline with power output. '
                      'BB% represents walk rate. '
                      'ISO (Isolated Power) shows extra-base hit ability. '
                      'Upper right quadrant indicates elite power hitters '
                      'with good plate discipline. Lower left suggests '
                      'contact-focused approach.',
        'multiply_x': False,
        'multiply_y': False
    },
    'Contact Quality Advanced': {
        'x_stat': 'EV',
        'y_stat': 'Barrel%',
        'description': 'Advanced power metrics analysis. '
                      'EV (Exit Velocity) shows raw power potential. '
                      'Barrel% indicates optimal contact rate. '
                      'Upper right quadrant represents elite power hitters. '
                      'High EV with low Barrel% suggests raw power not '
                      'being fully optimized.',
        'multiply_x': False,
        'multiply_y': True
    }
}


def get_fantrax_free_agents():
    """
    Fetch and process Fantrax free agent data.

    Returns:
        tuple: (set of free agent names in lowercase, list of free agent details)
        
    Raises:
        requests.exceptions.RequestException: For network-related errors
        ValueError: For data validation errors
    """
    try:
        # Get player ID mapping data
        player_response = requests.get(
            PLAYER_IDS_URL, 
            params={'sport': FANTRAX_PARAMS['sport']},
            timeout=(3.05, 27)  # (connect timeout, read timeout)
        )
        player_response.raise_for_status()
        player_data = player_response.json()
        
        print(f"Retrieved {len(player_data)} player mappings")
        
        # Get league info
        league_response = requests.get(LEAGUE_INFO_URL, params=FANTRAX_PARAMS)
        league_response.raise_for_status()
        league_data = league_response.json()
        
        player_info = league_data.get('playerInfo', {})
        if not player_info:
            raise ValueError("No player info found in league data")
            
        print(f"Found {len(player_info)} players in league data")
        
        # Process free agents
        free_agents = []
        fa_names = set()
        current_season = datetime.now().year
        
        for player_id, league_player_data in player_info.items():
            if isinstance(league_player_data, dict) and league_player_data.get('status') == 'FA':
                if player_id in player_data:
                    player_details = player_data[player_id]
                    name = player_details.get('name', '')
                    
                    standardized_name = standardize_player_name(name)
                    if not standardized_name:
                        print(f"Warning: Could not standardize name for player ID {player_id}")
                        continue
                        
                    new_entry = {
                        'id': player_id,
                        'name': standardized_name,
                        'team': player_details.get('team', ''),
                        'position': player_details.get('position', ''),
                        'season': current_season,
                        **league_player_data
                    }
                    free_agents.append(new_entry)
                    fa_names.add(standardized_name.lower())
        
        print(f"Processed {len(free_agents)} free agents")
        return fa_names, free_agents
        
    except requests.exceptions.RequestException as e:
        print(f"Network error: {str(e)}")
        raise
    except ValueError as e:
        print(f"Data validation error: {str(e)}")
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise


def standardize_player_name(name: str) -> str:
    """Standardize player name format."""
    if ',' in name:
        lastname, firstname = name.split(',', 1)
        return f"{firstname.strip()} {lastname.strip()}"
    return name.strip()



def create_comparison_plot(plot_data, player1, player2, x_metric, y_metric, multiply_x, multiply_y):
    """
    Create a scatter plot comparing two players with quadrant analysis
    
    Parameters:
        plot_data (DataFrame): Baseball statistics for all players
        player1 (str): Name of first player to highlight
        player2 (str): Name of second player to highlight
        x_metric (str): Statistical metric for x-axis
        y_metric (str): Statistical metric for y-axis
        multiply_x (bool): Whether to convert x-axis values to percentages
        multiply_y (bool): Whether to convert y-axis values to percentages
    
    Returns:
        plotly.graph_objects.Figure: Interactive scatter plot with quadrant analysis
    
    Note:
        - Grey dots represent all players in dataset
        - Blue dot represents player1
        - Red dot represents player2
        - Dashed lines represent league median values
        - Quadrants are labeled with high/low combinations
    """
    # Calculate medians
    median_y = plot_data[y_metric].median()
    median_x = plot_data[x_metric].median()
    
    # Create plot
    fig = go.Figure()

    # Add background points
    fig.add_trace(go.Scatter(
        x=plot_data[x_metric],
        y=plot_data[y_metric],
        mode='markers',
        name='All Players',
        marker=dict(color='lightgrey', size=8),
        text=plot_data['Name'],
        hovertemplate="<b>%{text}</b><br>" +
                     f"{x_metric}: %{{x:.1f}}<br>" +
                     f"{y_metric}: %{{y:.1f}}<br>"
    ))

    # Add selected players
    for player_data, name, color in [(plot_data[plot_data['Name'] == player1], player1, 'rgb(64, 132, 244)'),
                                   (plot_data[plot_data['Name'] == player2], player2, 'rgb(244, 89, 89)')]:
        fig.add_trace(go.Scatter(
            x=player_data[x_metric],
            y=player_data[y_metric],
            mode='markers',
            name=name,
            marker=dict(
                color=color,
                size=12,
                line=dict(width=2, color='black')
            ),
            text=[name],
            hovertemplate="<b>%{text}</b><br>" +
                         f"{x_metric}: %{{x:.1f}}<br>" +
                         f"{y_metric}: %{{y:.1f}}<br>"
        ))

    # Calculate ranges for annotations
    x_range = max(plot_data[x_metric]) - min(plot_data[x_metric])
    y_range = max(plot_data[y_metric]) - min(plot_data[y_metric])

    # Update layout
    fig.update_layout(
        title=f"{y_metric} vs {x_metric} Quadrant Analysis",
        xaxis_title=x_metric,
        yaxis_title=y_metric,
        showlegend=True,
        height=800,
        annotations=[
            dict(
                x=median_x + (x_range * 0.25),
                y=median_y + (y_range * 0.25),
                text=f"High {x_metric} / High {y_metric}",
                showarrow=False
            ),
            dict(
                x=median_x - (x_range * 0.25),
                y=median_y + (y_range * 0.25),
                text=f"Low {x_metric} / High {y_metric}",
                showarrow=False
            ),
            dict(
                x=median_x + (x_range * 0.25),
                y=median_y - (y_range * 0.25),
                text=f"High {x_metric} / Low {y_metric}",
                showarrow=False
            ),
            dict(
                x=median_x - (x_range * 0.25),
                y=median_y - (y_range * 0.25),
                text=f"Low {x_metric} / Low {y_metric}",
                showarrow=False
            )
        ]
    )

    # Add quadrant lines
    fig.add_hline(y=median_y, line_dash="dash", line_color="gray")
    fig.add_vline(x=median_x, line_dash="dash", line_color="gray")

    return fig

def analyze_season_trends(stats_df, player_name):
    """
    Create visualization of player's performance trends across seasons
    
    Parameters:
        stats_df (DataFrame): Multi-season statistics
        player_name (str): Name of player to analyze
    
    Returns:
        Figure: Plotly figure with trend analysis
    """
    player_stats = stats_df[stats_df['Name'] == player_name]
    
    if player_stats.empty:
        return None
        
    fig = go.Figure()
    
    # Key metrics to track over time
    metrics = ['wRC+', 'OPS', 'ISO', 'BB%', 'K%', 'Barrel%', 'HardHit%']
    
    for metric in metrics:
        if metric in player_stats.columns:
            y_values = player_stats[metric]
            if metric in PERCENTAGE_COLUMNS:
                y_values = y_values * 100
                
            fig.add_trace(go.Scatter(
                x=player_stats['Season'],
                y=y_values,
                name=metric,
                mode='lines+markers'
            ))
    
    fig.update_layout(
        title=f"{player_name} - Career Trends",
        xaxis_title="Season",
        yaxis_title="Value",
        hovermode='x unified',
        height=600,
        showlegend=True
    )
    
    return fig

def calculate_season_deltas(stats_df, metric):
    """Calculate year-over-year changes in specified metric"""
    stats_df = stats_df.sort_values(['Name', 'Season'])
    stats_df[f'{metric}_YOY_Change'] = stats_df.groupby('Name')[metric].diff()
    return stats_df

def create_season_comparison_plot(stats_df, season1, season2, metric):
    """
    Create scatter plot comparing player performance between two seasons
    
    Parameters:
        stats_df (DataFrame): Multi-season statistics
        season1 (int): First season to compare
        season2 (int): Second season to compare
        metric (str): Statistical metric to compare
    
    Returns:
        Figure: Plotly figure with season comparison
    """
    season1_stats = stats_df[stats_df['Season'] == season1]
    season2_stats = stats_df[stats_df['Season'] == season2]
    
    # Merge seasons on player name
    comparison = pd.merge(
        season1_stats[['Name', metric]],
        season2_stats[['Name', metric]],
        on='Name',
        suffixes=(f'_{season1}', f'_{season2}')
    )
    
    fig = go.Figure()
    
    # Add diagonal line for reference
    max_val = max(comparison[f'{metric}_{season1}'].max(),
                 comparison[f'{metric}_{season2}'].max())
    min_val = min(comparison[f'{metric}_{season1}'].min(),
                 comparison[f'{metric}_{season2}'].min())
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        name='No Change'
    ))
    
    fig.add_trace(go.Scatter(
        x=comparison[f'{metric}_{season1}'],
        y=comparison[f'{metric}_{season2}'],
        mode='markers+text',
        text=comparison['Name'],
        textposition='top center',
        name='Players'
    ))
    
    fig.update_layout(
        title=f'{metric} Comparison: {season1} vs {season2}',
        xaxis_title=f'{season1} {metric}',
        yaxis_title=f'{season2} {metric}',
        height=800
    )
    
    return fig

def display_player_comparison(filtered, seasons_available):
    """
    Enhanced player comparison with season selection
    
    Args:
        filtered (pd.DataFrame): Filtered baseball statistics
        seasons_available (list): List of available seasons
        
    Raises:
        ValueError: If selected players are not found in dataset
        KeyError: If required statistical columns are missing
    """
    st.subheader("Player Comparison Tool")
    
    # Season selection
    selected_season = st.selectbox(
        'Select Season for Comparison',
        options=sorted(seasons_available, reverse=True)
    )
    
    season_data = filtered[filtered['Season'] == selected_season]
    
    # Player selection
    col1, col2 = st.columns(2)
    with col1:
        player1 = st.selectbox('Select Player 1', 
                             options=sorted(season_data['Name'].unique()), 
                             key='player1')
    with col2:
        player2 = st.selectbox('Select Player 2', 
                             options=sorted(season_data['Name'].unique()), 
                             key='player2')
    
    if player1 == player2:
        st.warning("Please select different players for comparison")
        return
    
    # Analysis selection
    selected_analysis = st.selectbox(
        "Select Analysis Type",
        options=list(STAT_COMBINATIONS.keys()),
        help="Choose which statistical relationship to examine"
    )

    # Display analysis description
    st.info(STAT_COMBINATIONS[selected_analysis]['description'])

    # Get metrics for selected analysis
    analysis_config = STAT_COMBINATIONS[selected_analysis]
    x_metric = analysis_config['x_stat']
    y_metric = analysis_config['y_stat']
    multiply_x = analysis_config['multiply_x']
    multiply_y = analysis_config['multiply_y']
    
    # Prepare plot data
    plot_data = filtered.copy()
    if multiply_x:
        plot_data[x_metric] = plot_data[x_metric] * 100
    if multiply_y:
        plot_data[y_metric] = plot_data[y_metric] * 100

    # Verify players exist in dataset
    player1_data = plot_data[plot_data['Name'] == player1]
    player2_data = plot_data[plot_data['Name'] == player2]
    
    if player1_data.empty or player2_data.empty:
        st.error("One or both selected players not found in the dataset")
        return

    # Create and display plot
    fig = create_comparison_plot(plot_data, player1, player2, x_metric, y_metric, multiply_x, multiply_y)
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_baseball_stats(start_year, end_year=None):
    """
    Fetch and process baseball stats for specified year range
    """
    if end_year is None:
        end_year = start_year
    
    all_seasons = []
    for year in range(start_year, end_year + 1):
        try:
            season_stats = pybaseball.batting_stats(year, qual=100)
            if season_stats.empty:
                st.warning(f"No data available for {year}")
                continue
            
            season_stats['Season'] = year
            season_stats.drop(['Age Rng', 'Dol'], axis=1, errors='ignore', inplace=True)
            all_seasons.append(season_stats)
            
        except Exception as e:
            st.error(f"Error fetching {year} data: {str(e)}")
            continue
    
    if not all_seasons:
        st.error("No data could be retrieved for any season")
        st.stop()
        
    combined_stats = pd.concat(all_seasons, ignore_index=True)
    return combined_stats


def process_free_agents(data, player_lookup):
    """Process league data to find free agents"""
    free_agents = []
    
    def process_players(data):
        if isinstance(data, dict):
            if 'status' in data and 'id' in data:
                if data['status'] == 'FA':
                    player_id = data['id']
                    if player_id in player_lookup:
                        player_entry = data.copy()
                        player_entry['name'] = player_lookup[player_id]
                        free_agents.append(player_entry)
            for value in data.values():
                process_players(value)
        elif isinstance(data, list):
            for item in data:
                process_players(item)

    process_players(data)
    return {player['name'].lower() for player in free_agents}, free_agents

def display_season_comparison(stats_df, start_year, end_year):
    """Display season comparison interface"""
    col1, col2 = st.columns(2)
    with col1:
        season1 = st.selectbox("Select First Season", 
                             sorted(stats_df['Season'].unique()), 
                             key='season1')
    with col2:
        season2 = st.selectbox("Select Second Season", 
                             sorted(stats_df['Season'].unique()), 
                             key='season2')
    
    metric = st.selectbox(
        "Select Metric to Compare",
        ['wRC+', 'OPS', 'ISO', 'BB%', 'K%', 'Barrel%', 'HardHit%']
    )
    
    fig = create_season_comparison_plot(stats_df, season1, season2, metric)
    st.plotly_chart(fig, use_container_width=True)

def display_yoy_changes(stats_df):
    """Display year-over-year changes analysis"""
    metric = st.selectbox(
        "Select Metric",
        ['wRC+', 'OPS', 'ISO', 'BB%', 'K%', 'Barrel%', 'HardHit%']
    )
    
    season = st.selectbox(
        "Select Season",
        sorted(stats_df['Season'].unique(), reverse=True)
    )
    
    delta_stats = calculate_season_deltas(stats_df, metric)
    improvers = delta_stats[delta_stats['Season'] == season].nlargest(
        20, f'{metric}_YOY_Change'
    )
    
    st.dataframe(
        improvers[['Name', metric, f'{metric}_YOY_Change', 'FA Status']],
        use_container_width=True
    )

def display_league_trends(stats_df):
    """Display league-wide trend analysis"""
    metrics = st.multiselect(
        "Select Metrics to Display",
        ['BB%', 'K%', 'GB%', 'FB%', 'Barrel%', 'HardHit%', 'wRC+'],
        default=['BB%', 'K%']
    )
    
    league_averages = stats_df.groupby('Season')[metrics].mean().reset_index()
    
    fig = go.Figure()
    for metric in metrics:
        fig.add_trace(go.Scatter(
            x=league_averages['Season'],
            y=league_averages[metric],
            name=metric,
            mode='lines+markers'
        ))
    
    fig.update_layout(
        title="MLB League-Wide Trends",
        xaxis_title="Season",
        yaxis_title="Value",
        height=600,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_current_stats(current_year_stats):
    """Display current season statistics"""
    st.header('Current Season Statistics')
    
    # Display options
    display_option = st.radio(
        'Select display option:',
        ['Top 20 Leaders by Category', 'Complete Dataset', 'Free Agents Only']
    )

    # Generate stat leaderboards
    stat_columns = [col for col in current_year_stats.columns 
                   if col not in ['IDfg', 'Name', 'Season', 'FA Status']]
    
    top_20_stats = {
        stat: current_year_stats.nlargest(20, stat)[['Name', stat, 'FA Status']].reset_index(drop=True)
        for stat in stat_columns
    }
    
    # Add index
    for stat in top_20_stats:
        top_20_stats[stat].index = top_20_stats[stat].index + 1

    # Layout config
    num_columns = 4
    stat_list = list(top_20_stats.keys())

    if display_option == 'Top 20 Leaders by Category':
        # Display top 20 players for each statistical category in a grid layout
        for i in range(0, len(stat_list), num_columns):
            cols = st.columns(num_columns)
            stats_in_row = stat_list[i:i + num_columns]
            
            for col, stat in zip(cols, stats_in_row):
                with col:
                    st.subheader(f'Top 20 - {stat}')
                    st.dataframe(top_20_stats[stat], height=400, use_container_width=True)

    elif display_option == 'Complete Dataset':
        st.subheader('Complete Dataset')
        st.dataframe(current_year_stats, use_container_width=True, hide_index=True)

    elif display_option == 'Free Agents Only':
        st.subheader("Available Free Agents")
        fa_only = current_year_stats[current_year_stats['FA Status'] == '🟢 Available']
        st.dataframe(
            fa_only.sort_values('wRC+', ascending=False),
            use_container_width=True,
            hide_index=True
        )


def main():
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    st.title('MLB Statistical Analysis')

    # Year range selection
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.selectbox(
            'Start Season',
            range(START_YEAR, CURRENT_YEAR + 1),
            index=len(range(START_YEAR, CURRENT_YEAR + 1)) - 1
        )
    with col2:
        end_year = st.selectbox(
            'End Season',
            range(start_year, CURRENT_YEAR + 1),
            index=len(range(start_year, CURRENT_YEAR + 1)) - 1
        )

    # Fetch and process baseball statistics
    with st.spinner('Fetching baseball statistics...'):
        stats_df = fetch_baseball_stats(start_year, end_year)
    
    st.success('Data loaded successfully!')

    # Filter for relevant statistical columns
    filtered = stats_df[[
        'IDfg', 'Name', 'Season', 'wRC+', 'OPS', 'BABIP', 'BABIP+', 'ISO',
        'O-Swing%', 'xwOBA', 'Pull%', 'SwStr%', 'EV', 'maxEV',
        'Barrel%', 'HardHit%', 'SB', 'BB%', 'K%', 'Z-Swing%', 
        'Contact%', 'Z-Contact%', 'Oppo%', 'GB%', 'FB%', 'LD%',
        'AVG', 'OBP', 'SLG'
    ]].copy()

    # Integrate free agent data
    fa_names, free_agents = get_fantrax_free_agents()
    filtered['IsFA'] = filtered['Name'].str.lower().isin(fa_names)
    filtered['FA Status'] = filtered['IsFA'].map({True: '🟢 Available', False: '🔴 Taken'})
    filtered = filtered.drop('IsFA', axis=1)

    # Convert percentages
    for col in PERCENTAGE_COLUMNS:
        if col in filtered.columns:
            filtered[col] = filtered[col] * 100

    # Sidebar navigation
    st.sidebar.title("Navigation")
    analysis_type = st.sidebar.radio(
        "Select Analysis Type",
        options=[
            "Current Season Stats",
            "Player Comparison",
            "Season Comparison",
            "Career Trends",
            "League Trends",
            "Year-over-Year Changes"
        ]
    )

    # Add description for each analysis type
    analysis_descriptions = {
        "Current Season Stats": "View current season statistics and leaders",
        "Player Comparison": "Compare two players across various metrics",
        "Season Comparison": "Compare league-wide statistics between seasons",
        "Career Trends": "Analyze player performance trends over time",
        "League Trends": "View league-wide trend analysis",
        "Year-over-Year Changes": "Identify biggest improvers and decliner"
    }
    
    st.sidebar.info(analysis_descriptions[analysis_type])

    # Display based on analysis type
    if analysis_type == "Current Season Stats":
        current_year_stats = filtered[filtered['Season'] == end_year]
        display_current_stats(current_year_stats)
        
    elif analysis_type == "Player Comparison":
        display_player_comparison(filtered, filtered['Season'].unique())
        
    elif analysis_type == "Season Comparison":
        display_season_comparison(filtered, start_year, end_year)
        
    elif analysis_type == "Career Trends":
        player = st.selectbox("Select Player", sorted(filtered['Name'].unique()))
        fig = analyze_season_trends(filtered, player)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
    elif analysis_type == "League Trends":
        display_league_trends(filtered)
        
    elif analysis_type == "Year-over-Year Changes":
        display_yoy_changes(filtered)

if __name__ == "__main__":
    main()
