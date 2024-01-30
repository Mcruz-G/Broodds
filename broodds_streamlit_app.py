import random
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib import style
import math

name_mapping = {
    'America' : 'América',
    'Atlas' : 'Atlas',
    'Atletico' : 'Atlético',
    'Cruz_Azul' : 'Cruz Azul',
    'FC_Juarez' : 'FC Juárez',
    'Guadalajara' : 'Guadalajara',
    'Leon' : 'León',
    'Mazatlan' : 'Mazatlán',
    'Monterrey' : 'Monterrey',
    'Necaxa' : 'Necaxa',
    'Pachuca' : 'Pachuca',
    'Puebla' : 'Puebla',
    'Pumas_UNAM' : 'UNAM',
    'Queretaro' : 'Querétaro',
    'Santos_Laguna' : 'Santos',
    'Tijuana' : 'Tijuana',
    'Toluca' : 'Toluca',
    'UANL' : 'UANL',
}

inverse_name_mapping = {v: k for k, v in name_mapping.items()}



def plot_metric(label, value, prefix="", suffix="", show_graph=False, color_graph="", col=None):
    fig = go.Figure()

    fig.add_trace(
        go.Indicator(
            value=value,
            gauge={"axis": {"visible": False}},
            number={
                "prefix": prefix,
                "suffix": suffix,
                "font.size": 28,
            },
            title={
                "text": label,
                "font": {"size": 24},
            },
        )
    )

    if show_graph:
        fig.add_trace(
            go.Scatter(
                y=random.sample(range(0, 101), 30),
                hoverinfo="skip",
                fill="tozeroy",
                fillcolor=color_graph,
                line={
                    "color": color_graph,
                },
            )
        )


    fig.update_xaxes(visible=False, fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True)
    fig.update_layout(
        # paper_bgcolor="lightgrey",
        margin=dict(t=30, b=0),
        showlegend=False,
        plot_bgcolor="black",
        height=100,
    )
    if not col:
        st.plotly_chart(fig, use_container_width=True)
    else:
        col.plotly_chart(fig, use_container_width=True)

def extract_results(df):
    result_categories = ['W', 'D', 'L']
    results = df.Result.value_counts(normalize=True)
    # turn results into a dictionary and handle the case if some of the categories are missing
    results_dict = {}
    if len(results) == 3:
        results_dict['W'] = results['W']
        results_dict['D'] = results['D']
        results_dict['L'] = results['L']
    
        return results_dict

    for category in result_categories:
        if category in results.index:
            results_dict[category] = results[category]
        else:
            results_dict[category] = 0
    return results_dict

def show_results_distribution(df, subheader=""):
    # Add smaller header
    st.subheader(subheader)
    
    column_1, column_2, column_3= st.columns(3)
    
    
    results_dict = extract_results(df)
    w = results_dict['W']
    d = results_dict['D']
    l = results_dict['L']
    with column_1:
        plot_metric("W", w*100, suffix="%")
    with column_2:
        plot_metric("D", d*100, suffix="%")
    with column_3:
        plot_metric("L", l*100, suffix="%") 

def show_match_sequence(results_df,over_line):

    colors = {'W': 'green', 'D': 'yellow', 'L': 'red'}
    results_df['Color'] = [colors[result] for result in results_df['Result']]
    results_df['Over'] = results_df[f'TotalGoals_>{over_line}'] 
    results_df['Both teams score'] = results_df['GF_>0 & GA_>0']

    column_1, column_2, column_3= st.columns(3)
    html_sequence = ''.join([f'<font color="{row["Color"]}" title="'
                            f"Date: {row['Date']} | "
                            f"GF: {row['GF']} | "
                            f"GA: {row['GA']} | "
                            f"xG: {row['xG']} | "
                            f"xGA: {row['xGA']} | "
                            f"TotalGoals: {row['TotalGoals']} | "
                            f"MetaEquipo: {row['MetaEquipo']} | "
                            f"Opponent: {row['Opponent']} | "
                            f"Over: {row['Over']} | "
                            f"Both teams score: {row['Both teams score']} | "
                            f"Venue: {row['Venue']} | "
                            f'" style="font-size: 30px;">{row["Result"]}</font>  ' for index, row in results_df.iterrows()])
    with column_2:
        st.markdown(html_sequence, unsafe_allow_html=True)

def show_overs_and_both_scores(df, over_line):
    column_1, column_2 = st.columns(2)
    summary = df[['GF_>0 & GA_>0',f'TotalGoals_>{over_line}']].sum() / len(df)
    with column_1:
        plot_metric("Both teams score", summary['GF_>0 & GA_>0']*100, suffix="%")
    with column_2:
        plot_metric("Over", summary[f'TotalGoals_>{over_line}']*100, suffix="%")

def show_historic_stats(df):
    column_1, column_2, column_3, column_4 = st.columns(4)
    with column_1:
        plot_metric("GF", df['GF'].mean(), suffix="")
    with column_2:
        plot_metric("GA", df['GA'].mean(), suffix="")
    with column_3:
        plot_metric("xG", df['xG'].mean(), suffix="")
    with column_4:
        plot_metric("xGA", df['xGA'].mean(), suffix="")

def plot_timeseries(df,col,plot_title, format_col=None, hline=None):
    df = df.sort_values(by='Date', ascending=True)
    df['MA_5'] = df[col].rolling(window=5).mean()
    timeseries = df.iloc[-10:]
    if col == 'xG':
        print(timeseries)

    # Set the background color to match the Streamlit theme
    plt.style.use('dark_background')

    # Plot the time series and the moving average
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.plot(timeseries['Date'], timeseries[col], label=col)
    ax.plot(timeseries['Date'], timeseries['MA_5'], label='5-SMA', linestyle='--', color='orange')
    if hline:
        ax.axhline(y=hline, color='red', linestyle='--', label='Over Line')
    ax.set_xlabel('Date')
    ax.set_ylabel(f'{col}')
    ax.set_title(plot_title, fontsize=7)
    ax.legend()
    # Adjust the size of y-axis ticks
    ax.tick_params(axis='x', labelsize=3)
    ax.tick_params(axis='y', labelsize=4)
    # Display the plot
    if format_col:
        format_col.pyplot(fig)
    else:
        st.pyplot(fig)

df = pd.read_csv("data/fullData.csv").iloc[:,1:]

if __name__ == "__main__":

    st.set_page_config(
        page_title="BroOdds LIGA MX Dashboard",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("<style> footer {visibility: hidden;} </style>", unsafe_allow_html=True)

    
    st.title("Welcome to BroOdds LIGA MX Dashboard")
    over_line = math.ceil(int(st.text_input("Over Line", 2)))
    col1, col2 = st.columns(2)
    


    metaequipos = df.MetaEquipo.unique().tolist()
    opponents = df.Opponent.unique().tolist()
    season_stages = df.SeasonStage.unique().tolist()

    # Add a selectbox to the sidebar:
    home_team = col1.selectbox(
        'Select Home Team',
        metaequipos
    )

    away_team = col2.selectbox(
        'Select Away Team',
        opponents
    )

    season_stages = st.multiselect(
    'Season Stages',
    season_stages,
    ['Apertura', 'Clausura'])

    with st.sidebar:
        selected = option_menu(
            menu_title="Hello BroOdder!",
            options=['Historic Match Results', 'Team Analysis', 'Goals Analysis',
                     'Home Team Goals Analysis', 'Away Team Goals Analysis'],
        )
        
    if selected == 'Historic Match Results':

        st.header("Match Analysis")
        # Historic Match Results
        historic_match_data = df[(df.MetaEquipo == home_team) & (df.Opponent == away_team) & (df.SeasonStage.isin(season_stages))].dropna(subset={'Result'}).sort_values(by='Date', ascending=True)
        show_results_distribution(historic_match_data, subheader=f"Historic {home_team} vs {away_team} Results")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        show_overs_and_both_scores(historic_match_data, over_line)

        historic_match_data_venue = df[(df.MetaEquipo == home_team) & (df.Opponent == away_team) & (df.Venue == 'Home')& (df.SeasonStage.isin(season_stages))].dropna(subset={'Result'}).sort_values(by='Date', ascending=True)
        show_results_distribution(historic_match_data_venue, subheader=f"Historic {home_team} vs {away_team} Results at {home_team}'s Venue")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        show_overs_and_both_scores(historic_match_data_venue, over_line)
        
        
        # Last 10 Match Results
        last_10_match_data = df[(df.MetaEquipo == home_team) & (df.Opponent == away_team)& (df.SeasonStage.isin(season_stages))].dropna(subset={'Result'}).sort_values(by='Date', ascending=True).tail(10)
        show_results_distribution(last_10_match_data, subheader=f"Last 10 {home_team} vs {away_team} Match Results")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        show_match_sequence(last_10_match_data, over_line)
        st.markdown("")
        st.markdown("")
        st.markdown("")
        show_overs_and_both_scores(last_10_match_data, over_line)

        
        # Last 10 Match Results in the Corresponding Venue
        last_10_match_data = df[(df.MetaEquipo == home_team) & (df.Opponent == away_team) & (df.Venue == 'Home')& (df.SeasonStage.isin(season_stages))].dropna(subset={'Result'}).sort_values(by='Date', ascending=True).tail(10)
        show_results_distribution(last_10_match_data, subheader=f"Last 10 {home_team} vs {away_team} Match Results in the {home_team}'s Venue")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        show_match_sequence(last_10_match_data, over_line)
        st.markdown("")
        st.markdown("")
        st.markdown("")
        show_overs_and_both_scores(last_10_match_data, over_line)

        st.subheader("Historic Match Stats")
        show_historic_stats(historic_match_data)
        st.subheader(f"Historic Match Stats at {home_team}'s Venue")
        show_historic_stats(historic_match_data_venue)
        st.subheader("Last 10 Match Stats")
        show_historic_stats(historic_match_data.sort_values(by='Date', ascending=True).tail(10))
        st.subheader(f"Last 10 Match Stats at {home_team}'s Venue")
        show_historic_stats(historic_match_data_venue.sort_values(by='Date', ascending=True).tail(10))
        st.markdown("---")

    if selected == 'Team Analysis':
        st.header(f"{home_team} Analysis")
        match_data = df[(df.MetaEquipo == home_team) & (df.Venue == 'Home')& (df.SeasonStage.isin(season_stages))].dropna(subset={'Result'}).sort_values(by='Date', ascending=True)
        show_results_distribution(match_data, subheader=f"{home_team}'s Historic Results at Home")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        show_overs_and_both_scores(match_data, over_line)
        
        # Last 10 home_team match results
        last_10_match_data = df[(df.MetaEquipo == home_team)& (df.SeasonStage.isin(season_stages))].dropna(subset={'Result'}).sort_values(by='Date', ascending=True).tail(10)
        show_results_distribution(last_10_match_data, subheader=f"Last 10 {home_team} Match Results")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        show_match_sequence(last_10_match_data, over_line)
        st.markdown("")
        st.markdown("")
        st.markdown("")
        show_overs_and_both_scores(last_10_match_data, over_line)
        st.markdown("")
        st.markdown("")
        st.markdown("")

        
        
        # Last 10 home_team match results in the corresponding venue
        last_10_match_data = df[(df.MetaEquipo == home_team) & (df.Venue == 'Home')& (df.SeasonStage.isin(season_stages))].dropna(subset={'Result'}).sort_values(by='Date', ascending=True).tail(10)
        show_results_distribution(last_10_match_data, subheader=f"Last 10 {home_team} Match Results at Home")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        show_match_sequence(last_10_match_data, over_line)
        st.markdown("")
        st.markdown("")
        st.markdown("")
        show_overs_and_both_scores(last_10_match_data, over_line)

        last_season_data = df[(df.MetaEquipo == home_team) & (df.SeasonStage.isin(['Apertura','Liguilla'])) & (df.Temporada == "2023-2024")].dropna(subset={'Result'}).sort_values(by='Date', ascending=True)
        show_results_distribution(last_season_data, subheader=f"Last Season {home_team} Match Results ")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        show_match_sequence(last_season_data, over_line)
        st.markdown("")
        st.markdown("")
        st.markdown("")
        show_overs_and_both_scores(last_season_data, over_line)

        last_season_venue_data = df[(df.MetaEquipo == home_team) & (df.Venue == 'Home')& (df.SeasonStage.isin(['Apertura','Liguilla'])) & (df.Temporada == "2023-2024")].dropna(subset={'Result'}).sort_values(by='Date', ascending=True)
        show_results_distribution(last_season_venue_data, subheader=f"Last Season {home_team} Match Results at Home")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        show_match_sequence(last_season_venue_data, over_line)
        st.markdown("")
        st.markdown("")
        st.markdown("")
        show_overs_and_both_scores(last_season_venue_data, over_line)

        st.header(f"{away_team} Analysis")
        match_data = df[(df.MetaEquipo == inverse_name_mapping[away_team]) & (df.Venue == 'Home')& (df.SeasonStage.isin(season_stages))].dropna(subset={'Result'}).sort_values(by='Date', ascending=True)
        show_results_distribution(match_data, subheader=f"{away_team}'s Historic Results Away")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        show_overs_and_both_scores(match_data, over_line)

        # Last 10 away_team match results
        last_10_match_data = df[(df.MetaEquipo == inverse_name_mapping[away_team])& (df.SeasonStage.isin(season_stages))].dropna(subset={'Result'}).sort_values(by='Date', ascending=True).tail(10)
        show_results_distribution(last_10_match_data, subheader=f"Last 10 {away_team} Match Results")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        show_match_sequence(last_10_match_data, over_line)
        st.markdown("")
        st.markdown("")
        st.markdown("")
        show_overs_and_both_scores(last_10_match_data, over_line)


        # Last 10 away_team match results in the corresponding venue
        last_10_match_data = df[(df.MetaEquipo == inverse_name_mapping[away_team]) & (df.Venue == 'Away')& (df.SeasonStage.isin(season_stages))].dropna(subset={'Result'}).sort_values(by='Date', ascending=True).tail(10)
        show_results_distribution(last_10_match_data, subheader=f"Last 10 {away_team} Match Results Away")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        show_match_sequence(last_10_match_data, over_line)
        st.markdown("")
        st.markdown("")
        st.markdown("")
        show_overs_and_both_scores(last_10_match_data, over_line)

        last_season_data = df[(df.MetaEquipo == inverse_name_mapping[away_team]) & (df.SeasonStage.isin(['Apertura','Liguilla'])) & (df.Temporada == "2023-2024")].dropna(subset={'Result'}).sort_values(by='Date', ascending=True)
        show_results_distribution(last_season_data, subheader=f"Last Season {away_team} Match Results ")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        show_match_sequence(last_season_data, over_line)
        st.markdown("")
        st.markdown("")
        st.markdown("")
        show_overs_and_both_scores(last_season_data, over_line)

        last_season_venue_data = df[(df.MetaEquipo == inverse_name_mapping[away_team]) & (df.Venue == 'Away')& (df.SeasonStage.isin(['Apertura','Liguilla'])) & (df.Temporada == "2023-2024")].dropna(subset={'Result'}).sort_values(by='Date', ascending=True)
        show_results_distribution(last_season_venue_data, subheader=f"Last Season {away_team} Match Results Away")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        show_match_sequence(last_season_venue_data, over_line)
        st.markdown("")
        st.markdown("")
        st.markdown("")
        show_overs_and_both_scores(last_season_venue_data, over_line)

    if selected == 'Home Team Goals Analysis':
        timeseries_data = df[(df.MetaEquipo == home_team)& (df.SeasonStage.isin(season_stages))].dropna(subset={'Result'}).sort_values(by='Date', ascending=True).tail(20)
        column_1, column_2 = st.columns(2)
        plot_timeseries(timeseries_data, "TotalGoals",f"{home_team}'s Games Total Goals Timeseries", column_1, hline=over_line)
        plot_timeseries(timeseries_data, "GF",f"{home_team}'s Goals in Favor Timeseries", column_2)
        plot_timeseries(timeseries_data, "GA",f"{home_team}'s Goals Against Timeseries", column_1)
        plot_timeseries(timeseries_data, "xG",f"{home_team}'s xG Timeseries", column_2)
        plot_timeseries(timeseries_data, "xGA",f"{home_team}'s xGA Timeseries",column_1)

        timeseries_data = df[(df.MetaEquipo == home_team) & (df.Venue == 'Home')& (df.SeasonStage.isin(season_stages))].dropna(subset={'Result'}).sort_values(by='Date', ascending=True).tail(20)
        plot_timeseries(timeseries_data, "TotalGoals",f"{home_team}'s Games Total Goals at Home Timeseries",column_1, hline=over_line)
        plot_timeseries(timeseries_data, "GF",f"{home_team}'s Goals in Favor at Home Timeseries",column_2)
        plot_timeseries(timeseries_data, "GA",f"{home_team}'s Goals Against at Home Timeseries",column_1)
        plot_timeseries(timeseries_data, "xG",f"{home_team}'s xG at Home Timeseries",column_2)
        plot_timeseries(timeseries_data, "xGA",f"{home_team}'s xGA at Home Timeseries", column_2)

    if selected == 'Away Team Goals Analysis':
        column_1, column_2 = st.columns(2)

        timeseries_data = df[(df.MetaEquipo == inverse_name_mapping[away_team])& (df.SeasonStage.isin(season_stages))].dropna(subset={'Result'}).sort_values(by='Date', ascending=True).tail(20)
        plot_timeseries(timeseries_data, "TotalGoals",f"{away_team}'s Games Total Goals Timeseries",column_1, hline=over_line)
        plot_timeseries(timeseries_data, "GF",f"{away_team}'s Goals in Favor Timeseries", column_2)
        plot_timeseries(timeseries_data, "GA",f"{away_team}'s Goals Against Timeseries", column_1)
        plot_timeseries(timeseries_data, "xG",f"{away_team}'s xG Timeseries", column_2)
        plot_timeseries(timeseries_data, "xGA",f"{away_team}'s xGA Timeseries", column_1)

        timeseries_data = df[(df.MetaEquipo == inverse_name_mapping[away_team]) & (df.Venue == 'Away')& (df.SeasonStage.isin(season_stages))].dropna(subset={'Result'}).sort_values(by='Date', ascending=True).tail(20)
        plot_timeseries(timeseries_data, "TotalGoals",f"{away_team}'s Games Total Goals Away Timeseries", column_1, hline=over_line)
        plot_timeseries(timeseries_data, "GF",f"{away_team}'s Goals in Favor Away Timeseries",  column_2)
        plot_timeseries(timeseries_data, "GA",f"{away_team}'s Goals Against Away Timeseries", column_1)
        plot_timeseries(timeseries_data, "xG",f"{away_team}'s xG Away Timeseries", column_2)
        plot_timeseries(timeseries_data, "xGA",f"{away_team}'s xGA Away Timeseries", column_2)
        st.markdown("---")

    if selected == 'Goals Analysis':
        column_1, column_2 = st.columns(2)

        timeseries_data = df[(df.MetaEquipo == home_team) & (df.Opponent == away_team)& (df.SeasonStage.isin(season_stages))].dropna(subset={'Result'}).sort_values(by='Date', ascending=True).tail(20)
        plot_timeseries(timeseries_data, "TotalGoals",f"{home_team}'s Games Total Goals Against {away_team} Timeseries", column_1, hline=over_line)
        plot_timeseries(timeseries_data, "GF",f"{home_team}'s Goals in Favor Against {away_team} Timeseries", column_2)
        plot_timeseries(timeseries_data, "GA",f"{home_team}'s Goals Against Against {away_team} Timeseries", column_1)
        plot_timeseries(timeseries_data, "xG",f"{home_team}'s xG Against {away_team} Timeseries", column_2)
        plot_timeseries(timeseries_data, "xGA",f"{home_team}'s xGA Against {away_team} Timeseries", column_1)

        timeseries_data = df[(df.MetaEquipo == home_team) & (df.Opponent == away_team) & (df.Venue == 'Home')& (df.SeasonStage.isin(season_stages))].dropna(subset={'Result'}).sort_values(by='Date', ascending=True).tail(20)
        print(timeseries_data.shape)
        plot_timeseries(timeseries_data, "TotalGoals",f"{home_team}'s Games Total Goals Against {away_team} at Home Timeseries", column_1, hline=over_line)
        plot_timeseries(timeseries_data, "GF",f"{home_team}'s Goals in Favor Against {away_team} at Home Timeseries", column_2)
        plot_timeseries(timeseries_data, "GA",f"{home_team}'s Goals Against Against {away_team} at Home Timeseries", column_1)
        plot_timeseries(timeseries_data, "xG",f"{home_team}'s xG Against {away_team} at Home Timeseries", column_2)
        plot_timeseries(timeseries_data, "xGA",f"{home_team}'s xGA Against {away_team} at Home Timeseries", column_2)


    
    # You can add more content below the columns
    st.markdown("---")
    st.subheader("Additional Content Below Columns")
    st.write("You can add more content below the columns.")