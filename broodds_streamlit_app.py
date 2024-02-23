import random
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib import style
import math
import numpy as np 

from matplotlib.offsetbox import (OffsetImage,AnnotationBbox)
from mplsoccer import FontManager, add_image
from PIL import Image
import os


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

image_name_mapping = {
    'America' : 'america',
    'Atlas' : 'atlas',
    'Atletico' : 'atleticosl',
    'Cruz_Azul' : 'cruzazul',
    'FC_Juarez' : 'juarez',
    'Guadalajara' : 'guadalajara',
    'Leon' : 'leon',
    'Mazatlan' : 'mazatlan',
    'Monterrey' : 'monterrey',
    'Necaxa' : 'necaxa',
    'Pachuca' : 'pachuca',
    'Puebla' : 'puebla',
    'Pumas_UNAM' : 'pumas',
    'Queretaro' : 'queretaro',
    'Santos_Laguna' : 'santos',
    'Tijuana' : 'tijuana',
    'Toluca' : 'toluca',
    'UANL' : 'tigres',
}

#Escudos
def getImage(path, zoom=0.1):
    return OffsetImage(plt.imread(path), zoom=zoom)

def load_parameters():
    URL3 = 'https://github.com/VanillaandCream/Catamaran-Tamil/blob/master/Fonts/Catamaran-Medium.ttf?raw=true'
    catamaran2 = FontManager(URL3)
    URL = 'https://github.com/google/fonts/blob/main/ofl/fjallaone/FjallaOne-Regular.ttf?raw=true'
    robotto_regular = FontManager(URL)
    URL2 = 'https://github.com/VanillaandCream/Catamaran-Tamil/blob/master/Fonts/Catamaran-ExtraBold.ttf?raw=true'
    catamaran = FontManager(URL2)
    URL4 = 'https://github.com/google/fonts/blob/main/ofl/bungeeinline/BungeeInline-Regular.ttf?raw=true'
    titulo = FontManager(URL4)
    return catamaran, catamaran2, robotto_regular, titulo


def make_scatter_team_plot(plot_data, xcolumn, ycolumn, title
                           , xlabel='Atributo 1'
                           , ylabel='Atributo 2'
                           , facecolor='#292323', color_plot='white'
                           , tournament='Liga MX'
                           , zoom = 0.07):
    path = os.getcwd()
    files = os.listdir('images/ligamx/')

    #Complete the code 
    catamaran, catamaran2, robotto_regular, titulo = load_parameters()
    #Seteo los parametros del scatter
    print(plot_data.columns)
    x, promx = plot_data[xcolumn], plot_data[xcolumn].mean() 
    y, promy = plot_data[ycolumn], plot_data[ycolumn].mean()

    #Figura, axis y el scatter
    fig, ax = plt.subplots(figsize =(20, 12))
    ax.scatter(x, y)

    #Detalles esteticos
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(color_plot)
    ax.spines['left'].set_color(color_plot)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
    ax.yaxis.set_tick_params(pad = 12,colors=color_plot,labelsize=12)
    ax.xaxis.set_tick_params(pad = 12,colors=color_plot,labelsize=12)

    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)
    ax.grid(aa=True, color =color_plot,linestyle ='-', linewidth = 0,alpha = 0.5)
    ax.grid(which='minor', color =color_plot,linestyle ='-', linewidth = 1.5, alpha=1)

    #Lineas de Promedio
    ax.axvline(promx, color=color_plot)
    ax.axhline(promy, color=color_plot)

 
    ax.set_xlabel(xlabel, fontproperties=robotto_regular.prop,fontsize=18,color=color_plot)
    ax.set_ylabel(ylabel, fontproperties=robotto_regular.prop,fontsize=18,color=color_plot)
    ax.set_title(f'{xlabel} Vs {ylabel} - {tournament}', fontproperties=robotto_regular.prop,
                        loc ='left', color=color_plot,fontsize = 25,fontweight="bold", pad=20)

    for index, row in plot_data.iterrows():
        x0, y0 = row[xcolumn], row[ycolumn]
        team = row['MetaEquipo']
        img_team_name = image_name_mapping[team]
        ab = AnnotationBbox(getImage(f'images/ligamx/{img_team_name}.png',zoom=zoom), (x0, y0), frameon=False)
        ax.add_artist(ab)

    st.pyplot(fig)



# Function to create a gradient color based on a value
def color_gradient(val, df, reference_col, reverse_color_scale=False):
    if reverse_color_scale:
        val *= -1

    min_val = df[[reference_col]].min() 
    max_val = df[[reference_col]].max() 
    normalized_val = (val - min_val) / (max_val - min_val)
    r = int(100 * (1 - normalized_val))
    g = int(100 * normalized_val)
    b = 0
    return f'background-color: rgb({r},{g},{b})'


def plot_pie_chart(results_dict, colors, title=None):
    # Create a dark background
    # Color mapping for each result type
    
    # Create a pie chart
    labels = list(results_dict.keys())
    sizes = list(results_dict.values())

    fig, ax = plt.subplots()

    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=[colors[result] for result in labels],
            wedgeprops={'edgecolor': 'white'},  # White edge color
        textprops={'color': 'white', 'size':15}); # White text color

    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.legend(labels)
    # Modify the background color of the pie chart image

    fig.patch.set_facecolor('black')
    # Display the pie chart in Streamlit
    st.pyplot(fig)

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
    column_1, column_2, column_3 = st.columns(3)
    with column_2:
        results_dict = {'GF > 0' : df[df.GF > 0].shape[0] / df.shape[0], 'GF = 0': 1 - df[df.GF > 0].shape[0] / df.shape[0]}
        colors = {'GF > 0': '#4CAF50', 'GF = 0': '#F44336'}
        plot_pie_chart(results_dict, colors)

    with column_3:
        results_dict = {'GA > 0' : df[df.GA > 0].shape[0] / df.shape[0], 'GA = 0': 1 - df[df.GA > 0].shape[0] / df.shape[0]}
        colors = {'GA > 0': '#4CAF50', 'GA = 0': '#F44336'}
        plot_pie_chart(results_dict, colors)
        
    with column_1:
    # Add smaller header
        colors = {'W': '#4CAF50', 'L': '#F44336', 'D': '#FFC107'}

        results_dict = extract_results(df)
        plot_pie_chart(results_dict, colors, subheader)
    

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
        results_dict = {'Ambos Anotan': summary['GF_>0 & GA_>0']*100, "No-Ambos Anotan" : 100 -summary['GF_>0 & GA_>0']*100} 
        colors = {'Ambos Anotan': '#4CAF50', 'No-Ambos Anotan': '#F44336'}

        plot_pie_chart(results_dict,  colors, 'Ambos Anotan')
    with column_2:
        colors = {'Over': '#4CAF50', 'Under': '#F44336'}

        results_dict = {'Over':summary[f'TotalGoals_>{over_line}']*100, 'Under': 100 - summary[f'TotalGoals_>{over_line}']*100}
        plot_pie_chart(results_dict, colors, 'Over')

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

def plot_timeseries(df,col,plot_title, format_col=None, hline=None, ma=True):
    df = df.sort_values(by='Date', ascending=True)
    if ma:
        df['MA_5'] = df[col].rolling(window=5).mean()
    timeseries = df.iloc[-10:]
    if col == 'xG':
        print(timeseries)

    # Set the background color to match the Streamlit theme
    plt.style.use('dark_background')

    # Plot the time series and the moving average
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.plot(timeseries['Date'], timeseries[col], label=col)
    if ma:
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

def set_streamlit_config():
    st.set_page_config(
        page_title="BroOdds LIGA MX Dashboard",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("<style> footer {visibility: hidden;} </style>", unsafe_allow_html=True)



df = pd.read_csv("data/scores_and_fixtures.csv").iloc[:,1:]

if __name__ == "__main__":

    # load_dotenv()
    set_streamlit_config()
    column_1, column_2, column_3 = st.columns(3)
    column_2.title("Welcome to BroOdds LIGA MX Dashboard")

    #Inputs
    
    #Overline
    over_line = math.ceil(int(st.text_input("Over Line", 2)))
    # Function to apply color based on logic
    def highlight_cells(x):
        color1 = 'red' if x['Result'] == 'L' else 'green' if x['Result'] == 'W' else 'yellow'
        color2 = 'red' if x['GF_>0 & GA_>0'] == 0 else 'green'
        color3 = 'red' if x[f'TotalGoals_>{over_line}'] == 0 else 'green'
        styles = [f'background-color: {color1}', f'background-color: {color2}', f'background-color: {color3}']
        return styles
    
    #Teams
    col1, col2 = st.columns(2)
    metaequipos = df.MetaEquipo.unique().tolist()
    opponents = df.Opponent.unique().tolist()
    # Add a selectbox to the sidebar:
    home_team = col1.selectbox(
        'Select Home Team',
        metaequipos
    )

    away_team = col2.selectbox(
        'Select Away Team',
        opponents
    )

    #Season Stage
    season_stages = df.SeasonStage.unique().tolist()
    season_stages = st.multiselect(
                                    'Season Stages',
                                    season_stages,
                                    season_stages)
    
    # SideBar
    # Display the logo using st.image()

    with st.sidebar:
        selected = option_menu(
            menu_title="Hello BroOdder!",
            options=['Historic Match Results', 'Team Analysis', 'Goals Analysis',
                     'Home Team Goals Analysis', 'Away Team Goals Analysis', 
                     'Season Analysis', '@Broodds Visuals'],
        )
        
    if selected == 'Historic Match Results':

        st.header("Match Analysis")

        # Historic Match Results
        st.subheader(f"Historic {home_team} vs {away_team} Results")
        historic_match_data = df[(df.MetaEquipo == home_team) & (df.Opponent == away_team) & (df.SeasonStage.isin(season_stages))].dropna(subset={'Result'}).sort_values(by='Date', ascending=True)
        column_1, column_2 = st.columns(2)
        with column_1:
            show_results_distribution(historic_match_data, subheader=f"Historic {home_team} vs {away_team} Results")
        with column_2:
            show_overs_and_both_scores(historic_match_data, over_line)

        st.markdown("")
        st.markdown("")
        st.markdown("")
        
        show_df = historic_match_data[['Date','MetaEquipo','Opponent','Venue', 'Result', 'GF', 'GA', 'xG', 'xGA', 'SeasonStage','GF_>0 & GA_>0',f'TotalGoals_>{over_line}']].sort_values(by='Date', ascending=False)

        st.dataframe(show_df.style.apply(lambda x: highlight_cells(x), axis=1, subset=['Result', 'GF_>0 & GA_>0', f'TotalGoals_>{over_line}']))
        
        # Historic Match Results at Venue
        st.subheader(f"Historic {home_team} vs {away_team} Results at {home_team}'s Venue")
        historic_match_data_venue = df[(df.MetaEquipo == home_team) & (df.Opponent == away_team) & (df.Venue == 'Home')& (df.SeasonStage.isin(season_stages))].dropna(subset={'Result'}).sort_values(by='Date', ascending=True)
        column_1, column_2 = st.columns(2)
        with column_1:
            show_results_distribution(historic_match_data_venue, subheader=f"Historic {home_team} vs {away_team} Results at {home_team}'s Venue")
        with column_2:
            show_overs_and_both_scores(historic_match_data_venue, over_line)
        st.markdown("")
        st.markdown("")
        st.markdown("")
        show_df = historic_match_data_venue[['Date','MetaEquipo','Opponent','Venue', 'Result', 'GF', 'GA', 'xG', 'xGA', 'SeasonStage','GF_>0 & GA_>0',f'TotalGoals_>{over_line}']].sort_values(by='Date', ascending=False)
        st.dataframe(show_df.style.apply(lambda x: highlight_cells(x), axis=1, subset=['Result', 'GF_>0 & GA_>0', f'TotalGoals_>{over_line}']))
        
        # Last 10 Match Results
        st.subheader(f"Last 10 {home_team} vs {away_team} Match Results")
        last_10_match_data = df[(df.MetaEquipo == home_team) & (df.Opponent == away_team)& (df.SeasonStage.isin(season_stages))].dropna(subset={'Result'}).sort_values(by='Date', ascending=True).tail(10)
        column_1, column_2 = st.columns(2)

        with column_1:
            show_results_distribution(last_10_match_data, subheader=f"Last 10 {home_team} vs {away_team} Match Results")
        with column_2:
            show_overs_and_both_scores(last_10_match_data, over_line)
            
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        show_df = last_10_match_data[['Date','MetaEquipo','Opponent','Venue', 'Result', 'GF', 'GA', 'xG', 'xGA', 'SeasonStage','GF_>0 & GA_>0',f'TotalGoals_>{over_line}']].sort_values(by='Date', ascending=False)
        
        
        st.dataframe(show_df.style.apply(lambda x: highlight_cells(x), axis=1, subset=['Result', 'GF_>0 & GA_>0', f'TotalGoals_>{over_line}']))
        
        # Last 10 Match Results in the Corresponding Venue
        st.subheader(f"Last 10 {home_team} vs {away_team} Match Results in the {home_team}'s Venue")
        last_10_match_data = df[(df.MetaEquipo == home_team) & (df.Opponent == away_team) & (df.Venue == 'Home')& (df.SeasonStage.isin(season_stages))].dropna(subset={'Result'}).sort_values(by='Date', ascending=True).tail(10)
        column_1, column_2 = st.columns(2)

        with column_1:
            show_results_distribution(last_10_match_data, subheader=f"Last 10 {home_team} vs {away_team} Match Results in the {home_team}'s Venue")
        with column_2:
            show_overs_and_both_scores(last_10_match_data, over_line)
            
        # show_match_sequence(last_10_match_data, over_line)
        show_df = last_10_match_data[['Date','MetaEquipo','Opponent','Venue', 'Result', 'GF', 'GA', 'xG', 'xGA', 'SeasonStage','GF_>0 & GA_>0',f'TotalGoals_>{over_line}']].sort_values(by='Date', ascending=False)
        st.dataframe(show_df.style.apply(lambda x: highlight_cells(x), axis=1, subset=['Result', 'GF_>0 & GA_>0', f'TotalGoals_>{over_line}']))

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
        st.subheader(f"{home_team}'s Historic Results at Home")
        show_results_distribution(match_data, subheader=f"{home_team}'s Historic Results at Home")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        show_overs_and_both_scores(match_data, over_line)
        
        show_df = match_data[['Date','MetaEquipo','Opponent','Venue', 'Result', 'GF', 'GA', 'xG', 'xGA', 'SeasonStage','GF_>0 & GA_>0',f'TotalGoals_>{over_line}']].sort_values(by='Date', ascending=False)
        st.dataframe(show_df.style.apply(lambda x: highlight_cells(x), axis=1, subset=['Result', 'GF_>0 & GA_>0', f'TotalGoals_>{over_line}']))

        # Last 10 home_team match results
        last_10_match_data = df[(df.MetaEquipo == home_team)& (df.SeasonStage.isin(season_stages))].dropna(subset={'Result'}).sort_values(by='Date', ascending=True).tail(10)
        st.subheader(f"Last 10 {home_team} Match Results")
        show_results_distribution(last_10_match_data, subheader=f"Last 10 {home_team} Match Results")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        show_overs_and_both_scores(last_10_match_data, over_line)
        st.markdown("")
        st.markdown("")
        st.markdown("")
        show_df = last_10_match_data[['Date','MetaEquipo','Opponent','Venue', 'Result', 'GF', 'GA', 'xG', 'xGA', 'SeasonStage','GF_>0 & GA_>0',f'TotalGoals_>{over_line}']].sort_values(by='Date', ascending=False)
        st.dataframe(show_df.style.apply(lambda x: highlight_cells(x), axis=1, subset=['Result', 'GF_>0 & GA_>0', f'TotalGoals_>{over_line}']))


        
        
        # Last 10 home_team match results in the corresponding venue
        st.subheader(f"Last 10 {home_team} Match Results at Home")
        last_10_match_data = df[(df.MetaEquipo == home_team) & (df.Venue == 'Home')& (df.SeasonStage.isin(season_stages))].dropna(subset={'Result'}).sort_values(by='Date', ascending=True).tail(10)
        show_results_distribution(last_10_match_data, subheader=f"Last 10 {home_team} Match Results at Home")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        show_overs_and_both_scores(last_10_match_data, over_line)
        show_df = last_10_match_data[['Date','MetaEquipo','Opponent','Venue', 'Result', 'GF', 'GA', 'xG', 'xGA', 'SeasonStage','GF_>0 & GA_>0',f'TotalGoals_>{over_line}']].sort_values(by='Date', ascending=False)
        st.dataframe(show_df.style.apply(lambda x: highlight_cells(x), axis=1, subset=['Result', 'GF_>0 & GA_>0', f'TotalGoals_>{over_line}']))


        st.subheader(f"Last Season {home_team} Match Results ")
        last_season_data = df[(df.MetaEquipo == home_team) & (df.SeasonStage.isin(['Apertura','Liguilla'])) & (df.Temporada == "2023-2024")].dropna(subset={'Result'}).sort_values(by='Date', ascending=True)
        show_results_distribution(last_season_data, subheader=f"Last Season {home_team} Match Results ")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        show_overs_and_both_scores(last_season_data, over_line)
        show_df = last_season_data[['Date','MetaEquipo','Opponent','Venue', 'Result', 'GF', 'GA', 'xG', 'xGA', 'SeasonStage','GF_>0 & GA_>0',f'TotalGoals_>{over_line}']].sort_values(by='Date', ascending=False)
        st.dataframe(show_df.style.apply(lambda x: highlight_cells(x), axis=1, subset=['Result', 'GF_>0 & GA_>0', f'TotalGoals_>{over_line}']))


        st.subheader(f"Last Season {home_team} Match Results at Home")
        last_season_venue_data = df[(df.MetaEquipo == home_team) & (df.Venue == 'Home')& (df.SeasonStage.isin(['Apertura','Liguilla'])) & (df.Temporada == "2023-2024")].dropna(subset={'Result'}).sort_values(by='Date', ascending=True)
        show_results_distribution(last_season_venue_data, subheader=f"Last Season {home_team} Match Results at Home")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        show_overs_and_both_scores(last_season_venue_data, over_line)
        show_df = last_season_venue_data[['Date','MetaEquipo','Opponent','Venue', 'Result', 'GF', 'GA', 'xG', 'xGA', 'SeasonStage','GF_>0 & GA_>0',f'TotalGoals_>{over_line}']].sort_values(by='Date', ascending=False)
        st.dataframe(show_df.style.apply(lambda x: highlight_cells(x), axis=1, subset=['Result', 'GF_>0 & GA_>0', f'TotalGoals_>{over_line}']))



        st.header(f"{away_team} Analysis")

        st.subheader(f"{away_team}'s Historic Results Away")
        match_data = df[(df.MetaEquipo == inverse_name_mapping[away_team]) & (df.Venue == 'Away')& (df.SeasonStage.isin(season_stages))].dropna(subset={'Result'}).sort_values(by='Date', ascending=True)
        show_results_distribution(match_data, subheader=f"{away_team}'s Historic Results Away")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        show_overs_and_both_scores(match_data, over_line)
        show_df = match_data[['Date','MetaEquipo','Opponent','Venue', 'Result', 'GF', 'GA', 'xG', 'xGA', 'SeasonStage','GF_>0 & GA_>0',f'TotalGoals_>{over_line}']].sort_values(by='Date', ascending=False)
        st.dataframe(show_df.style.apply(lambda x: highlight_cells(x), axis=1, subset=['Result', 'GF_>0 & GA_>0', f'TotalGoals_>{over_line}']))

        # Last 10 away_team match results
        last_10_match_data = df[(df.MetaEquipo == inverse_name_mapping[away_team])& (df.SeasonStage.isin(season_stages))].dropna(subset={'Result'}).sort_values(by='Date', ascending=True).tail(10)
        st.subheader(f"Last 10 {away_team} Match Results")
        show_results_distribution(last_10_match_data, subheader=f"Last 10 {away_team} Match Results")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        show_overs_and_both_scores(last_10_match_data, over_line)
        show_df = last_10_match_data[['Date','MetaEquipo','Opponent','Venue', 'Result', 'GF', 'GA', 'xG', 'xGA', 'SeasonStage','GF_>0 & GA_>0',f'TotalGoals_>{over_line}']].sort_values(by='Date', ascending=False)
        st.dataframe(show_df.style.apply(lambda x: highlight_cells(x), axis=1, subset=['Result', 'GF_>0 & GA_>0', f'TotalGoals_>{over_line}']))

        # Last 10 away_team match results in the corresponding venue
        st.subheader(f"Last 10 {away_team} Match Results Away")
        last_10_match_data = df[(df.MetaEquipo == inverse_name_mapping[away_team]) & (df.Venue == 'Away')& (df.SeasonStage.isin(season_stages))].dropna(subset={'Result'}).sort_values(by='Date', ascending=True).tail(10)
        show_results_distribution(last_10_match_data, subheader=f"Last 10 {away_team} Match Results Away")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        show_overs_and_both_scores(last_10_match_data, over_line)
        show_df = last_10_match_data[['Date','MetaEquipo','Opponent','Venue', 'Result', 'GF', 'GA', 'xG', 'xGA', 'SeasonStage','GF_>0 & GA_>0',f'TotalGoals_>{over_line}']].sort_values(by='Date', ascending=False)
        st.dataframe(show_df.style.apply(lambda x: highlight_cells(x), axis=1, subset=['Result', 'GF_>0 & GA_>0', f'TotalGoals_>{over_line}']))

        st.subheader(f"Last Season {away_team} Match Results ")
        last_season_data = df[(df.MetaEquipo == inverse_name_mapping[away_team]) & (df.SeasonStage.isin(['Apertura','Liguilla'])) & (df.Temporada == "2023-2024")].dropna(subset={'Result'}).sort_values(by='Date', ascending=True)
        show_results_distribution(last_season_data, subheader=f"Last Season {away_team} Match Results ")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        show_overs_and_both_scores(last_season_data, over_line)
        show_df = last_season_data[['Date','MetaEquipo','Opponent','Venue', 'Result', 'GF', 'GA', 'xG', 'xGA', 'SeasonStage','GF_>0 & GA_>0',f'TotalGoals_>{over_line}']].sort_values(by='Date', ascending=False)
        st.dataframe(show_df.style.apply(lambda x: highlight_cells(x), axis=1, subset=['Result', 'GF_>0 & GA_>0', f'TotalGoals_>{over_line}']))

        st.subheader(f"Last Season {away_team} Match Results Away")
        last_season_venue_data = df[(df.MetaEquipo == inverse_name_mapping[away_team]) & (df.Venue == 'Away')& (df.SeasonStage.isin(['Apertura','Liguilla'])) & (df.Temporada == "2023-2024")].dropna(subset={'Result'}).sort_values(by='Date', ascending=True)
        show_results_distribution(last_season_venue_data, subheader=f"Last Season {away_team} Match Results Away")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        show_overs_and_both_scores(last_season_venue_data, over_line)
        show_df = last_season_venue_data[['Date','MetaEquipo','Opponent','Venue', 'Result', 'GF', 'GA', 'xG', 'xGA', 'SeasonStage','GF_>0 & GA_>0',f'TotalGoals_>{over_line}']].sort_values(by='Date', ascending=False)
        st.dataframe(show_df.style.apply(lambda x: highlight_cells(x), axis=1, subset=['Result', 'GF_>0 & GA_>0', f'TotalGoals_>{over_line}']))

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


    if selected == 'Season Analysis':
        st.markdown("---")
        st.subheader("Positions Table")
        temporadas = df.Temporada.unique().tolist()[::-1]
        season_stages = ['Apertura','Clausura']
        jornadas = list(range(1,19))

        # Add a selectbox to the sidebar:
        temporada = st.selectbox(
            'Select Season',
            temporadas
        )
        # Add a selectbox to the sidebar:
        stage = st.selectbox(
            'Select Season Stage',
            season_stages
        )

        jornada = st.selectbox(
            'Select Jornada',
            jornadas
        )
        
        columns = ['Date','Temporada','MetaEquipo','ranking','current_points','current_goals','current_goals_against','current_goals_difference','current_wins','current_losses','current_draws']
        st.dataframe(df[(df.Temporada == temporada) & (df.SeasonStage == stage) & (df.Jornada == jornada)][columns].sort_values(by='ranking'))
        
        st.markdown("---")
        column_1, column_2, column_3 = st.columns(3)

        column_2.subheader("Analysis by Jornada")
        column_1, column_2 = st.columns(2)
        
        columns = ['Date','Temporada','SeasonStage','MetaEquipo','Opponent','Venue','Result','GF','GA','ranking']
        data = df[(df.MetaEquipo == home_team) & (df.Jornada == jornada) & (df.Temporada >= "2021-2022") ][columns].sort_values(by='Date')
        plot_timeseries(data, col='ranking',plot_title=f"{home_team}'s ranking performance along Jornada {jornada}",format_col=column_1, hline=None, ma=False)
        column_1.dataframe(data)
        
        
        data = df[(df.MetaEquipo == inverse_name_mapping[away_team]) & (df.Jornada == jornada) & (df.Temporada >= "2021-2022") ][columns].sort_values(by='Date')
        plot_timeseries(data, col='ranking',plot_title=f"{away_team}'s ranking performance along Jornada {jornada}",format_col=column_2, hline=None, ma=False)
        column_2.dataframe(data)

        st.markdown('---')
        column_1, column_2 = st.columns(2)

        columns = ['MetaEquipo', 'ranking',
                   'current_goals','current_exp_goals',
                   ]

        data = df[(df.Temporada == temporada) & (df.SeasonStage == stage)][columns]
        data = data.groupby('MetaEquipo').max().reset_index()
        data['Offensive Superavit'] = data['current_goals'] - data['current_exp_goals']
        data = data.sort_values(by='Offensive Superavit', ascending=False)
        data[['ranking', 'current_goals']] = data[['ranking', 'current_goals']].astype(int)
        data = data.rename(columns={'current_goals':'GF', 'current_exp_goals':'xG'})

        column_1.subheader("Offensive Superavit")

        match_filter = column_1.checkbox("Filter Match Teams")

        if match_filter:
            data = data[data.MetaEquipo.isin([home_team, inverse_name_mapping[away_team]])]
        
        column_1.dataframe(data.style.applymap(lambda x: color_gradient(x, data, 'Offensive Superavit'), subset=['Offensive Superavit']).format("{:.2f}", subset=['xG', 'Offensive Superavit']))



        columns = ['MetaEquipo',
                   'current_goals_against','current_exp_goals_against',
                   ]

        data = df[(df.Temporada == temporada) & (df.SeasonStage == stage)][columns]
        data = data.groupby('MetaEquipo').max().reset_index()
        data['Defensive Superavit'] = data['current_goals_against'] - data['current_exp_goals_against']
        data = data.sort_values(by='Defensive Superavit', ascending=True)
        data[['current_goals_against']] = data[['current_goals_against']].astype(int)
        data = data.rename(columns={'current_goals_against':'GA', 'current_exp_goals_against':'xGA'})
        
        column_2.subheader("Defensive Superavit")
        
        match_filter = column_2.checkbox("Filter Match Teams ")
        
        if match_filter:
            data = data[data.MetaEquipo.isin([home_team, inverse_name_mapping[away_team]])]
        column_2.dataframe(data.style.applymap(lambda x: color_gradient(x, data, 'Defensive Superavit', reverse_color_scale=True), subset=['Defensive Superavit']).format("{:.2f}", subset=['xGA', 'Defensive Superavit']))
        st.markdown("---")
        st.subheader("Real vs Expected")
        columns = ['MetaEquipo',
                   'current_points','current_exp_points', 'current_goals_difference'
                   ]
        data = df[(df.Temporada == temporada) & (df.SeasonStage == stage)].dropna(subset={'Result'})[columns]
        aux_data = df[(df.Temporada == temporada) & (df.SeasonStage == stage) & (df.Jornada == jornada)][['MetaEquipo','ranking']].rename({'ranking':"current_ranking"})

        data = data.groupby('MetaEquipo').max().reset_index()
        data = data.merge(aux_data, on=['MetaEquipo'], how='left')
        data['Points Superavit'] = data['current_points'] - data['current_exp_points']
        data[[ 'current_points','current_exp_points', 'current_goals_difference', 'Points Superavit','ranking']] = data[[ 'current_points','current_exp_points', 'current_goals_difference', 'Points Superavit','ranking']].astype(int)
        match_filter = st.checkbox("Filter Match Teams  " )
        
        if match_filter:
            data = data[data.MetaEquipo.isin([home_team, inverse_name_mapping[away_team]])]
        
        st.markdown("By Points Superavit")
        st.dataframe(data.sort_values(by='Points Superavit',  ascending=False).style.applymap(lambda x: color_gradient(x, data, 'Points Superavit'), subset=['Points Superavit']))
        
        st.markdown("By Ranking")
        st.dataframe(data.sort_values(by='ranking').style.applymap(lambda x: color_gradient(x, data, 'Points Superavit'), subset=['Points Superavit']))
        
        st.markdown("By Goals Difference")
        st.dataframe(data.sort_values(by='current_goals_difference', ascending=False).style.applymap(lambda x: color_gradient(x, data, 'Points Superavit'), subset=['Points Superavit']))
        
        
        
        st.subheader("Summary by Jornada")
        
        columns = ['MetaEquipo','Date',"Jornada","Venue","GF","GA","Result",'Opponent']
        data = df[(df.Temporada == temporada) & (df.SeasonStage == stage)][columns].dropna(subset={'Result'})
        data['HomeTeam'] = data.apply(lambda x: x['MetaEquipo'] if x['Venue'] == 'Home' else inverse_name_mapping[x['Opponent']], axis=1)
        data['HomeGoals'] = data.apply(lambda x: x['GF'] if x['Venue'] == 'Home' else x['GA'], axis=1) 
        data['AwayGoals'] = data.apply(lambda x: x['GF'] if x['Venue'] == 'Away' else x['GA'], axis=1)
        data['HomeTeamWins'] = data.apply(lambda x: True if (((x['Result'] == 'W') and (x['Venue'] == 'Home')) or ((x['Result'] == "L") and (x['Venue'] == 'Away'))) else False, axis=1)
        data['AwayTeamWins'] = data.apply(lambda x: True if (((x['Result'] == 'W') and (x['Venue'] == 'Away')) or ((x['Result'] == "L") and (x['Venue'] == 'Home'))) else False, axis=1 )
        data['Draws'] = data['Result'] == 'D'
        columns = ['Jornada','HomeTeam','HomeGoals','AwayGoals','HomeTeamWins', 'AwayTeamWins','Draws', ]
        data = data[columns].drop_duplicates()
        data = data.groupby(by=['Jornada', 'HomeTeam']).sum().reset_index()
        data = data.groupby(by=['Jornada']).sum().reset_index()
        # data = data.iloc[:,1:]
        # data[columns] = data[columns].astype(int)
        # columns.remove('Jornada')
        # data[columns] = data[columns] // 2


        st.dataframe(data[['Jornada','HomeGoals','AwayGoals','HomeTeamWins','AwayTeamWins','Draws']])

    if selected == '@Broodds Visuals':
        # Scatter Plot 

        temporadas = df.Temporada.unique().tolist()[::-1]
        # Add a selectbox to the sidebar:
        temporada = st.selectbox(
            'Select Season',
            temporadas
        )
        xvars = ['xG','xGA', 'GF','GA']
        x_var = st.selectbox('Select X', xvars)

        yvars = list(set(xvars) - set(x_var))
        y_var = st.selectbox('Select Y', yvars)

        
        columns = ['MetaEquipo',
                   'xG','xGA', 'GF','GA'
                   ]
        
        data = df[(df.Temporada == temporada) & (df.SeasonStage.isin(season_stages))].dropna(subset={'Result'})[columns]
        data = data.groupby('MetaEquipo').sum().reset_index()
        st.dataframe(data)

        make_scatter_team_plot(data, xcolumn=x_var,ycolumn= y_var, title="", xlabel=x_var, ylabel=y_var, zoom=0.063)
        


    # You can add more content below the columns
    st.markdown("---")
    st.subheader("Additional Content Below Columns")
    st.write("You can add more content below the columns.")