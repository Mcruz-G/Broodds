import pandas as pd
import streamlit as st


from utils import *

scores_and_fixtures_df = pd.read_csv("data/csvdata/scores_and_fixtures.csv").iloc[:,1:]

if __name__ == "__main__":

    set_streamlit_config()

    over_line, cells_format = initial_layout()        

    home_team, away_team = team_selection_layout(scores_and_fixtures_df)

    season_stages = season_stage_layout(scores_and_fixtures_df)

    selected = sidebar_layout()        

    if selected == 'Historic Match Results':
        historic_match_results(scores_and_fixtures_df, home_team, away_team, over_line, season_stages, cells_format)
    
    if selected == 'Team Analysis':
        team_analysis(scores_and_fixtures_df, home_team, away_team, over_line, season_stages, cells_format)
    
    if selected == 'Home Team Goals Analysis':
        home_team_goal_analysis(scores_and_fixtures_df, home_team, over_line, season_stages)
    
    if selected == 'Away Team Goals Analysis':
        away_team_goal_analysis(scores_and_fixtures_df, away_team, over_line, season_stages)
    
    if selected == 'Goals Analysis':
        goal_analysis(scores_and_fixtures_df, home_team, away_team, over_line, season_stages)
    
    if selected == 'Season Analysis':
        season_analysis(scores_and_fixtures_df, season_stages, home_team, away_team)


    # You can add more content below the columns
    st.markdown("---")

    st.subheader("For additional content contact macruzgom@gmail.com")
    