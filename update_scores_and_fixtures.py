import pandas as pd
import sqlite3
import time


# Define a function to determine the SeasonStage
def determine_season_stage(date, round_info):
    if "Apertura" in round_info and "Regular Season" in round_info:
        return "Apertura"
    elif "Clausura" in round_info and "Regular Season" in round_info:
        return "Clausura"
    elif "Guardianes" in round_info:
        return "Guardianes"
    else:
        month = pd.to_datetime(date).month
        if month < 6:
            return "Liguilla-Clausura"
        return "Liguilla-Apertura"

# Define a function to extract the year
def extract_year(round_info, date):
    if "Quarter-finals" in round_info or "Semi-finals" in round_info or "Repechaje" or "Reclasificacion" in round_info or "Finals" in round_info:
        year = pd.to_datetime(date).year
        year = str(year)
        return year
    return ''.join(filter(str.isdigit, round_info))

# Define a function to determine the Season Type
def determine_season_type(round_info):
    if len(round_info.split('— ')) > 1:
        return round_info.split('— ')[-1]
    if len(round_info.split(' ')) > 1:
        return round_info.split(' ')[-2] + " " + round_info.split(' ')[-1]
    return round_info.split(' ')[-1]

# Función para extraer la temporada del enlace
def extraer_temporada(link):
    temporada = link.split('/')[-2]
    return temporada

def fbref_pull_and_store_data(metadata):
    # Procesar cada fila y actualizar/crear bases de datos SQL
    # complete this ditionary with the metadata of the tables you want to update
    unique_teams  = metadata.MetaEquipo.unique().tolist()
    datos_equipo = []  # Inicializar el DataFrame para los datos del equipo
    data_dict = {x : metadata[metadata.MetaEquipo == x] for x in unique_teams}

    for _, grupo in data_dict.items():
        print(f"Procesando base de datos: ")

        for _, fila in grupo.iterrows():
            tabla_nombre = fila['Tabla']
            link = fila['Link']
            metaequipo = fila['MetaEquipo']
            temporada = extraer_temporada(link) 

            print(f"Descargando datos de {tabla_nombre}, Temporada: {temporada} desde {link}")
            
            time.sleep(1)  # Espera 1 segundo entre cada solicitud
            datos_tabla = pd.read_html(link)
            datos_tabla = datos_tabla[1]
            datos_tabla['Temporada'] = temporada  # Agregar la temporada como columna
            datos_tabla['MetaEquipo'] = metaequipo
            
            datos_tabla.columns = datos_tabla.columns.str.replace(' ', '_') # Reemplazar espacios con guiones bajos 
            # datos_tabla.columns = datos_tabla.columns.droplevel(0)
            datos_equipo.append(datos_tabla)
    return pd.concat(datos_equipo, ignore_index=True, sort=False)

def data_cleaning(df):
    df['Formation'] = df.Formation.apply(lambda x: "" if x == None else x)

    df['Formation'] = df.Formation.apply(lambda x: x if '◆' not in x else x.split('◆')[0])
    df['GF'] = df['GF'].apply(lambda x: x if x != None else '0')
    df['GF'] = df['GF'].apply(lambda x: x if "(" not in x else x.split("(")[0])
    df['GF'] = df['GF'].astype(float)
    df['GA'] = df['GA'].apply(lambda x: x if x != None else '0')
    df['GA'] = df['GA'].apply(lambda x: x if "(" not in x else x.split("(")[0])
    df['GA'] = df['GA'].astype(float)
    df['xG'] = df['xG'].apply(lambda x: x if x != None else '0')
    df['xG'] = df['xG'].astype(float)

    df['xGA'] = df['xGA'].apply(lambda x: x if x != None else '0')
    df['xGA'] = df['xGA'].astype(float)
    return df


def update_scores_and_features(metadata, current_season='2022-2023'):
    initial_year = current_season.split('-')[0]

    filtered_metadata = metadata[metadata.Tabla.isin(list(filter(lambda x: x.split('_')[-1] == 'features', metadata.Tabla.unique().tolist())))]
    filtered_metadata = filtered_metadata[filtered_metadata.Link.isin(list(filter(lambda x: x.split('/')[-2].split('-')[0] == initial_year, metadata.Link.unique().tolist())))]

    db_path = "data/sqldata/Historic_scores_and_fixtures.db"
    conn = sqlite3.connect(db_path)

    retrieved_new_data = fbref_pull_and_store_data(filtered_metadata)
    retrieved_new_data['SeasonStage'] = retrieved_new_data.apply(lambda row: determine_season_stage(row['Date'], row['Round']), axis=1)
    retrieved_new_data['Yr'] = retrieved_new_data.apply(lambda row: extract_year(row['Round'], row['Date']), axis=1)
    retrieved_new_data['Season Type'] = retrieved_new_data['Round'].apply(determine_season_type)
    retrieved_new_data['Formation'] = retrieved_new_data.Formation.apply(lambda x: "" if x == None else x)

    # retrieved_new_data['Formation'] = retrieved_new_data.Formation.apply(lambda x: x if '◆' not in x else x.split('◆')[0])
    # For rows where the year is not directly available, we'll use the date to determine the year
    for index, row in retrieved_new_data.iterrows():
        if pd.isnull(row['Yr']):
            retrieved_new_data.at[index, 'Yr'] = pd.to_datetime(row['Date']).year

    current_total_data = pd.read_sql_query("SELECT * FROM scores_and_fixture", conn)
    current_total_data = current_total_data[~current_total_data.Temporada.isin([current_season])]
    current_total_data = pd.concat([current_total_data, retrieved_new_data], ignore_index=True, sort=False)
    # Apply the functions to the DataFrame


    current_total_data.to_sql('scores_and_fixture', conn, if_exists='replace', index=False)

    
def add_current_points(df):
    # Ordena los datos por temporada, seasonstage, Jornada y fecha para asegurar un orden correcto
    df.sort_values(by=['Temporada', 'SeasonStage', 'MetaEquipo', 'Jornada', 'Date'], inplace=True)
    # Inicializa las columnas 'current_points', 'current_wins' y 'current_goals' en NaN (espacios vacíos)
    df['current_points'] = ''
    df['current_wins'] = ''
    df['current_goals'] = ''
    df['current_ranking_points'] = ''
    df['current_ranking_wins'] = ''
    df['current_ranking_goals'] = ''
    df['current_ranking_score'] = ''
    df['current_goals_difference'] = ''
    df['ranking'] = ''


    temporadas = df.Temporada.unique().tolist()
    temporadas.remove("2020-2021")
    stages = ['Apertura','Clausura']
    jornadas = list(range(18))

    for temp in temporadas:
        for stage in stages:
            puntos_acumulados = {}
            wins_acumulados = {}
            goals_acumulados = {}
            diferencia_goles_acumulados = {}
            sample = df[(df.Temporada == temp) & (df.SeasonStage == stage)].sort_values(by='Date')
            
            for index, row in sample.iterrows():
                equipo = row['MetaEquipo']

                # Verifica si el equipo ya está en el diccionario de puntos acumulados
                if equipo not in puntos_acumulados:
                    puntos_acumulados[equipo] = 0
                    wins_acumulados[equipo] = 0
                    goals_acumulados[equipo] = 0
                    diferencia_goles_acumulados[equipo] = 0

                # Asigna los puntos según el resultado del partido
                if row['Result'] == 'W':
                    puntos_acumulados[equipo] += 3
                    wins_acumulados[equipo] += 1

                elif row['Result'] == 'D':
                    puntos_acumulados[equipo] += 1

                # Suma los goles
                goals_acumulados[equipo] += row['GF']
                diferencia_goles_acumulados[equipo] += row['GF'] - row['GA']

                # Asigna los valores acumulados a las columnas correspondientes
                sample.at[index, 'current_points'] = puntos_acumulados[equipo]
                sample.at[index, 'current_wins'] = wins_acumulados[equipo]
                sample.at[index, 'current_goals'] = goals_acumulados[equipo]
                sample.at[index, 'current_goals_difference'] = diferencia_goles_acumulados[equipo]
                

            for jornada in jornadas:
                subsample = sample[sample.Jornada == jornada]
                subsample.sort_values(by=['current_points','current_goals_difference','current_goals'], inplace=True, ascending=False)
                subsample['ranking'] = range(1,len(subsample) + 1)
                sample.loc[subsample.index] = subsample
                
            df.loc[sample.index] = sample
    return df

def add_columns(df):

    df['TotalGoals'] = df['GF'] + df['GA']
    df['GoalsDifference'] = df['GF'] - df['GA']
    # Write a set of columns that describe with a boolean if the GF is > than 0,1,2,3,4,5
    for i in range(0, 6):
        df[f'GF_>{i}'] = df['GF'].apply(lambda x: 1 if x > i else 0)
        df[f'GA_>{i}'] = df['GA'].apply(lambda x: 1 if x > i else 0)
        df[f'GF_<{i}'] = df['GF'].apply(lambda x: 1 if x < i else 0)
        df[f'GA_<{i}'] = df['GA'].apply(lambda x: 1 if x < i else 0)
        df[f'GF_={i}'] = df['GF'].apply(lambda x: 1 if x == i else 0)
        df[f'GA_={i}'] = df['GA'].apply(lambda x: 1 if x == i else 0)
        df[f'TotalGoals_>{i}'] = df['TotalGoals'].apply(lambda x: 1 if x > i else 0)
        df[f'GoalsDifference_>{i}'] = df['GoalsDifference'].apply(lambda x: 1 if x > i else 0)
        df[f'TotalGoals_<{i}'] = df['TotalGoals'].apply(lambda x: 1 if x < i else 0)
        df[f'GoalsDifference_<{i}'] = df['GoalsDifference'].apply(lambda x: 1 if x < i else 0)
        df[f'TotalGoals_={i}'] = df['TotalGoals'].apply(lambda x: 1 if x == i else 0)
        df[f'GoalsDifference_={i}'] = df['GoalsDifference'].apply(lambda x: 1 if x == i else 0)
        for j in range(0, 6):
            df[f'GF_>{i} & GA_>{j}'] = df.apply(lambda row: 1 if row['GF'] > i and row['GA'] > j else 0, axis=1)
            df[f'GF_>{i} & GA_<{j}'] = df.apply(lambda row: 1 if row['GF'] > i and row['GA'] < j else 0, axis=1)
            df[f'GF_>{i} & GA_={j}'] = df.apply(lambda row: 1 if row['GF'] > i and row['GA'] == j else 0, axis=1)
            df[f'GF_<{i} & GA_>{j}'] = df.apply(lambda row: 1 if row['GF'] < i and row['GA'] > j else 0, axis=1)
            df[f'GF_<{i} & GA_<{j}'] = df.apply(lambda row: 1 if row['GF'] < i and row['GA'] < j else 0, axis=1)
            df[f'GF_<{i} & GA_={j}'] = df.apply(lambda row: 1 if row['GF'] < i and row['GA'] == j else 0, axis=1)
            df[f'GF_={i} & GA_>{j}'] = df.apply(lambda row: 1 if row['GF'] == i and row['GA'] > j else 0, axis=1)
            df[f'GF_={i} & GA_<{j}'] = df.apply(lambda row: 1 if row['GF'] == i and row['GA'] < j else 0, axis=1)
            df[f'GF_={i} & GA_={j}'] = df.apply(lambda row: 1 if row['GF'] == i and row['GA'] == j else 0, axis=1)
            df[f'TotalGoals_>{i} & GoalsDifference_>{j}'] = df.apply(lambda row: 1 if row['TotalGoals'] > i and row['GoalsDifference'] > j else 0, axis=1)
            df[f'TotalGoals_>{i} & GoalsDifference_<{j}'] = df.apply(lambda row: 1 if row['TotalGoals'] > i and row['GoalsDifference'] < j else 0, axis=1)
            df[f'TotalGoals_>{i} & GoalsDifference_={j}'] = df.apply(lambda row: 1 if row['TotalGoals'] > i and row['GoalsDifference'] == j else 0, axis=1)
            df[f'TotalGoals_<{i} & GoalsDifference_>{j}'] = df.apply(lambda row: 1 if row['TotalGoals'] < i and row['GoalsDifference'] > j else 0, axis=1)
            df[f'TotalGoals_<{i} & GoalsDifference_<{j}'] = df.apply(lambda row: 1 if row['TotalGoals'] < i and row['GoalsDifference'] < j else 0, axis=1)
            df[f'TotalGoals_<{i} & GoalsDifference_={j}'] = df.apply(lambda row: 1 if row['TotalGoals'] < i and row['GoalsDifference'] == j else 0, axis=1)
            df[f'TotalGoals_={i} & GoalsDifference_>{j}'] = df.apply(lambda row: 1 if row['TotalGoals'] == i and row['GoalsDifference'] > j else 0, axis=1)
            df[f'TotalGoals_={i} & GoalsDifference_<{j}'] = df.apply(lambda row: 1 if row['TotalGoals'] == i and row['GoalsDifference'] < j else 0, axis=1)
            df[f'TotalGoals_={i} & GoalsDifference_={j}'] = df.apply(lambda row: 1 if row['TotalGoals'] == i and row['GoalsDifference'] == j else 0, axis=1)



    # Write the column "Jornada" being the rownumber ordered by Date with partition by MetaEquipo, Temporada, SeasonStage only for those rows where the 'Season Type' is 'Regular Season'
    df['Jornada'] = df[df['Season Type'] == 'Regular Season'].groupby(['MetaEquipo', 'Temporada', 'SeasonStage'])['Date'].rank(method='dense', ascending=True).astype(int)
    df['xG'] = df['xG'].fillna(0)
    df['xGA'] = df['xGA'].fillna(0)
    
    df['Expected_GF'] = df['xG'].apply(lambda x: round(x))
    df['Expected_GA'] = df['xGA'].apply(lambda x: round(x))

    # Write the column "Expected_Results" Which is "W" if round(xG) > round(xGA), "D" if round(xG) == round(xGA) and "L" if round(xG) < round(xGA)
    df['Expected_Results'] = df.apply(lambda row: 'W' if row['Expected_GF'] > row['Expected_GA'] else 'D' if row['Expected_GF'] == row['Expected_GA'] else 'L', axis=1)
    df['Expected_Goals_Difference'] = df['Expected_GF'] - df['Expected_GA']

    #Write the column "game_num" being the rownumber ordered by Date with partition by MetaEquipo, Temporada
    df['game_num'] = df.groupby(['MetaEquipo', 'Temporada'])['Date'].rank(method='dense', ascending=True).astype(int)
    
    df['points'] = df.apply(lambda row: 3 if row['Result'] == 'W' else 1 if row['Result'] == 'D' else 0, axis=1)
    df['expected_points'] = df.apply(lambda row: 3 if row['Expected_Results'] == 'W' else 1 if row['Result'] == 'D' else 0, axis=1)
    df = add_current_points(df)
    return df

def read_scores_and_fixture_db():

    conn = sqlite3.connect(f'data/sqldata/Historic_scores_and_fixtures.db')
    c = conn.cursor()
    query = """ 
                SELECT * 
                FROM scores_and_fixture
            """
    c.execute(query)
    c.fetchall()
    df = pd.read_sql_query(query, conn)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df

if __name__ == '__main__':
    metadata = pd.read_csv("data/csvdata/metadata.csv")
    update_scores_and_features(metadata, current_season='2023-2024')
    df = read_scores_and_fixture_db()
    df = data_cleaning(df)
    df = add_columns(df)
    df.to_csv('data/csvdata/scores_and_fixtures.csv')
