import os
os.environ['USE_PYGEOS'] = '0'

import dash
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import shapely
import geopandas as gpd

from dash import dash_table
from dash import dcc
from dash import html 
from dash.dependencies import State, Input, Output
from dash.exceptions import PreventUpdate
from datetime import datetime as dt

from lapin.analysis.occup import compile_occupancy
from lapin.analysis.rempla import compile_parking_time
from lapin.analysis.restrictions import RestrictionHandler
from lapin.analysis.utils import restriction_aggfunc
from lapin.analysis import constants 
from lapin.figures import config
from lapin.config import ROADS_DB_CONNECTION

#######################
####### GLOBALS #######
#######################

DATA_PATH = "./data/raw_data_project.csv"
SEG_GEOM_PATH = ROADS_DB_CONNECTION['filename']

# There might not be locales on the server
FRENCH_DAY = {
    0: 'Lundi',
    1: 'Mardi',
    2: 'Mercredi',
    3: 'Jeudi',
    4: 'Vendredi',
    5: 'Samedi',
    6: 'Dimanche'
}

FRENCH_MONTH = {
    1: 'janvier',
    2: 'février',
    3: 'mars',
    4: 'avril',
    5: 'mai',
    6: 'juin',
    7: 'juillet',
    8: 'août',
    9: 'septembre',
    10: 'octobre',
    11: 'novembre',
    12: 'décembre',
}

RAW_COLS_OI = ['data_index', 'uuid', 'datetime', 'lat', 'lng', 'plaque',
       'plate_state', 'segment', 'point_on_segment', 'lap', 'dir_veh',
       'modification', 'side_of_street', 'restrictions', 'res_days',
       'res_hour_from', 'res_hour_to', 'nb_places_total'] 

######################
###### DASH APP ######
######################

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
)
app.title = "Analyse LAPI"
server = app.server

# Plotly mapbox public token
mapbox_access_token = 'pk.eyJ1IjoiYWxhdXJlbnQzNCIsImEiOiJja28xcnFocTIwb2QyMnd0ZG5oc2pvaDl4In0.iOefsxCQnpJSarh39T2aIg' 

######################
####### HELPER #######
######################

def date_to_human_readable(datetime: pd.Series, locale='en_US') -> pd.Series:
    if not pd.api.types.is_datetime64_any_dtype(datetime):
       datetime = pd.to_datetime(datetime) 

    return datetime.dt.day_of_week.map(FRENCH_DAY) + " " + \
           datetime.dt.day.astype(str) + " " + \
           datetime.dt.month.map(FRENCH_MONTH) + " " + \
           datetime.dt.year.astype(str)

###########################
#######  READ DATA ########
###########################

# Initialize data frame
print('Read raw data...')
df = pd.read_csv(
    DATA_PATH,
    dtype={'data_index': str, 'modification':str}
)
# add columns
df[constants.DATETIME] = pd.to_datetime(df[constants.DATETIME])
df['day'] = date_to_human_readable(df[constants.DATETIME])
df['hour'] = df[constants.DATETIME].dt.hour
df = df[~df.is_restrict]

# Initialize segment
segments_geo = gpd.read_file(SEG_GEOM_PATH)
segments_geo[constants.SIDE_OF_STREET] = segments_geo.COTE.map(
    {'Gauche':-1, 'Droite':1}
)

#########################
####### GET DATA ########
#########################

def get_dataframe(day, hours):
    return df[(df['day'] == day) & df['hour'].isin(hours)].copy()

def get_parking_time(day, hours):
    ## Parking time
    df = get_dataframe(day, hours)
    park_time = compile_parking_time(df)    

    park_time['datetime'] = pd.to_datetime(park_time["datetime"])
    park_time['day'] = date_to_human_readable(park_time['datetime'])
    park_time['park_time'] = park_time['park_time'].div(3600).round(1)
    park_time['hour_from'] = park_time['arrival_time'].dt.hour
    park_time['hour_to'] = park_time['departure_time'].dt.hour
    park_time['hour'] = park_time.apply(
        func=lambda x: np.arange(x.hour_from, x.hour_to+1, 1),
        axis=1
    )
    park_time = park_time.explode('hour', ignore_index=True)
     
    return park_time

def get_occupancy(day, hours):
    ## Occupancy
    df = get_dataframe(day, hours)
    occup = compile_occupancy(df)

    occup['time'] = pd.to_datetime(occup['time'])
    occup['day'] = date_to_human_readable(occup['time'])
    occup['hour'] = occup['time'].dt.hour
    occup['occ'] = occup["occ"].apply(
        lambda x: round(x, 2) if isinstance(x, float) else x
    )
    
    return occup

def aggregate_pk_time(day, hours):
    park_time = get_parking_time(day, hours)
    park_time_dh = park_time.pivot_table(
        values=['park_time'],
        index=[constants.SEGMENT, constants.SIDE_OF_STREET, 'day', 'hour'],
        aggfunc=lambda x: restriction_aggfunc(x.to_list()),
        dropna=False
    ).reset_index().dropna()

    park_time_dh['park_time'] = park_time_dh['park_time'].round(1)

    return park_time_dh.rename(columns={
        'park_time': 'Temps de stat.'
    })

def aggregate_occup(day, hours):
    occup = get_occupancy(day, hours)
    occup_dh = occup.pivot_table(
        values=['occ', 'veh_sighted', constants.CAP_N_VEH],
        index=[constants.SEGMENT, constants.SIDE_OF_STREET, 'day', 'hour'],
        aggfunc=lambda x: restriction_aggfunc(x.to_list()),
        dropna=False
    ).reset_index().dropna()

    occup_dh['occ'] = occup_dh["occ"].apply(
        lambda x: round(x, 2) if isinstance(x, float) else x
    )

    return occup_dh.rename(columns={
        'occ': 'Occupation',
        'veh_sighted': 'Nb. Véhicule vus',
        constants.CAP_N_VEH: 'Capacité',
    })


# Parameters
## Day list
day_list = list(df.sort_values('datetime')['day'].unique())
day_hour_dict = df.groupby('day')['hour'].unique()

## Cost Metric
metric_display = [
    "Occupation",
    "Temps de stat."
]
metric_columns = {
    "Occupation": [
        'Occupation',
        'Nb. Véhicule vus',
        'Capacité', 
    ],
    "Temps de stat.": [
        'Temps de stat.'
    ]
}
metric_to_rm = {
    'Occupation': ['Temps de stat.'],
    'Temps de stat.' : ['Occupation']
}

metric_aggregation = {
    'Occupation'       :restriction_aggfunc,
    'Nb. Véhicule vus' :restriction_aggfunc,
    'Capacité'         :restriction_aggfunc,
    'Temps de stat.'   :restriction_aggfunc
}

DF = df.copy()

def generate_aggregation(aggregation, day_selected, hour_selected, metric_selected):
    if metric_selected == 'Occupation':
        data_filtered = aggregate_occup(day_selected, hour_selected)
        data_filtered['Temps de stat.'] = np.nan
    else:
        data_filtered = aggregate_pk_time(day_selected, hour_selected)
        data_filtered['Nb. Véhicule vus'] = np.nan
        data_filtered['Capacité'] = np.nan
        data_filtered['Occupation'] = np.nan

    grouped = (
        data_filtered.groupby([constants.SEGMENT, constants.SIDE_OF_STREET])
        .agg(aggregation)
        .reset_index()
    )
    
    grouped = grouped.join(
        segments_geo.set_index(['ID_TRC', 'side_of_street'])[['geometry']],
        on=[constants.SEGMENT, 'side_of_street'],
        how='left'
    ).drop_duplicates(keep='first')

    #as geopandas.GeoDataFrame
    grouped = gpd.GeoDataFrame(grouped, geometry='geometry', crs=segments_geo.crs)
    grouped['Occupation'] = grouped["Occupation"].apply(
        lambda x: round(x, 2) if isinstance(x, float) else x
    )
    grouped['Temps de stat.'] = grouped['Temps de stat.'].round(1)

    return grouped
    
def build_upper_left_panel():
    return html.Div(
        id="upper-left",
        className="six columns",
        children=[
            html.P(
                className="section-title",
                children="Choisir un segment sur la carte pour afficher les détails ci-dessous",
            ),
            html.Div(
                className="control-row-1",
                children=[
                    html.Div(
                        id="day-select-outer",
                        children=[
                            html.Label("Choisir un jours"),
                            dcc.Dropdown(
                                id="day-select",
                                options=[{"label": i, "value": i} for i in day_list],
                                value=day_list[0],
                            ),
                        ],
                    ),
                    html.Div(
                        id="select-metric-outer",
                        children=[
                            html.Label("Choisir une métrique"),
                            dcc.Dropdown(
                                id="metric-select",
                                options=[{"label": i, "value": i} for i in metric_display],
                                value=metric_display[0],
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                id="hour-select-outer",
                className="control-row-2",
                children=[
                    html.Label("Choisir une heure"),
                    html.Div(
                        id="hour-selector",
                        children=dcc.RangeSlider(
                            id="hour-select",
                            min=day_hour_dict[day_list[0]][0], 
                            max=day_hour_dict[day_list[0]][-1], 
                            step=1,
                            value=[day_hour_dict[day_list[0]][0], day_hour_dict[day_list[0]][1]], 
                            marks={int(h):f"{h:02d}:00" for h in day_hour_dict[day_list[0]]},
                            allowCross=False,
                        ),
                    ),
                ],
            ),
            html.Div(
                id="table-container",
                className="table-container",
                children=[
                    html.Div(
                        id="table-upper",
                        children=[
                            html.P("Données agrégées"),
                            dcc.Loading(children=html.Div(id="data-aggregated-container")),
                        ],
                    ),
                ],
            ),
        ],
    )


def get_color(gdf, metric_selected):

    if metric_selected == 'Temps de stat.':
        numeric_cuts=config.REMPLA_NUM_CUTS
        num_colors=config.REMPLA_COLORS
        base_cat_colors=config.BASIC_CAT_COLORS
        num_colors = {k:f'#{int(v[0]*255):02x}{int(v[1]*255):02x}{int(v[2]*255):02x}' for k,v in num_colors.items()}
    else:
        numeric_cuts=config.OCC_NUM_CUTS
        num_colors=config.OCC_COLORS
        base_cat_colors=config.BASIC_CAT_COLORS

    # grab all numeric data
    data_num = gdf[[
        pd.api.types.is_number(x) and not pd.isna(x) for x in gdf[metric_selected]
    ]].copy()

    if pd.api.types.is_numeric_dtype(gdf[metric_selected].dtype):
        data_num = gdf[[not pd.isna(x) for x in gdf[metric_selected]]].copy()

    labels=list(numeric_cuts.keys())
    data_num['category'] = pd.cut(
        data_num[metric_selected], 
        [0] + list(numeric_cuts.values()),
        labels=labels,
        include_lowest=True
    )

    # grab all categorical data
    data_cat = pd.DataFrame()
    if not pd.api.types.is_numeric_dtype(gdf[metric_selected].dtype):
        data_cat = gdf[[
            not (pd.api.types.is_number(x) or pd.isna(x) ) for x in gdf[metric_selected]
        ]].copy()
        data_cat['category'] = data_cat[metric_selected].copy()

    # grab all nan
    data_na = gdf[[pd.isna(x) for x in gdf[metric_selected]]].copy()
    data_na['category'] = [
        'Sans données' for i in range(data_na[metric_selected].shape[0])
    ]

    # bin them
    data_c = pd.concat([data_num, data_cat, data_na])

    data_c['category'] = data_c['category'].astype(str)
    base_cat_colors = {
        key: base_cat_colors[key] for key in data_c['category'].unique() \
            if key in base_cat_colors.keys()
    }
    catColors = dict(base_cat_colors, **num_colors)
    data_c['color_cat'] = data_c['category'].map(catColors)

    return data_c

def generate_geo_map(gdf: gpd.GeoDataFrame, metric_selected: str):

    gdf = get_color(gdf=gdf, metric_selected=metric_selected)

    curbs = []
    iterate_on = zip(
        gdf.geometry, 
        gdf[constants.SEGMENT],
        gdf[constants.SIDE_OF_STREET],
        gdf[metric_selected],
        gdf['Nb. Véhicule vus'],
        gdf['Capacité'],
        gdf['color_cat'],
    )

    mean_lng, mean_lat = list(gdf.unary_union.centroid.coords)[0]

    for feature, seg, sos, metric, nb_veh, cap, color in iterate_on:
        lats = []
        lons = []

        if isinstance(feature, shapely.geometry.linestring.LineString):
            linestrings = [feature]
        elif isinstance(feature, shapely.geometry.multilinestring.MultiLineString):
            linestrings = feature.geoms
        else:
            continue
        for linestring in linestrings:
            x, y = linestring.xy
            lats = np.append(lats, y)
            lons = np.append(lons, x)

        text_info = (seg.__str__()
            + "<br>" 
            + metric_selected.__str__()
            + ':'
            + f" {metric}"
            + "<br> Capacité:"
            + f" {cap}"
            + "<br> Nombre de véhicule:"
            + " {:,.2f}".format(nb_veh))
        if metric_selected == 'Temps de stat.':
            text_info = (seg.__str__()
                + "<br>" 
                + metric_selected.__str__()
                + ':'
                + f" {metric}")


        curb = go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode="markers+lines",
            line=dict(
                color=color,
            ),
            opacity=0.8,
            selected=dict(marker={"color": "#ffff00"}),
            customdata=[(seg,sos, metric, nb_veh, cap) for i in range(len(lats))],
            hoverinfo="text",
            text=text_info, 
        )
        curbs.append(curb)

    layout = go.Layout(
        margin=dict(l=10, r=10, t=20, b=10, pad=5),
        plot_bgcolor="#171b26",
        paper_bgcolor="#171b26",
        clickmode="event+select",
        hovermode="closest",
        showlegend=False,
        mapbox=go.layout.Mapbox(
            accesstoken=mapbox_access_token,
            bearing=10,
            center=go.layout.mapbox.Center(
                lat=mean_lat, lon=mean_lng
            ),
            pitch=5,
            zoom=14,
            style="mapbox://styles/mapbox/light-v11",
        ),
    )

    return {"data": curbs, "layout": layout}

app.layout = html.Div(
    className="container scalable",
    children=[
        html.Div(
            id="banner",
            className="banner",
            children=[
                html.H6("Analyse d'occupation par LAPI"),
                html.Img(src=app.get_asset_url("LOGO OFFICIEL.png")),
            ],
        ),
        html.Div(
            id="upper-container",
            className="row",
            children=[
                build_upper_left_panel(),
                html.Div(
                    id="geo-map-outer",
                    className="six columns",
                    children=[
                        html.P(
                            id="map-title",
                            children="Analyses du {}".format(
                                day_list[0]
                            ),
                        ),
                        html.Div(
                            id="geo-map-loading-outer",
                            children=[
                                dcc.Loading(
                                    id="loading",
                                    children=dcc.Graph(
                                        id="geo-map",
                                        figure={
                                            "data": [],
                                            "layout": dict(
                                                plot_bgcolor="#171b26",
                                                paper_bgcolor="#171b26",
                                            ),
                                        },
                                    ),
                                )
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            id='table-lower',
            children=[
                html.P("Données brutes"),
                dcc.Loading(
                    children=html.Div(id="raw-data-container")
                ),
            ],
        ),
        #html.Div(
            #id="lower-container",
            #children=[
                #dcc.Graph(
                    #id="procedure-plot",
                    #figure=generate_procedure_plot(
                        #data_dict[day_list[1]], metric[0], init_day, []
                    #),
                #)
            #],
        #),
    ],
)


@app.callback(
    [
        Output("hour-select", "min"),
        Output("hour-select", "max"),
        Output("hour-select", "value"),
        Output("hour-select", "marks"),
        Output("map-title", "children"),
    ],
    [Input("day-select", "value"),],
)
def update_region_dropdown(day_select):
    min=day_hour_dict[day_select][0] 
    max=day_hour_dict[day_select][-1] 
    value=[int(day_hour_dict[day_select][0]), int(day_hour_dict[day_select][1])] 
    marks={int(h):f"{h:02d}:00" for h in day_hour_dict[day_select]}

    return (
        int(min),
        int(max),
        value,
        marks,
        "Analyses du {}".format(day_select),
    )

@app.callback(
    Output("data-aggregated-container", "children"),
    [
        Input("geo-map", "selectedData"),
        Input("metric-select", "value"),
        Input("day-select", "value"),
        Input("hour-select", "value")
    ],
)
def update_aggregated_datatable(geo_select, metric_select, day_select, hours_select):

    if metric_select == 'Occupation':
        day_agg = get_occupancy(day_select, hours_select)
    else:
        day_agg = get_parking_time(day_select, hours_select)
    # make table from geo-select
    geo_data_list = [] 

    ctx = dash.callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if prop_id == "geo-map" and geo_select is not None:

            for point in geo_select["points"]:
                if "customdata" in point.keys():
                    segment = point["customdata"][0]
                    sos = point["customdata"][1]
                    dff = day_agg[
                        (day_agg["segment"] == segment) &\
                        (day_agg["side_of_street"] == sos)
                    ]

                    geo_data_list.append(dff)

    # keep only metric column
    #cols_to_rm = [metric_columns[x] for x in metric_to_rm[metric_select]]
    #cols_to_rm = [item for sublist in cols_to_rm for item in sublist]
    #cols = [col for col in day_agg.columns if col not in cols_to_rm]

    if geo_data_list:
        geo_data_df = pd.concat(geo_data_list)
        #geo_data_df = geo_data_df[cols]
    else:
        geo_data_df = pd.DataFrame(data=day_agg)

    # remove cols
    geo_data_df.drop(
        columns=['geometry', 'segment_geodbl_geom', 'day', 'hour'],
        inplace=True, 
        errors='ignore'
    ) 
    geo_data_df.drop_duplicates(inplace=True)
    try:
        geo_data_df['datetime'] = geo_data_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        pass
    try:
        geo_data_df['time'] = geo_data_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        pass

    geo_data_df = geo_data_df.drop_duplicates()
    data_json = geo_data_df.to_dict("rows")

    return dash_table.DataTable(
        id="data-aggregated-table",
        columns=[{"name": i, "id": i} for i in geo_data_df.columns],
        data=data_json,
        sort_action="native",
        page_size=5,
        style_cell={"background-color": "#242a3b", "color": "#7b7d8d"},
        style_as_list_view=False,
        style_header={"background-color": "#1f2536", "padding": "0px 5px"},
        export_format="xlsx",
    )


@app.callback(
    Output("raw-data-container", "children"),
    [
        Input("geo-map", "selectedData"),
        Input("day-select", "value"),
        Input("hour-select", "value")
    ],
)
def update_raw_data(geo_select, day_select, hour_select):
    raw_data_select = []

    ctx = dash.callback_context
    prop_id = ""
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if prop_id == "geo-map" and geo_select is not None:
        for point in geo_select["points"]:
            if "customdata" in point.keys():
                seg = point["customdata"][0]
                sos = point["customdata"][1]
                raw_data_select.append(DF[(
                    (DF.segment == seg) & 
                    (DF.side_of_street == sos) &
                    (DF.day == day_select) &
                    DF.hour.isin(hour_select)
                )][RAW_COLS_OI])

    if raw_data_select:
        raw_data_select = pd.concat(raw_data_select)
    else:
        raw_data_select = pd.DataFrame(data=DF, columns=DF[RAW_COLS_OI].columns)

    try:
        raw_data_select['datetime'] = raw_data_select['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        pass

    raw_data_select.drop(columns=['segment_geodbl_geom'], inplace=True, errors='ignore') 

    raw_data_select.drop_duplicates(inplace=True)
    raw_data_select = raw_data_select[RAW_COLS_OI]

    return dash_table.DataTable(
        id="raw-data-table",
        columns=[{"name": i, "id": i} for i in raw_data_select.columns],
        data=raw_data_select.to_dict("rows"),
        sort_action="native",
        style_cell={
            "textOverflow": "ellipsis",
            "background-color": "#242a3b",
            "color": "#7b7d8d",
        },
        sort_mode="multi",
        page_size=5,
        style_as_list_view=False,
        style_header={"background-color": "#1f2536", "padding": "2px 12px 0px 12px"},
        export_format="xlsx",
    )


@app.callback(
    Output("geo-map", "figure"),
    [
        Input("metric-select", "value"),
        Input("hour-select", "value"),
        Input("day-select", "value"),
    ],
)
def update_geo_map(metric_select, hour_select, day_select):
    # generate geo map from state-select, procedure-select
    day_hour_agg_data = generate_aggregation(
        aggregation=metric_aggregation,
        day_selected=day_select,
        hour_selected=hour_select,
        metric_selected=metric_select
    )

    return generate_geo_map(day_hour_agg_data, metric_select)


if __name__ == "__main__":
    app.run_server(debug=True)