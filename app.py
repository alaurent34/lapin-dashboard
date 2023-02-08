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

FRENCH_DAY = {
    0: 'Lundi',
    1: 'Mardi',
    2: 'Mercredi',
    3: 'Jeudi',
    4: 'Vendredi',
    5: 'Samedi',
    6: 'Dimanche'
}

RAW_COLS_OI = ['data_index', 'uuid', 'datetime', 'lat', 'lng', 'plaque',
       'plate_state', 'segment', 'point_on_segment', 'lap', 'dir_veh',
       'modification', 'side_of_street', 'restrictions', 'res_days',
       'res_hour_from', 'res_hour_to', 'nb_places_total'] 

def date_to_human_readable(datetime: pd.Series, locale='fr') -> pd.Series:
    if not pd.api.types.is_datetime64_any_dtype(datetime):
       datetime = pd.to_datetime(datetime) 

    return datetime.dt.day_name(locale=locale) + " " + \
           datetime.dt.day.astype(str) + " " + \
           datetime.dt.month_name(locale=locale).str.lower() + " " + \
           datetime.dt.year.astype(str)

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
)
app.title = "TEST LAPI"
server = app.server


# Plotly mapbox public token
mapbox_access_token = 'pk.eyJ1IjoiYWxhdXJlbnQzNCIsImEiOiJja28xcnFocTIwb2QyMnd0ZG5oc2pvaDl4In0.iOefsxCQnpJSarh39T2aIg' 

# Initialize data frame
print('Read raw data...')
df = pd.read_csv(
    '../../output/20000_Test/cache/data_enhanced_restrict.csv'
    #'./data/raw_data_sample.csv',
)
df[constants.DATETIME] = pd.to_datetime(df[constants.DATETIME])
#df['day'] = df[constants.DATETIME].dt.day_of_week.map(FRENCH_DAY)
df['day'] = date_to_human_readable(df[constants.DATETIME])
df['hour'] = df[constants.DATETIME].dt.hour
df = df[~df.is_restrict]

# Generate all metrics
## Parking time
print('Compute parking time...')
park_time = compile_parking_time(df)    
# remove car that were here on start/end
#park_time = park_time[~park_time.first_or_last]
park_time['datetime'] = pd.to_datetime(park_time["datetime"])
#park_time['day'] = park_time['datetime'].dt.day_of_week.map(FRENCH_DAY)
park_time['day'] = date_to_human_readable(park_time['datetime'])
park_time['hour_from'] = park_time['arrival_time'].dt.hour
park_time['hour_to'] = park_time['departure_time'].dt.hour
park_time['hour'] = park_time.apply(
    func=lambda x: np.arange(x.hour_from, x.hour_to+1, 1),
    axis=1
)
park_time = park_time.explode('hour', ignore_index=True)

## Occupancy
print('Compute occupancy...')
occup = compile_occupancy(df)
occup['time'] = pd.to_datetime(occup['time'])
#occup['day'] = occup['time'].dt.day_of_week.map(FRENCH_DAY) 
occup['day'] = date_to_human_readable(occup['time'])
occup['hour'] = occup['time'].dt.hour

## Merge metrics
park_time_dh = park_time.pivot_table(
    values=['park_time'],
    index=[constants.SEGMENT, constants.SIDE_OF_STREET, 'day', 'hour'],
    aggfunc=lambda x: restriction_aggfunc(x.to_list()),
    dropna=False
).reset_index().dropna()
occup_dh = occup.pivot_table(
    values=['occ', 'veh_sighted', constants.CAP_N_VEH],
    index=[constants.SEGMENT, constants.SIDE_OF_STREET, 'day', 'hour'],
    aggfunc=lambda x: restriction_aggfunc(x.to_list()),
    dropna=False
).reset_index().dropna()
data = pd.merge(
    left=occup_dh,
    right=park_time_dh,
    on=[constants.SEGMENT, constants.SIDE_OF_STREET, 'day', 'hour'],
    how='left'#'outer'
)
# Rename metrics
data.rename(
    columns={
        'occ': 'Occupation',
        'veh_sighted': 'Nb. Véhicule vus',
        constants.CAP_N_VEH: 'Capacité',
        'park_time': 'Temps de stat.'

    },
    inplace=True
)
DATA = data.copy()

# Initialize segment
print('Read segments geom...')
#segments = gpd.read_file(
#    '../../output/20000_Test/cache/stationnements_par_segment.csv'
#)
#segments_geo = gpd.read_file(ROADS_DB_CONNECTION['filename'])
segments_geo = gpd.read_file('./data/segment_geo_sample.geojson')
segments_geo[constants.SIDE_OF_STREET] = segments_geo.COTE.map(
    {'Gauche':-1, 'Droite':1}
)

# Restriction Handler
#resH = RestrictionHandler(segments)

# Parameters
## Day list
day_list = list(df.sort_values('datetime')['day'].unique())
#day_list = sorted(
    #list(data['day'].unique()),
    #key=lambda x: {v: k for k, v in FRENCH_DAY.items()}[x]
#)

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

def generate_aggregation(df, aggregation, hour_selected):
    if hour_selected:
        data_filtered = df[
            df['hour'].isin(hour_selected)
        ]
    else:
        data_filtered = df

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
                            html.Label("Select a day"),
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
                        id="checklist-container",
                        children=dcc.Checklist(
                            id="hour-select-all",
                            options=[{'label': 'Selectionnez toutes les heures', 'value': 'All'}],
                            value=[],
                        ),
                    ),
                    html.Div(
                        id="hour-select-dropdown-outer",
                        children=dcc.Dropdown(
                            id="hour-select", multi=True, searchable=True,
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

    if metric_selected == 'Temps de stationnement':
        numeric_cuts=config.REMPLA_NUM_CUTS
        num_colors=config.REMPLA_COLORS
        base_cat_colors=config.BASIC_CAT_COLORS
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

        metric_display = round(metric, 2).__str__() if isinstance(metric, float) else metric.__str__()
        curb = go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode="markers+lines",
            #marker=go.scattermapbox.Marker(
            #    #line=dict(width = 1),
            #    color = color,
            #),
            line=dict(
                color=color,
            ),
            opacity=0.8,
            selected=dict(marker={"color": "#ffff00"}),
            customdata=[(seg,sos, metric, nb_veh, cap) for i in range(len(lats))],
            hoverinfo="text",
            text=seg.__str__()
            + "<br>" 
            + metric_selected.__str__()
            + ':'
            + f" {metric}"
            + "<br> Capacité:"
            + f" {cap}"
            + "<br> Nombre de véhicule:"
            + " {:,.2f}".format(nb_veh),
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
            zoom=12,
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
                html.Img(src=app.get_asset_url("plotly_logo_white.png")),
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
                            children="Analyses au jours de {}".format(
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
        Output("hour-select", "value"),
        Output("hour-select", "options"),
        Output("map-title", "children"),
    ],
    [Input("hour-select-all", "value"), Input("day-select", "value"),],
)
def update_region_dropdown(select_all, day_select):
    day_data = DATA[DATA.day == day_select]
    hours = day_data["hour"].sort_values().unique()
    options = [{"label": i, "value": i} for i in hours]

    ctx = dash.callback_context
    if ctx.triggered[0]["prop_id"].split(".")[0] == "hour-select-all":
        if select_all == ["All"]:
            value = [i["value"] for i in options]
        else:
            value = dash.no_update
    else:
        value = hours[:4]
    return (
        value,
        options,
        "Analyses au jours de {}".format(day_select),
    )


@app.callback(
    Output("checklist-container", "children"),
    [Input("hour-select", "value")],
    [State("hour-select", "options"), State("hour-select-all", "value")],
)
def update_checklist(selected, select_options, checked):
    if len(selected) < len(select_options) and len(checked) == 0:
        raise PreventUpdate()

    elif len(selected) < len(select_options) and len(checked) == 1:
        return dcc.Checklist(
            id="hour-select-all",
            options=[{"label": "Selectionnez toutes les heures", "value": "All"}],
            value=[],
        )

    elif len(selected) == len(select_options) and len(checked) == 1:
        raise PreventUpdate()

    return dcc.Checklist(
        id="hour-select-all",
        options=[{"label": "Selectionnez toutes les heures", "value": "All"}],
        value=["All"],
    )


@app.callback(
    Output("data-aggregated-container", "children"),
    [
        Input("geo-map", "selectedData"),
    ],
    [
        State("metric-select", "value"),
        State("day-select", "value"),
        State("hour-select", "value")
    ],
)
def update_aggregated_datatable(geo_select, metric_select, day_select, hours_select):
    day_agg = generate_aggregation(
        df=DATA[DATA.day == day_select],
        aggregation=metric_aggregation,
        hour_selected=hours_select
    )
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
    cols_to_rm = [metric_columns[x] for x in metric_to_rm[metric_select]]
    cols_to_rm = [item for sublist in cols_to_rm for item in sublist]
    cols = [col for col in day_agg.columns if col not in cols_to_rm]

    if geo_data_list:
        geo_data_df = pd.concat(geo_data_list)
        geo_data_df = geo_data_df[cols]
    else:
        geo_data_df = pd.DataFrame(data=DATA, columns=cols)

    geo_data_df.drop(columns=['geometry'], inplace=True, errors='ignore') 

    geo_data_df = geo_data_df.drop_duplicates()
    data_json = geo_data_df.to_dict("rows")

    return dash_table.DataTable(
        id="data-aggregated-table",
        columns=[{"name": i, "id": i} for i in geo_data_df.columns],
        data=data_json,
        filter_action="native",
        page_size=20,
        style_cell={"background-color": "#242a3b", "color": "#7b7d8d"},
        style_as_list_view=False,
        style_header={"background-color": "#1f2536", "padding": "0px 5px"},
    )


@app.callback(
    Output("raw-data-container", "children"),
    [
        Input("geo-map", "selectedData"),
    ],
    [
        State("day-select", "value"),
        State("hour-select", "value")
    ]
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
                raw_data_select.append(df[(
                    (df.segment == seg) & 
                    (df.side_of_street == sos) &
                    (df.day == day_select) &
                    df.hour.isin(hour_select)
                )])

    if raw_data_select:
        raw_data_select = pd.concat(raw_data_select)
    else:
        raw_data_select = pd.DataFrame(data=df, columns=df.columns)

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
        filter_action="native",
        sort_action="native",
        style_cell={
            "textOverflow": "ellipsis",
            "background-color": "#242a3b",
            "color": "#7b7d8d",
        },
        sort_mode="multi",
        page_size=50,
        style_as_list_view=False,
        style_header={"background-color": "#1f2536", "padding": "2px 12px 0px 12px"},
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
        df=data[data.day == day_select],
        aggregation=metric_aggregation,
        hour_selected=hour_select
    )

    return generate_geo_map(day_hour_agg_data, metric_select)


if __name__ == "__main__":
    app.run_server(debug=True)