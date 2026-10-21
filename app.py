from __future__ import annotations

import logging
import time

import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

import config
import geofilter
from pipeline import Receiver
from tracker import TrackedView

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
BORDER = "#e4e3de"


def _fmt(value, fmt: str, unit: str = "") -> str:
    return "—" if value is None else f"{value:{fmt}}{unit}"


def _position_text(v: TrackedView) -> str:
    if v.position_age_s is None:
        return "none"
    if v.estimated:
        return f"estimated, last fix {v.position_age_s:.0f} s ago"
    return "live"


def _visible(views: list[TrackedView], show_other: bool) -> list[TrackedView]:
    return [v for v in views if show_other or v.classification.is_commercial]


def build_figure(views: list[TrackedView], show_other: bool = True) -> go.Figure:
    fig = go.Figure()

    ring_lat, ring_lon = geofilter.circle_points()
    fig.add_trace(go.Scattermap(
        lat=ring_lat, lon=ring_lon, mode="lines", name=f"{config.RADIUS_KM:.0f} km range",
        line=dict(color=config.COLOR_RADIUS, width=2), hoverinfo="skip",
    ))
    fig.add_trace(go.Scattermap(
        lat=[config.BASE_LAT], lon=[config.BASE_LON], mode="markers", name=f"Receiver ({config.BASE_NAME})",
        marker=dict(size=10, color=config.COLOR_BASE),
        hovertemplate=f"<b>Receiver</b><br>{config.BASE_NAME}<br>"
                      f"{config.BASE_LAT:.4f}°N {config.BASE_LON:.4f}°E<extra></extra>",
    ))

    on_map = [v for v in _visible(views, show_other) if v.in_range]
    groups = [("Commercial", config.COLOR_COMMERCIAL, [v for v in on_map if v.classification.is_commercial])]
    if show_other:
        groups.append(("Other / unknown", config.COLOR_OTHER,
                       [v for v in on_map if not v.classification.is_commercial]))

    for name, color, group in groups:
        ac = [v.aircraft for v in group]
        fig.add_trace(go.Scattermap(
            lat=[a.lat for a in ac], lon=[a.lon for a in ac], mode="markers",
            name=f"{name} ({len(ac)})",
            marker=dict(size=11, color=color,
                        opacity=[config.ESTIMATED_MARKER_OPACITY if v.estimated else 1.0 for v in group]),
            customdata=[[
                a.callsign or "(no callsign)", a.icao,
                _fmt(a.altitude_ft, ",d", " ft"),
                _fmt(a.speed_kt, ".0f", " kt"),
                _fmt(a.track_deg, ".0f", "°"),
                _fmt(a.vertical_rate_fpm, "+,d", " fpm"),
                _fmt(v.distance_km, ".1f", " km"),
                f"{v.classification.label} ({v.classification.reason})",
                _position_text(v),
            ] for v, a in zip(group, ac)],
            hovertemplate=(
                "<b>%{customdata[0]}</b>  %{customdata[1]}<br>"
                "Altitude: %{customdata[2]}<br>"
                "Speed: %{customdata[3]}  Track: %{customdata[4]}<br>"
                "Vertical rate: %{customdata[5]}<br>"
                "Distance: %{customdata[6]}<br>"
                "Position: %{customdata[8]}<br>"
                "<span style='color:#52514e'>%{customdata[7]}</span><extra></extra>"
            ),
        ))

    fig.update_layout(
        map=dict(style=config.MAP_STYLE, zoom=config.MAP_ZOOM,
                 center=dict(lat=config.BASE_LAT, lon=config.BASE_LON)),
        margin=dict(l=0, r=0, t=0, b=0),
        uirevision="keep",
        paper_bgcolor=SURFACE,
        font=dict(color=TEXT, family="system-ui, -apple-system, Segoe UI, sans-serif"),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(252,252,251,0.9)", bordercolor=BORDER, borderwidth=1),
        hoverlabel=dict(bgcolor="white", font_color=TEXT, bordercolor=BORDER),
    )
    return fig


def build_status(status: dict, views: list[TrackedView]) -> str:
    in_range = [v for v in views if v.in_range]
    commercial = sum(v.classification.is_commercial for v in in_range)
    no_pos = sum(v.aircraft.lat is None for v in views)
    sdr = status["sdr_state"]
    if status.get("sdr_error") and sdr != "connected":
        sdr += f" — {status['sdr_error']} (retrying every {config.SDR_RETRY_SECONDS:.0f} s)"
    return (f"SDR: {sdr}  ·  {status['messages_per_s']:.1f} msg/s  ·  "
            f"{len(in_range)} aircraft ≤ {config.RADIUS_KM:.0f} km ({commercial} commercial)  ·  "
            f"{no_pos} tracked without position  ·  "
            f"CRC ok {status['crc_ok']:,} / fixed {status['crc_fixed']:,} / discarded {status['crc_bad']:,}")


def build_table(views: list[TrackedView], show_other: bool, now: float) -> html.Table:
    rows = sorted((v for v in _visible(views, show_other) if v.in_range or v.aircraft.lat is None),
                  key=lambda v: (v.distance_km is None, v.distance_km or 0))
    head = ["Callsign", "ICAO", "Class", "Altitude", "Speed", "V/S", "Distance", "Seen"]
    cell = {"padding": "4px 10px", "borderBottom": f"1px solid {BORDER}", "whiteSpace": "nowrap"}
    num = {**cell, "textAlign": "right", "fontVariantNumeric": "tabular-nums"}
    body = []
    for v in rows:
        a = v.aircraft
        swatch = config.COLOR_COMMERCIAL if v.classification.is_commercial else config.COLOR_OTHER
        body.append(html.Tr([
            html.Td(a.callsign or "—", style=cell),
            html.Td(a.icao, style={**cell, "fontFamily": "monospace"}),
            html.Td([html.Span(style={"display": "inline-block", "width": 8, "height": 8,
                                      "borderRadius": 4, "background": swatch, "marginRight": 6}),
                     v.classification.label], title=v.classification.reason, style=cell),
            html.Td(_fmt(a.altitude_ft, ",d", " ft"), style=num),
            html.Td(_fmt(a.speed_kt, ".0f", " kt"), style=num),
            html.Td(_fmt(a.vertical_rate_fpm, "+,d"), style=num),
            html.Td(_fmt(v.distance_km, ".1f", " km") + (" (est.)" if v.estimated else "")
                    if a.lat is not None else "no position", style=num),
            html.Td(f"{now - a.last_seen:.0f} s", style=num),
        ]))
    return html.Table(
        [html.Thead(html.Tr([html.Th(h, style={**cell, "textAlign": "left", "color": TEXT_2,
                                               "fontWeight": 600}) for h in head])),
         html.Tbody(body)],
        style={"borderCollapse": "collapse", "fontSize": 13, "width": "100%"},
    )


def create_app(receiver: Receiver) -> Dash:
    app = Dash(__name__, title="ADS-B · Petah Tikva")
    app.layout = html.Div(style={"fontFamily": "system-ui, -apple-system, Segoe UI, sans-serif",
                                 "background": SURFACE, "color": TEXT, "padding": "12px 16px"}, children=[
        html.Div(style={"display": "flex", "flexWrap": "wrap", "alignItems": "baseline",
                        "gap": "8px 24px"}, children=[
            html.H2(f"ADS-B within {config.RADIUS_KM:.0f} km of {config.BASE_NAME}",
                    style={"margin": "0 0 4px 0", "fontSize": 20}),
            dcc.Checklist(id="show-other", inline=True,
                          options=[{"label": " Show non-commercial / unknown", "value": "other"}],
                          value=["other"] if config.SHOW_NON_COMMERCIAL_DEFAULT else []),
        ]),
        html.Div(id="status", style={"color": TEXT_2, "fontSize": 13, "margin": "4px 0 10px"}),
        dcc.Graph(id="map", style={"height": "68vh"}, config={"displaylogo": False, "scrollZoom": True}),
        html.Div(id="table", style={"overflowX": "auto", "marginTop": 12}),
        dcc.Interval(id="tick", interval=config.REFRESH_INTERVAL_MS),
    ])

    @app.callback(Output("map", "figure"), Output("status", "children"), Output("table", "children"),
                  Input("tick", "n_intervals"), Input("show-other", "value"))
    def refresh(_n, show_value):
        now = time.time()
        views = receiver.store.snapshot(now)
        show_other = "other" in (show_value or [])
        return (build_figure(views, show_other),
                build_status(receiver.status(), views),
                build_table(views, show_other, now))

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    receiver = Receiver()
    receiver.start()
    try:
        create_app(receiver).run(host=config.DASH_HOST, port=config.DASH_PORT, debug=config.DASH_DEBUG)
    finally:
        receiver.stop()


if __name__ == "__main__":
    main()
