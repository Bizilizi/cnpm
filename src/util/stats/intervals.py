import typing as t

import numpy as np
import plotly.graph_objects as go
import scipy.stats as st


def _draw_intervals(intervals: np.ndarray, x_axis: t.Optional[np.ndarray] = None):
    """
    Support function that draws intervals

    @param intervals
    """

    fig = go.Figure(
        [
            go.Scatter(
                name="Measurement",
                y=intervals[:, 1],
                x=x_axis,
                mode="lines",
                line=dict(color="rgb(31, 119, 180)"),
            ),
            go.Scatter(
                name="Upper Bound",
                y=intervals[:, 2],
                x=x_axis,
                mode="lines",
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False,
            ),
            go.Scatter(
                name="Lower Bound",
                y=intervals[:, 0],
                x=x_axis,
                marker=dict(color="#444"),
                line=dict(width=0),
                mode="lines",
                fillcolor="rgba(68, 68, 68, 0.3)",
                fill="tonexty",
                showlegend=False,
            ),
        ]
    )

    return fig


def trust_interval(
    data: np.ndarray,
    x_axis: t.Optional[np.ndarray] = None,
    alpha: float = 0.95,
    title: str = "Trust intervals",
    y_title: str = "Measurement",
    x_title: str = "Argument",
) -> go.Figure:
    """
    Draws trust intervals for data samples

    @param data: ND array with shape [n, y]. Where statistics calculated over n axis
    @param x_axis: Optional. ND array for x axis values with shape [y]
    @param alpha: Trust interval alpha. Default 0.95
    @param title: Figure title
    @param y_title: Figure Y axis title
    @param x_title: Figure X axis title
    @return: Instance of plotly Figure
    """

    intervals = []
    for i in range(1, data.shape[-1]):
        sample = data[:, i]
        mean = np.mean(sample)
        bottom, top = st.t.interval(
            alpha=alpha,
            df=len(sample) - 1,
            loc=mean,
            scale=st.sem(sample),
        )

        intervals.append([bottom, mean, top])

    intervals = np.array(intervals)

    fig = _draw_intervals(intervals, x_axis)
    fig.update_layout(
        yaxis_title=y_title, xaxis_title=x_title, title=title, hovermode="x"
    )

    return fig


def min_max_interval(
    data: np.ndarray,
    x_axis: t.Optional[np.ndarray] = None,
    title: str = "Min/Max intervals",
    y_title: str = "Measurement",
    x_title: str = "Argument",
) -> go.Figure:
    """
    Draws min max intervals for data samples

    @param data: ND array with shape [x, y]. Where statistics calculated over x axis
    @param x_axis: Optional. ND array for x axis values with shape [y]
    @param alpha: Trust interval alpha. Default 0.95
    @param title: Figure title
    @param y_title: Figure Y axis title
    @param x_title: Figure X axis title
    @return: Instance of plotly Figure
    """

    intervals = []
    for i in range(1, data.shape[-1]):
        sample = data[:, i]
        bottom, top = sample.min(), sample.max()
        mean = (bottom + top) / 2

        intervals.append([bottom, mean, top])

    intervals = np.array(intervals)

    fig = _draw_intervals(intervals)
    fig.update_layout(
        yaxis_title=y_title, xaxis_title=x_title, title=title, hovermode="x"
    )

    return fig


def raw_lines(
    data: np.ndarray,
    num_samples: int = 50,
    title: str = "Sampled lines",
    y_title: str = "Measurement",
    x_title: str = "Argument",
):
    sub_idx = np.random.choice(
        np.arange(data.shape[0]), size=min(num_samples, data.shape[0]), replace=False
    )
    sub_samples = data[sub_idx]

    fig = go.Figure(
        data=[
            go.Scatter(y=sub_samples[i], showlegend=False)
            for i in range(sub_samples.shape[0])
        ]
    )
    fig.update_layout(
        yaxis_title=y_title, xaxis_title=x_title, title=title, hovermode="x"
    )

    return fig
