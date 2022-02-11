import altair as alt
import pandas as pd
import numpy as np

SERIES_ALL = [
    "training_accuracy",
    "training_loss",
    "validation_accuracy",
    "validation_loss",
]

SERIES_ACC = [
    "training_accuracy",
    "validation_accuracy",
]

SERIES_LOSS = [
    "training_loss",
    "validation_loss",
]


def model_state_to_chart_sources(model_state, series=SERIES_ALL, title="value"):
    source = []
    for serie in series:
        values = model_state.get(serie)
        for epoch, value in enumerate(values):
            source.append(
                {
                    "Serie": serie.split("_")[0],
                    "epoch": epoch + 1,
                    f"{title}": value,
                }
            )

    return alt.InlineData(values=source)


def source_to_chart(source, title="value", height=300, use_percent=True):
    nearest = alt.selection(
        type="single", nearest=True, on="mouseover", fields=["epoch"], empty="none"
    )

    yAxisDefault = alt.Y(f"{title}:Q")
    yAxisPercent = alt.Y(f"{title}:Q", axis=alt.Axis(format="%"))
    yAxis = yAxisPercent if use_percent else yAxisDefault

    line = (
        alt.Chart(source)
        .mark_line()
        .encode(
            alt.X("epoch:Q"),
            yAxis,
            color="Serie:N",
        )
    )

    selectors = (
        alt.Chart(source)
        .mark_point()
        .encode(
            x="epoch:Q",
            opacity=alt.value(0),
        )
        .add_selection(nearest)
    )

    # Draw points on the line, and highlight based on selection
    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    # Draw text labels near the points, and highlight based on selection
    text = line.mark_text(align="left", dx=10, dy=-10).encode(
        text=alt.condition(nearest, f"{title}:Q", alt.value(" ")),
    )

    # Draw a rule at the location of the selection
    rules = (
        alt.Chart(source)
        .mark_rule(color="gray")
        .encode(
            x="epoch:Q",
        )
        .transform_filter(nearest)
    )

    # Put the five layers into a chart and bind the data
    return alt.layer(line, selectors, points, rules, text).properties(
        width="container", height=height
    )


def to_chart(model, series=SERIES_ALL, title="value", height=300, use_percent=True):
    source = model_state_to_chart_sources(model, series, title)
    return source_to_chart(source, title=title, height=height, use_percent=use_percent)


def acc_loss_charts(model_state):
    acc = to_chart(model_state, series=SERIES_ACC, title="Accuracy", use_percent=True)
    loss = to_chart(model_state, series=SERIES_LOSS, title="Loss", use_percent=False)
    return acc, loss


def prediction_chart(prediction_list):
    df = pd.DataFrame(
        {
            "Score": prediction_list,
            "Class": list(range(10)),
        }
    )
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Score", axis=alt.Axis(title=None)),
            y=alt.Y("Class:O", axis=alt.Axis(title="Labels"), sort="-x"),
        )
        .properties(width="container", height=200)
    )
    return chart


def class_accuracy(matrix):
    confusion_matrix = matrix.copy()
    for i in range(10):
        ratio = np.sum(confusion_matrix[i])
        confusion_matrix[i] *= 100 / ratio

    confusion_matrix = np.around(confusion_matrix)

    x = [confusion_matrix[i, i] for i in range(10)]

    df = pd.DataFrame(
        {
            "acc": x,
            "class": list(range(10)),
        }
    )

    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            alt.X("acc:Q", axis=alt.Axis(title="Accuracy")),
            alt.Y("class:O", axis=alt.Axis(title="Classes")),
        )
    )
    text = bars.mark_text(
        align="left",
        baseline="middle",
        dx=3,  # Nudges text to right so it doesn't appear on top of the bar
    ).encode(text="acc:Q")

    return (bars + text).properties(width="container", height=500)


def confusion_matrix_chart(matrix):
    x, y = np.meshgrid(range(0, 10), range(0, 10))
    df = pd.DataFrame(
        {
            "x": x.ravel(),
            "y": y.ravel(),
            "z": matrix.ravel(),
        }
    )

    chart = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X("x:O", axis=alt.Axis(title="Ground truth")),
            y=alt.Y("y:O", axis=alt.Axis(title="Classification")),
            color=alt.Color(
                "z:Q",
                title="",
                scale=alt.Scale(
                    scheme="inferno",
                    domain=[-15, np.amax(matrix)],
                ),
            ),
            # tooltip=[
            #     alt.Tooltip('z:Q', title='%')
            # ],
        )
    )

    text = chart.mark_text(baseline="middle",).encode(
        text=alt.condition(
            alt.datum.z == 0,
            alt.value(""),
            alt.Text("z:Q"),
        ),
        color=alt.condition(alt.datum.z < 20, alt.value("white"), alt.value("black")),
    )

    return (chart + text).properties(width="container", height=500)
