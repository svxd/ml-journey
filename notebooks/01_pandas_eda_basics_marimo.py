import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    z = [1, 2, 3]
    x = max(z)
    """)
    return


if __name__ == "__main__":
    app.run()
