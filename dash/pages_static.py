import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

from common import MarkdownApp, BootstrapApp, header, breadcrumb_layout


class Methodology(MarkdownApp):

    title = "Methodology"

    markdown = """
# Methodology

**This page is under construction.**

It will contain the description of the models and other aspects of the
methodology used to forecast the time series.

While we are busy with this document, we recommend “Forecasting: Principles
and Practice” textbook freely available at
[otexts.com/fpp2/](https://otexts.com/fpp2/)
    """


class About(BootstrapApp):

    title = "About"

    def setup(self):

        self.layout = html.Div(
            header
            + [
                dcc.Location(id="url", refresh=False),
                dbc.Container(
                    [
                        breadcrumb_layout(
                            [("Home", "/"), (f"{self.title}", "")]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.H2("Our Mission"),
                                        html.Ol(
                                            [
                                                html.Li(
                                                    "To make forecasting models accessible to everyone."
                                                ),
                                                html.Li(
                                                    "To provide the latest financial and economic forecasts of the commonly used time series."
                                                ),
                                            ]
                                        ),
                                    ],
                                    lg=12,
                                )
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.H2("About"),
                                        html.P(
                                            "The Business Forecast Lab was established in ...."
                                        ),
                                    ],
                                    lg=12,
                                )
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.H2("Members"),
                                        html.P(
                                            "The Business Forecast Lab was established in ...."
                                        ),
                                        html.H4(
                                            "Andrey Vasnev", className="mt-3"
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Img(
                                                            src="https://business.sydney.edu.au/__data/assets/image/0006/170556/vasnev.png",
                                                            height="200px",
                                                        )
                                                    ],
                                                    lg=2,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.P(
                                                            "Andrey Vasnev (Perm, 1976) graduated in Applied Mathematics from Moscow State University in 1998. In 2001 he completed his Master's degree in Economics in the New Economic School, Moscow. In 2006 he received Ph.D. degree in Economics from the Department of Econometrics and Operations Research at Tilburg University under the supervision of Jan R. Magnus. He worked as a credit risk analyst in ABN AMRO bank before joining the University of Sydney."
                                                        ),
                                                        html.A(
                                                            "https://business.sydney.edu.au/staff/andrey.vasnev",
                                                            href="https://business.sydney.edu.au/staff/andrey.vasnev",
                                                        ),
                                                    ],
                                                    lg=9,
                                                ),
                                            ]
                                        ),
                                        html.H4(
                                            "Richard Gerlach", className="mt-3"
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Img(
                                                            src="https://business.sydney.edu.au/__data/assets/image/0003/170553/RichardGerlach.jpg",
                                                            height="200px",
                                                        )
                                                    ],
                                                    lg=2,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.P(
                                                            "Richard Gerlach's research interests lie mainly in financial econometrics and time series. His work has concerned developing time series models for measuring, forecasting and managing risk in financial markets as well as computationally intensive Bayesian methods for inference, diagnosis, forecasting and model comparison for these models. Recent focus has been on nonlinear threshold heteroskedastic models for volatility, Value-at-Risk and Expected Shortfall forecasting. He has developed structural break and intervention detection tools for use in state space models; also has an interest in estimating logit models incorporating misclassification and variable selection. His applied work has involved forecasting risk levels during and after the Global Financial Crisis; assessing asymmetry in major international stock markets, in response to local and exogenous factors; co-integration analysis assessing the effect of the Asian financial crisis on long term relationships between international real estate investment markets; stock selection for financial investment using logit models; option pricing and hedging involving barriers; and factors influencing the 2004 Federal election."
                                                        ),
                                                        html.P(
                                                            "His research papers have been published in Journal of the American Statistical Association, Journal of Business and Economic Statistics, Journal of Time Series Analysis and the International Journal of Forecasting. He has been an invited speaker and regular presenter at international conferences such as the International conference for Computational and Financial Econometrics, the International Symposium on Forecasting and the International Statistical Institute sessions."
                                                        ),
                                                        html.A(
                                                            "https://business.sydney.edu.au/staff/richard.gerlach",
                                                            href="https://business.sydney.edu.au/staff/richard.gerlach",
                                                        ),
                                                    ],
                                                    lg=9,
                                                ),
                                            ]
                                        ),
                                        html.H4("Chao Wang", className="mt-3"),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Img(
                                                            src="https://business.sydney.edu.au/__data/assets/image/0012/279678/wang.jpg",
                                                            height="200px",
                                                        )
                                                    ],
                                                    lg=2,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.P(
                                                            "Dr Chao Wang received his PhD degree in Econometrics from The University of Sydney. He has two master degrees major in Machine Learning & Data Mining from Helsinki University of Technology and Mechatronic Engineering from Beijing Institute of Technology respectively."
                                                        ),
                                                        html.P(
                                                            "Chao Wang’s main research interests are financial econometrics and time series modelling. He has developed a series of parametric and non-parametric volatility models incorporating intra-day and high frequency volatility measures (realized variance, realized range, etc) applied on the financial market risk forecasting, employing Bayesian adaptive Markov chain Monte Carlo estimation. His work has also considered different techniques, including scaling and sub-sampling, to deal with the micro-structure noisy of the high frequency volatility measures. Further, Chao’s research interests also include big data, machine learning and data mining, text mining, etc."
                                                        ),
                                                        html.A(
                                                            "https://business.sydney.edu.au/staff/chao.wang",
                                                            href="https://business.sydney.edu.au/staff/chao.wang",
                                                        ),
                                                    ],
                                                    lg=9,
                                                ),
                                            ]
                                        ),
                                    ],
                                    lg=12,
                                )
                            ]
                        ),
                    ],
                    className="mb-5",
                ),
            ]
        )
