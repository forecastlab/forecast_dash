import dash

from common import markdown_layout

title = "Methodology"

dash.register_page(
    __name__, 
    title=title,
)

### markdown content ###
markdown = """
# Data

Data are sourced from:
- [Australian Macro Database](http://ausmacrodata.org)
- [St Louis Federal Reserve](https://api.stlouisfed.org)
- [UK Office of National Statistics](https://www.ons.gov.uk)
- [World Bank](https://data.worldbank.org/)

# Methodology

The available models are listed in the [Leaderboard](/leaderboard/).
These are based on the benchmark models used in the M4 Competition \[0\].

The models are run on each dataset according to the time series cross-validation
scheme described in \[1\], Sect 3.4. The forecast horizon depends upon the frequency of the
underlying time series, that is 6 for yearly, 8 for quarterly and 18 for monthly data.

![time series cross-validation](https://otexts.com/fpp2/fpp_files/figure-html/cv1-1.png)
\(Image reproduced from \[1\] with permission.\)

The forecast accuracy or cross-validation score is computed by averaging
the mean-squared forecast error over the test sets and forecast horizons.
The model with the best forecast accuracy is selected by the Forecast Lab
as the preferred model. Forecasts from the other available models may be
selected from the drop-down menu in each Series page.


\[0\] Makridakis, S., Spiliotis, E. and Assimakopoulos, V.,
      _The M4 Competition: 100,000 time series and 61 forecasting methods,_
      Int. J. Forecasting 36 \(2020\) 54-74

\[1\] Hyndman, R. and Athanasopoulos, G.,
      _Forecasting: Principles and Practice_
      OTexts: Melbourne, Australia.
      [otexts.com/fpp2/](https://otexts.com/fpp2/)

"""

layout = markdown_layout(
    title, markdown
)