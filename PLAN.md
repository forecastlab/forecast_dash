# UK ONS Series Update Plan

## Problem

All configured UK ONS series currently use the old endpoint:

```text
https://api.ons.gov.uk/timeseries/{id}/dataset/{dataset}/data
```

Live checks show those URLs now return `404`, including UK inflation:

```text
https://api.ons.gov.uk/timeseries/czbh/dataset/mm23/data
```

The ONS beta API works when called with the public ONS content URI:

```text
https://api.beta.ons.gov.uk/v1/data?uri=/economy/inflationandpriceindices/timeseries/czbh/mm23
```

The beta response is close to the old shape: it still has top-level frequency
buckets such as `years`, `quarters`, and `months`, with records containing
`date`, `value`, `year`, `month`, `quarter`, `sourceDataset`, and `updateDate`.

## Current Code

ONS downloads are handled by `Ons.download` in:

```text
updater/download.py
```

The configured ONS series live in:

```text
shared_config/data_sources.json
```

The current downloader expects JSON keys like `months` or `quarters`, then
normalizes the result to a dataframe indexed by `date` with one numeric `value`
column. That normalized contract should remain unchanged.

## Fix Plan

1. Add ONS beta URL support in `Ons.download`.

   Keep the normalized output unchanged:

   ```text
   date index + numeric value column
   ```

   Select the response bucket from `frequency`:

   ```text
   MS -> months
   Q  -> quarters
   Y or YS -> years
   ```

   Parse beta monthly dates such as:

   ```text
   2026 MAY
   ```

   Parse beta quarterly dates such as:

   ```text
   2025 Q4
   ```

   Drop blank or non-numeric values safely, sort by date, and set:

   ```text
   df.index.name = "date"
   ```

2. Update all ONS URLs in `shared_config/data_sources.json`.

   UK Inflation (RPI):

   ```text
   https://api.beta.ons.gov.uk/v1/data?uri=/economy/inflationandpriceindices/timeseries/czbh/mm23
   ```

   UK Unemployment:

   ```text
   https://api.beta.ons.gov.uk/v1/data?uri=/employmentandlabourmarket/peoplenotinwork/unemployment/timeseries/mgsx/lms
   ```

   UK GDP Growth:

   ```text
   https://api.beta.ons.gov.uk/v1/data?uri=/economy/grossdomesticproductgdp/timeseries/ihyq/qna
   ```

   UK Current Account Balance:

   ```text
   https://api.beta.ons.gov.uk/v1/data?uri=/economy/nationalaccounts/balanceofpayments/timeseries/hbop/pnbp
   ```

   UK Average Weekly Earnings Growth:

   ```text
   https://api.beta.ons.gov.uk/v1/data?uri=/employmentandlabourmarket/peopleinwork/earningsandworkinghours/timeseries/kac3/lms
   ```

   UK Retail Sales Index:

   ```text
   https://api.beta.ons.gov.uk/v1/data?uri=/businessindustryandtrade/retailindustry/timeseries/j5c4/drsi
   ```

3. Preserve backward compatibility if useful.

   Prefer updating `Ons.download` so it can handle both the old and beta
   response shapes. The normalized output is the same, so a separate `OnsBeta`
   class is probably unnecessary unless we want a clean cutover.

4. Add focused tests.

   Add tests for:

   - Beta monthly parsing with a small `months` fixture.
   - Beta quarterly parsing with a small `quarters` fixture.
   - Empty or mismatched bucket handling, for example a monthly frequency
     requested from a response with no monthly observations.
   - Mocked `requests.get`, so tests do not depend on live ONS availability.

5. Verify with a live smoke test.

   Run only the six configured ONS downloads into a temporary directory.

   Confirm each pickle contains:

   - a non-empty dataframe
   - a datetime index named `date`
   - a numeric `value` column
   - the expected latest observation from the beta endpoint

   Current live observations checked on 2026-06-23 include:

   - UK Inflation (RPI): latest monthly observation is `2026 MAY = 3.1`
   - ONS public page release date: `17 June 2026`
   - ONS public page next release: `22 July 2026`

6. Optional adjacent cleanup.

   `DataSource.data_versioning()` appears to read:

   ```text
   {download_path}/{self.title}.pkl
   ```

   while downloads are written using the slugified filename:

   ```text
   {download_path}/{self.filename}.pkl
   ```

   That likely prevents stable version tracking. This is separate from the ONS
   API fix, but worth addressing while touching downloader code.

## References

- https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/czbh/mm23
- https://api.beta.ons.gov.uk/v1/data?uri=/economy/inflationandpriceindices/timeseries/czbh/mm23
