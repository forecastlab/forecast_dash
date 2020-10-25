import asyncio
from pyppeteer import launch
import time

async def load_page_n_times(browser, n):
    page = await browser.newPage()
    for i in range(n):
        await page.goto('http://0.0.0.0:80/search/')

async def main(n_conns, n_requests):

    browser = await launch()

    tasks = [
        load_page_n_times(browser, n_requests)
        for i in range(n_conns)
    ]

    start = time.time()
    await asyncio.gather(*tasks)
    print(f"Time elapsed: {time.time() - start}")

    await browser.close()


asyncio.run(main(64, 2))