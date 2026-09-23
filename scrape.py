"""Download tagged stills from shot.cafe and record their tags in labels.csv.

Every image is saved once as datasets/images/<id>.<ext>. When the same still
shows up under several tags (e.g. close-up and low-angle), its tags are merged
into one labels.csv row, so labels build themselves as you scrape more tags.

    python scrape.py                          # all tags in common.HEADS
    python scrape.py --tags close-up low-angle --max-pages 5
"""
import argparse
import asyncio
import logging
import os
import re
from urllib.parse import quote

import aiohttp
from bs4 import BeautifulSoup

from common import ALL_TAGS, IMAGE_DIR, read_labels, write_labels

BASE_URL = 'https://shot.cafe'
HEADERS = {'User-Agent': 'shotai-dataset-builder (research)'}

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger('scrape')


def parse_page(html):
    """Return [{'id', 'img_url', 'film'}] for every still on a tag page."""
    soup = BeautifulSoup(html, 'html.parser')
    stills = []
    for link in soup.select('.box a.box-img-load'):
        img = link.find('img')
        alt = img.get('alt', '') if img else ''
        # alt looks like "Still from Pig (2021) that has been tagged with: close-up"
        match = re.match(r'Still from (.+?) that has been tagged with', alt)
        film = match.group(1) if match else link.get('data-img-url', '').split('/')[0]
        stills.append({
            'id': link['data-img-id'],
            'img_url': BASE_URL + link['data-img-img'],
            'film': film,
        })
    return stills


async def fetch(session, limit, url, as_bytes=False, retries=3):
    for attempt in range(retries):
        try:
            async with limit, session.get(url) as response:
                if response.status == 200:
                    return await (response.read() if as_bytes else response.text())
                log.warning(f'{url}: HTTP {response.status}')
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            log.warning(f'{url}: {e}')
        await asyncio.sleep(2 ** attempt)
    return None


async def download(session, limit, still, rows):
    ext = os.path.splitext(still['img_url'])[1] or '.jpg'
    path = os.path.join(IMAGE_DIR, still['id'] + ext)
    if not os.path.exists(path):
        data = await fetch(session, limit, still['img_url'], as_bytes=True)
        if data is None:
            return False
        with open(path, 'wb') as f:
            f.write(data)
    rows.setdefault(still['id'], {'id': still['id'], 'file': os.path.basename(path),
                                  'film': still['film'], 'tags': set()})
    return True


async def scrape_tag(session, limit, tag, max_pages, rows):
    downloaded = 0
    for page in range(1, max_pages + 1):
        html = await fetch(session, limit, f'{BASE_URL}/tag/{quote(tag)}?page={page}')
        stills = parse_page(html) if html else []
        if not stills:
            break
        # Tag already-downloaded stills without fetching them again.
        for still in stills:
            if still['id'] in rows:
                rows[still['id']]['tags'].add(tag)
        new = [s for s in stills if s['id'] not in rows]
        results = await asyncio.gather(*(download(session, limit, s, rows) for s in new))
        for still, ok in zip(new, results):
            if ok:
                rows[still['id']]['tags'].add(tag)
        downloaded += sum(results)
    log.info(f'{tag}: {page} page(s), {downloaded} new image(s)')


async def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tags', nargs='+', default=ALL_TAGS, help='shot.cafe tags to scrape')
    parser.add_argument('--max-pages', type=int, default=1000, help='page limit per tag')
    parser.add_argument('--concurrency', type=int, default=4, help='simultaneous requests')
    args = parser.parse_args()

    os.makedirs(IMAGE_DIR, exist_ok=True)
    rows = read_labels()
    limit = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(headers=HEADERS, timeout=timeout) as session:
        for tag in args.tags:
            await scrape_tag(session, limit, tag, args.max_pages, rows)
            write_labels(rows)  # save progress after every tag
    log.info(f'labels.csv now has {len(rows)} image(s)')


if __name__ == '__main__':
    asyncio.run(main())
