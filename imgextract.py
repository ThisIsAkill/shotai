import os
import asyncio
import aiohttp
from aiohttp import ClientSession
from bs4 import BeautifulSoup
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Global counters for stats
success_count = 0
error_count = 0

# Function to download and save an image asynchronously
async def download_image(session, img_url, save_path):
    global success_count, error_count
    retries = 3
    for attempt in range(retries):
        try:
            async with session.get(img_url) as response:
                if response.status == 200:
                    with open(save_path, 'wb') as file:
                        while True:
                            chunk = await response.content.read(1024)
                            if not chunk:
                                break
                            file.write(chunk)
                    logging.info(f"Downloaded {img_url} to {save_path}")
                    success_count += 1
                    return
                else:
                    logging.warning(f"Failed to download {img_url}, status code: {response.status}")
        except Exception as e:
            logging.error(f"Error downloading {img_url}: {e}")
        await asyncio.sleep(1)
    logging.error(f"Failed to download {img_url} after {retries} attempts")
    error_count += 1

# Function to scrape images from a page asynchronously
async def scrape_page(session, base_url, page_number, path_to_save):
    url = f'{base_url}{page_number}'
    try:
        async with session.get(url) as response:
            if response.status == 200:
                html_content = await response.text()
                soup = BeautifulSoup(html_content, 'html.parser')
                image_elements = soup.select('.box a.box-img-load')

                tasks = []
                for image_element in image_elements:
                    img_url = 'https://shot.cafe' + image_element.get('data-img-img')
                    img_id = image_element.get('data-img-id')
                    img_extension = os.path.splitext(img_url)[1]
                    save_path = os.path.join(path_to_save, f'{img_id}{img_extension}')

                    if not os.path.exists(save_path):
                        tasks.append(download_image(session, img_url, save_path))

                await asyncio.gather(*tasks)
                logging.info(f"Finished scraping page {page_number}")
            else:
                logging.warning(f"Failed to fetch {url}, status code: {response.status}")
    except Exception as e:
        logging.error(f"Error scraping page {page_number}: {e}")

# Main function to loop through pages and scrape images asynchronously
async def main():
    global success_count, error_count

    # Get user inputs for the base URL, directory name, and page range
    base_url = input("Enter the base URL (e.g., https://shot.cafe/tag/close-up?page=): ")
    directory_name = input("Enter the name of the directory to create in the datasets folder: ")
    start_page = int(input("Enter the start page number: "))
    end_page = int(input("Enter the end page number: "))

    # Define the path to save the images
    path_to_save = os.path.join('datasets', directory_name)

    # Ensure the save directory exists
    os.makedirs(path_to_save, exist_ok=True)

    async with ClientSession() as session:
        tasks = [scrape_page(session, base_url, page_number, path_to_save) for page_number in range(start_page, end_page + 1)]
        await asyncio.gather(*tasks)

    # Print stats
    print('All images have been downloaded.')
    print(f'Successfully downloaded images: {success_count}')
    print(f'Failed downloads: {error_count}')

# Run the main function
asyncio.run(main())
